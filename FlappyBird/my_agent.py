import pygame
from pygame.locals import *
import random
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim


import pickle

# All agent must provide a method .action(env,not_crash,...)
#   - `env` is the raw input coming from the game state. inp = x,y
#       where x = list of horizontal distance to all pipes
#             y = list of vertical distance to all lower pipes

class RandomAgent():
    def Action(self,env,score,not_crash):
        action = random.random() >= 0.93
        return action

class HumanAgent():
    def Action(self,env,score,not_crash):
        action = 0
        for event in pygame.event.get():
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                action = 1
        return action

######################################################################################
###################################### ANNAgent ######################################
######################################################################################

#The Network agent learns the Q-value
class baselineANN(nn.Module):
    def __init__(self):
        super(baselineANN, self).__init__()
        self.name = "baseline"
        
        # Input has 2 neurons:
        #   - Horizontal distance to 1st pipe
        #   - Vertical distance to 1st lower pipe
        # Output has 2 neurons (each represents the Q-value of 1 action)
        self.FC = nn.Linear(2, 2)
        
    def forward(self, network_inp):
        return self.FC(network_inp)
        
class ANN(nn.Module):
	#TODO: add configurability e.g. num_layers, num_neurons in each layer
    def __init__(self):
        super(ANN, self).__init__()
        self.name = "ANN_32_16"
        
        # Input has 4 neurons:
        #   - Horizontal distance to 1st and 2nd pipes
        #   - Vertical distance to 1st and 2nd lower pipes
        # Output has 2 neurons (each represents the Q-value of 1 action)
        inp_size = 4
        out_size = 2
        self.FC = nn.Sequential(
            nn.Linear(inp_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_size)
        )
        
    def forward(self, network_inp):
        return self.FC(network_inp)

class ANNAgent():
    def __init__(self,is_baseline):
        # 1. Const values
        
        # 1.1. Const values for RL framework
        self.gamma = 0.99 # discounted coefficient
        self.init_epsilon = 0.1 # probability to take random action, decrease (anneal) over time
        self.final_epsilon = 0.0001 # prob for random action won't decrease under this value
        self.epsilon_anneal_rate = (self.init_epsilon-self.final_epsilon)/10000
        self.replay_mem_size = 10000 # experience replay
        self.batch_size = 32 # minibatch size sampled randomly from replay_mem
        
        # 1.2. Const values for network
        self.learning_rate = 3e-4
        self.is_baseline = is_baseline
        self.model = baselineANN() if is_baseline else ANN()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 2. Values depending on the game state
        #     Here, the game state is simply the network input e.g. distance(s) to the pipe(s)
        
        # Store tuples of (prev_state,prev_action,cur_state,cur_reward), where `prev_action` is the action that
        #   causes the transition from prev_state to cur_state
        self.replay_mem = []
        self.action = None
        self.epsilon = self.init_epsilon
        self.iter = 0 #To determine model checkpoint
        
        self.score = 0
        
        # 3. Values for model checkpoint
        self.checkpoint_dir = os.path.dirname(os.path.realpath(__file__))+"/model_checkpoint"
        self.model_file = self.checkpoint_dir+"/"+self.model.name+"_bs_"+str(self.batch_size)
        self.replay_mem_file = self.checkpoint_dir+"/replay_mem"
        self.loss_history = []
        self.loss_history_file = self.checkpoint_dir+"/loss_history"
      	
      	# Load model if exists
        if os.path.exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            self.epsilon = self.final_epsilon
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Load replay_mem if exists
        if os.path.exists(self.replay_mem_file):
            with open (self.replay_mem_file, 'rb') as f:
                self.replay_mem = pickle.load(f)
        
        # Load loss_history if exists
        #if os.path.exists(self.loss_history_file):
        #    with open (self.loss_history_file, 'rb') as f:
        #        self.loss_history = pickle.load(f)
        
        #temp variable, use only once at the 1st iter
        self.init_state = None
    
    def Action(self,env,score,not_crash,is_train=False):
        if self.iter == 0:
            #Fist iteration, just save the init_state
            self.init_state = self._get_network_input(env)
            self.action = 0
        elif self.iter == 1:
            cur_state = self._get_network_input(env)
            self.replay_mem.append((self.init_state,0,cur_state,0.1))
            if len(self.replay_mem) >= self.replay_mem_size:
                self.replay_mem.pop(0)
        else:
            # Update replay memory
            cur_state = self._get_network_input(env)
            prev_state = self.replay_mem[-1][2] #cur_state of last iteration
            prev_reward = self.replay_mem[-1][3]
            
            cur_reward = 1
            if not_crash:
                if score > self.score:
                    cur_reward = 100
            else:
                cur_reward = -100
            
            prev_action = self.action
            
            self.replay_mem.append((prev_state,prev_action,cur_state,cur_reward))
            if len(self.replay_mem) >= self.replay_mem_size:
                self.replay_mem.pop(0)
            
            # Decide on next action
            if random.random() <= self.epsilon:
                #Exploration: With probability epsilon, choose random action
                self.action = random.randint(0,1)
            else:
                #Exploitation: Choose optimal action
                network_out = self._get_network_output(cur_state)
                self.action = torch.argmax(network_out)
            # Epsilon annealing
            self.epsilon = max(self.epsilon-self.epsilon_anneal_rate, self.final_epsilon)
            
            if is_train:
                # Sample a random minibatch
                batch_replay_mems = random.sample(self.replay_mem, min(len(self.replay_mem), self.batch_size))
                batch_prev_states,batch_actions,batch_cur_states,batch_cur_rewards = self._unpack_replay_mem(batch_replay_mems)
                
                #q_predict = Q(s,a)
                #          = model(prev_states)[:,a]
                batch_q_predict = self.model(batch_prev_states).gather(-1,batch_actions.view(-1,1)).squeeze()
                #q_actual = r + gamma * max_a'(Q(s',a'))
                #         = cur_reward + gamma * model(cur_state).max(dim=1)
                batch_q_actual = batch_cur_rewards + self.gamma * torch.max(self.model(batch_cur_states),dim=1)[0].detach()
                
                
                #model training
                self.optimizer.zero_grad()
                batch_loss = self.criterion(batch_q_predict, batch_q_actual)
                batch_loss.backward()
                self.optimizer.step()
                
                #Saving stuffs:
                #if not not_crash:
                    #Save training loss across the whole replay_ mem at the end of every game
                    #self.loss_history.append(batch_loss)
                
                if self.iter %10000 == 0:
                    # Model checkpoint
                    torch.save(self.model.state_dict(), self.model_file)
                    with open(self.replay_mem_file, 'wb') as f:
                        pickle.dump(self.replay_mem, f)
                    #with open(self.loss_history_file, 'wb') as f:
                    #    pickle.dump(self.loss_history, f)
                
            #if self.iter %10000 == 0:
            #    # Replay memory
            #    with open(self.replay_mem_file, 'wb') as f:
            #        pickle.dump(self.replay_mem, f)
        
        self.iter += 1
        self.score = score
        return self.action
    
    def _get_network_output(self,network_inp):
        network_inp = torch.tensor(network_inp).float()
        network_inp = torch.unsqueeze(network_inp,0)
        if torch.cuda.is_available():
            network_inp = network_inp.cuda()
        network_out = self.model(network_inp)
        network_out = torch.squeeze(network_out)
        return network_out
    
    def _get_network_input(self,env):
        '''`env` is the raw input from the game states i.e. distances to all visible pipes
            We need to preprocess this into the form that our model needs
            i.e. only distance to the 1st pipe for baseline model, distances to 2 nearest pipes for actual model
        '''
        x,y = env # raw input
        if x[0] > 150:
            offset = x[0] - 150
            x = [x[i] - offset for i in range(len(x)) ]
        if self.is_baseline:
            return [x[0],y[0]]
        return [x[0],y[0],x[1],y[1]]
    
    def _unpack_replay_mem(self,replay_mems):
        '''Given a list of replays, unpack it into a list of prev_states, actions, cur_states, cur_rewards
        '''
        prev_states = torch.tensor([replay[0] for replay in replay_mems]).float()
        actions = torch.tensor([replay[1] for replay in replay_mems])
        cur_states = torch.tensor([replay[2] for replay in replay_mems]).float()
        cur_rewards = torch.tensor([replay[3] for replay in replay_mems]).float()
        if torch.cuda.is_available():
            prev_states = prev_states.cuda()
            actions = actions.cuda()
            cur_states = cur_states.cuda()
            cur_rewards = cur_rewards.cuda()
        return prev_states,actions,cur_states,cur_rewards
