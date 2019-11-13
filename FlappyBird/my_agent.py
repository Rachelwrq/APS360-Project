import pygame
from pygame.locals import *
import random

import torch
import torch.nn as nn
import torch.optim as optim

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
        self.name = "baselineANN"
        
        # Input has 2 neurons:
        #   - Horizontal distance to 1st pipe
        #   - Vertical distance to 1st lower pipe
        # Output has 2 neurons (each represents the Q-value of 1 action)
        self.FC = nn.Linear(3, 2)
        
    def forward(self, network_inp):
        return self.FC(network_inp)
        
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.name = "ANN"
        
        # Input has 4 neurons:
        #   - Horizontal distance to 1st and 2nd pipes
        #   - Vertical distance to 1st and 2nd lower pipes
        # Output has 2 neurons (each represents the Q-value of 1 action)
        inp_size = 5
        out_size = 2
        self.FC = nn.Sequential(
            nn.Linear(inp_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_size)
        )
        
    def forward(self, network_inp):
        return self.FC(network_inp)

class ANNAgent():
    def __init__(self,is_baseline):
        # 1. Const values
        # 1.1. Const values for network
        self.model = baselineANN() if is_baseline else ANN()
        #TODO
        #if exists model checkpoint:
        #   self.model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.is_baseline = is_baseline
        self.criterion = nn.MSELoss()
        lr = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 1.2. Const values for RL framework
        self.gamma = 0.99 # discounted coefficient
        self.init_epsilon = 0.1 # probability to take random action, decrease (anneal) over time
        self.final_epsilon = 0.0001 # prob for random action won't decrease under this value
        self.epsilon_anneal_rate = (self.init_epsilon-self.final_epsilon)/10000
        self.replay_mem_size = 10000 # experience replay
        self.batch_size = 32 # minibatch size sampled randomly from replay_mem
        
        # 2. Values depending on the game state
        #     Here, the game state is simply the network input e.g. distance(s) to the pipe(s)
        
        # Store tuples of (prev_state,prev_action,cur_state,cur_reward), where `prev_action` is the action that
        #   causes the transition from prev_state to cur_state
        self.replay_mem = []
        self.action = None
        self.epsilon = self.init_epsilon
        self.iter = 0 #To determine model checkpoint
        
        self.score = 0
        
        #temp variable, use only once at the 1st iter
        self.init_state = None
        
    
    def Action(self,env,score,not_crash,is_train=True):
        if self.iter == 0:
            #Fist iteration, just save the init_state
            self.init_state = self._get_network_input(env)
            self.action = 0
        elif self.iter == 1:
            cur_state = self._get_network_input(env)
            self.replay_mem.append((self.init_state,0,cur_state,1))
        else:
            # Update replay memory
            cur_state = self._get_network_input(env)
            prev_state = self.replay_mem[-1][2] #cur_state of last iteration
            prev_reward = self.replay_mem[-1][3]
            if not_crash:
                cur_reward = prev_reward + 1
                if score > self.score:
                    cur_reward = prev_reward + 100
            else:
                cur_reward = 0
            
            prev_action = self.action
            
            self.replay_mem.append((prev_state,prev_action,cur_state,cur_reward))
            if len(self.replay_mem) == self.replay_mem_size:
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
                batch = random.sample(self.replay_mem, min(len(self.replay_mem), self.batch_size))
                #Unpack value
                batch_prev_state = torch.tensor([replay[0] for replay in batch]).float()
                batch_cur_state = torch.tensor([replay[2] for replay in batch]).float()
                batch_cur_reward = torch.tensor([replay[3] for replay in batch]).float()
                if torch.cuda.is_available():
                    batch_prev_state = batch_prev_state.cuda()
                    batch_cur_state = batch_cur_state.cuda()
                    batch_cur_reward = batch_cur_reward.cuda()
                
                #q_actual = r + gamma * max_a'(Q(s',a'))
                #q_predict = Q(s,a)
                q_predict = torch.max(self.model(batch_prev_state),dim=1)[0]
                q_actual = batch_cur_reward + self.gamma * torch.max(self.model(batch_cur_state),dim=1)[0].detach()
                
                
                #model training
                self.optimizer.zero_grad()
                loss = self.criterion(q_predict, q_actual)
                loss.backward()
                self.optimizer.step()
                
                #if self.iter %1000 == 0:
                #    print "q_predict = {}".format(q_predict)
                #    print "q_actual = {}".format(q_actual)
                
                #model checkpoint
                
                #save data for validation
                
                #After certain iterations (1000?10000?), save model and validation data in files
                # Maybe number of iteration as well (in case of resuming training after close)
            
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
        playery,x,y = env # raw input
        if self.is_baseline:
            return [playery,x[0],y[0]]
        return [playery,x[0],y[0],x[1],y[1]]

