import math, pygame, sys, os, copy, time, random
import numpy as np
import pygame.gfxdraw
from pygame.locals import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import puzzlebobble
from puzzlebobble import *
import torch.optim as optim


# Settings of the game
FPS          = 120
WINDOWWIDTH  = 640
WINDOWHEIGHT = 480
TEXTHEIGHT   = 20
BUBBLERADIUS = 20
BUBBLEWIDTH  = BUBBLERADIUS * 2
BUBBLELAYERS = 6
BUBBLEYADJUST = 5
STARTX = WINDOWWIDTH / 2
STARTY = WINDOWHEIGHT - 27
ARRAYWIDTH = 16
ARRAYHEIGHT = 14
DIE = 9
ARRAYWIDTH = 16


# Colours
#            R    G    B
GRAY     = (100, 100, 100)
NAVYBLUE = ( 60,  60, 100)
WHITE    = (255, 255, 255)
RED      = (255,   0,   0)
GREEN    = (  0, 255,   0)
BLUE     = (  0,   0, 255)
YELLOW   = (255, 255,   0)
ORANGE   = (255, 128,   0)
PURPLE   = (255,   0, 255)
CYAN     = (  0, 255, 255)
BLACK    = (  0,   0,   0)
COMBLUE  = (233, 232, 255)

RIGHT = 'right'
LEFT  = 'left'
BLANK = '.'

# background colour
BGCOLOR    = WHITE

# Colours in the game
COLORLIST = [RED, GREEN, YELLOW, ORANGE, CYAN]

# Put display on true for visualisation
DISPLAY = False

# Replay Memory
REPLAYMEMORY = []

# Hyper parameters
BATCHSIZE = 128
AMOUNTOFSHOTS = 500
NUMBEROFACTIONS = 20
GRIDSIZE = 11
discount = 0.99
learning_rate = 0.00008
size_RM = 7500
ITERATIONS = 400000

def main():
    # Game initalization
    global FPSCLOCK, DISPLAYSURF, DISPLAYRECT, MAINFONT
    pygame.init()
    if DISPLAY == True:
        pygame.display.set_caption('Puzzle Bobble')
        DISPLAYSURF, DISPLAYRECT = makeDisplay()
    createMemory()
    network = train()
    test(network)


# Training of the self-learning agent
def train():

    # initialize game
    direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()

    # hyperparameters
    epsilon = 0.9

    # counters
    moves = 0
    wins = 0
    gameover = 0
    games = 0
    average_loss = 0
    average_reward = 0

    # with or without display
    display = False
    delay = 0

    # Tensor types
    STATE = T.tensor4()
    NEWSTATE = T.tensor4()
    REWARD = T.icol()
    DISCOUNT = T.col()
    ACTION = T.icol()

    # building network
    network = CNN()
    target_network = CNN()

    # get parameters from trained network
    """
    with np.load('5_colours_20shots.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)"""

    params = lasagne.layers.get_all_params(network)

    all_params = lasagne.layers.helper.get_all_param_values(network)
    lasagne.layers.helper.set_all_param_values(target_network, all_params)

    # get maximum q_value and particular action
    qvals = lasagne.layers.get_output(network, STATE)
    bestAction = qvals.argmax(-1)
    qval = qvals[0][ACTION]

    # get max Q_value of next state
    next_q_vals = lasagne.layers.get_output(target_network, NEWSTATE)
    maxNextValue = next_q_vals.max()

    # loss function with Stochastic Gradient Descent
    target = (REWARD + DISCOUNT * T.max(next_q_vals, axis=1, keepdims=True))
    diff = target - qvals[T.arange(BATCHSIZE), ACTION.reshape((-1,))].reshape((-1, 1))
    loss = 0.5 * diff ** 2
    loss = T.mean(loss)
    grad = T.grad(loss, params)
    updates = lasagne.updates.rmsprop(grad, params, learning_rate)
    updates = lasagne.updates.apply_momentum(updates, params, 0.9)

    # theano function for training and predicting q_values
    f_train = theano.function([STATE, ACTION, REWARD, NEWSTATE, DISCOUNT], loss, updates=updates, allow_input_downcast=True)
    f_predict = theano.function([STATE], bestAction, allow_input_downcast=True)
    f_qvals = theano.function([STATE], qvals, allow_input_downcast=True)
    f_max = theano.function([NEWSTATE], maxNextValue, allow_input_downcast=True)

    # get state
    state = gameState(bubbleArray, newBubble.color)
    while moves < ITERATIONS:

        if display == True:
            DISPLAYSURF.fill(BGCOLOR)
        # act random or greedy
        chance = random.uniform(0, 1)
        launchBubble = True
        if chance < epsilon:
            action = random.randint(0, NUMBEROFACTIONS - 1)
        else:
            predict_state = np.reshape(state, (1, 8, GRIDSIZE * 2, ARRAYWIDTH * 2))
            action = int(f_predict(predict_state))
        direction = (action * 8) + 10
        newBubble.angle = direction

        # process game
        bubbleArray, alive, deleteList, nextBubble = processGame(launchBubble, newBubble, bubbleArray, score, arrow, direction, alive, display, delay)

        # get reward for the action
        getout, wins, reward, gameover = getReward(alive, getout, wins, deleteList, gameover)

        # getting new bubble for shooting
        newBubble = Bubble(nextBubble.color)
        newBubble.angle = arrow.angle

        # get the newstate
        newState = gameState(bubbleArray, newBubble.color)

        # storage of replay memory
        if getout == True:
            REPLAYMEMORY.append((state, action, reward, newState, 0))
        else:
            REPLAYMEMORY.append((state, action, reward, newState, discount))

        # delete one tuple is replay memory becomes too big
        if len(REPLAYMEMORY) > size_RM:
            REPLAYMEMORY.pop(0)

        # training the network
        states, actions, rewards, newstates, discounts = get_batch()
        loss = f_train(states, actions, rewards, newstates, discounts)

        average_loss = average_loss + loss
        average_reward = average_reward + reward

        if moves % 1000 == 0 and moves > 0:
            print("Amount of actions taken: ", moves)
            print("Average loss: ", average_loss/1000.0)
            print("Average Reward: ", average_reward/1000.0)
            print("Amount of wins: ", wins)
            average_reward = 0 
            average_loss = 0
            if epsilon > 0.1:
                epsilon = epsilon - 0.01

        # updating the target network
        if moves % 2500 == 0:
            target_network = build_network()
            all_param_values = lasagne.layers.get_all_param_values(network)
            lasagne.layers.set_all_param_values(target_network, all_param_values)

        # change the state to newState
        state = newState

        moves = moves + 1
        shots = shots + 1

        if getout == True or shots == AMOUNTOFSHOTS:
            games = games + 1
            direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()
        state = gameState(bubbleArray, newBubble.color)

    # saving parameters of the network
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    return network


# test the performance of the algorithm
def test(network):

    # initalize game
    direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()

    # Display of the game 
    display = False
    delay = 0


    # counters for evaluation
    moves = 0
    wins = 0
    gameover = 0
    games = 0
    average_reward = 0
    gameover = 0
    average_score = 0
    average_shot = 0
    average_shot_win = 0

    # Tensor type
    STATE = T.tensor4()

    # loading parameters of a trained model
    """
    with np.load('5_colours_20shots_improve.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)"""

    # function for getting the best action
    qvals = lasagne.layers.get_output(network, STATE,  deterministic=True)
    bestAction = qvals.argmax(-1)


    f_predict = theano.function([STATE], bestAction, allow_input_downcast=True)



    # get current state
    state = gameState(bubbleArray, newBubble.color)

    while games < 1000:
        if display == True:
            DISPLAYSURF.fill(BGCOLOR)

        # performing the best action
        state = np.reshape(state, (1, 8, GRIDSIZE * 2, ARRAYWIDTH * 2))
        action = int(f_predict(state))
        direction = (action * 8) + 10
        launchBubble = True
        newBubble.angle = direction

        # process game
        bubbleArray, alive, deleteList, nextBubble = processGame(launchBubble, newBubble, bubbleArray, score, arrow, direction, alive, display, delay)
        shots = shots + 1

        # adding extra ball layers
        """
        if shots % 32 == 0 and shots > 0:
            bubbleArray = addLayer(bubbleArray)"""

        # average amount of balls popped per shot
        if len(deleteList) > 2:
            average_reward = len(deleteList) + average_reward
        getout, wins, reward, gameover = getReward(alive, getout, wins, deleteList, gameover)

        # new bubble for shooting
        newBubble = Bubble(nextBubble.color)
        newBubble.angle = arrow.angle

        moves = moves + 1

        if getout == True or shots == AMOUNTOFSHOTS:
            if alive == "win":
                average_shot_win = average_shot_win + shots
            average_score = average_score + score.total
            games = games + 1
            if games % 50 == 0:
                print (games, "/1000")
            average_shot = average_shot + shots
            direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()


        # update state
        state = gameState(bubbleArray, newBubble.color)
    print("Amount of games played: ", games)
    print("Amount of wins: ", wins)
    print("Amount of loses: ", gameover)
    print("Average reward: ", average_reward/float(moves))
    print("Average score: ", average_score/float(games))
    print("Average amount of shots per game: ", average_shot/float(games))
    if wins > 0:
        print("Average amount of shots per winning game: ", average_shot_win/float(wins))

# creating Replay memory
def createMemory():

    # start game
    direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()

    # Display of the game
    display = False
    delay = 0

    # counters
    gameover = 0
    games = 0
    wins = 0
    moves = 0
    terminal = 0
    layer = 0

    # get the current game state
    state = gameState(bubbleArray, newBubble.color)

    while moves < 5000:
        if display == True:
            DISPLAYSURF.fill(BGCOLOR)

        # performing random action
        action = random.randint(0, NUMBEROFACTIONS - 1)
        direction = (action * 8) + 10
        launchBubble = True
        newBubble.angle = direction

        # process game
        bubbleArray, alive, deleteList, nextBubble = processGame(launchBubble, newBubble, bubbleArray, score, arrow, direction, alive, display, delay)
        shots = shots + 1

        # getting reward for action
        getout, wins, reward, gameover = getReward(alive, getout, wins, deleteList, gameover)

        # new bubble for shooting
        newBubble = Bubble(nextBubble.color)
        newBubble.angle = arrow.angle

        # getting new state
        newState = gameState(bubbleArray, newBubble.color)

        moves = moves + 1

        # storage of the replay memory
        if getout == True:
            REPLAYMEMORY.append((state, action, reward, newState, 0))
        else:
            REPLAYMEMORY.append((state, action, reward, newState, discount))

        # restart game when game is won or lost
        if getout == True or shots == AMOUNTOFSHOTS:
            games = games + 1
            direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()
        state = gameState(bubbleArray, newBubble.color)
    print("Size of the Replay Memory: ", len(REPLAYMEMORY))


# for getting a batch for training
def get_batch():
    counter = 0
    actions  = np.zeros((BATCHSIZE, 1))
    discounts = np.zeros((BATCHSIZE, 1))
    rewards = np.zeros((BATCHSIZE, 1))
    states = np.zeros((BATCHSIZE, 8, GRIDSIZE * 2, ARRAYWIDTH *2))
    newstates = np.zeros((BATCHSIZE, 8, GRIDSIZE * 2, ARRAYWIDTH *2))
    while counter < BATCHSIZE:
        random_action = random.randint(0, len(REPLAYMEMORY) - 1)
        (state, action, reward, newstate, discount) = REPLAYMEMORY[random_action]
        actions[counter][0] = action
        rewards[counter][0] = reward
        discounts[counter][0] = discount
        states[counter] = state
        newstates[counter] = newstate
        counter = counter + 1
    return states, actions, rewards, newstates, discounts 


# adding extra lines of balls to the game
def addLayer(bubbleArray):
    gameColorList = updateColorList(bubbleArray)
    if gameColorList[0] == WHITE:
        return bubbleArray
    bubbleArray = np.delete(bubbleArray, -1, 0)
    bubbleArray = np.delete(bubbleArray, -1, 0)
    newRow = []
    for index in range(len(bubbleArray[0])):
        newRow.append(BLANK)
    newRow = np.asarray(newRow)
    newRow = np.reshape(newRow, (1,16))
    bubbleArray = np.concatenate((newRow, bubbleArray))
    bubbleArray = np.concatenate((newRow, bubbleArray))
    for n in range(2):
        for index in range(len(bubbleArray[0])):
            random.shuffle(gameColorList)
            newBubble = Bubble(gameColorList[0], n, index)
            bubbleArray[n][index] = newBubble
    bubbleArray[1][15] = BLANK
    return bubbleArray


# reward function
def getReward(alive, getout, wins, deleteList, gameover):
    if alive == "win":
        wins = wins + 1
        reward = 20.
        getout = True
    elif alive == "lose":
        reward = -15.
        getout = True
        gameover = gameover + 1
    elif len(deleteList) > 2:
        getout = False
        reward = 1. * len(deleteList)
    elif len(deleteList) == 2:
        getout = False
        reward = 1.
    else: 
        reward = -1.
        getout = False
    return getout, wins, reward, gameover


# to restart the game
def restartGame():
    direction = None
    launchBubble = False
    gameColorList = copy.deepcopy(COLORLIST)
    arrow = Arrow()
    bubbleArray = makeBlankBoard()
    setBubbles(bubbleArray, gameColorList)
    nextBubble = Bubble(gameColorList[0])
    nextBubble.rect.right = WINDOWWIDTH - 5
    nextBubble.rect.bottom = WINDOWHEIGHT - 5
    score = Score()
    alive = "alive"
    newBubble = Bubble(nextBubble.color)
    newBubble.angle = arrow.angle
    shots = 0
    getout = False
    loss_game = 0
    return direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game


# building the network
class baselineANN(nn.Module):
    def __init__(self):
        super(baselineANN, self).__init__()
        self.name = "baseline"
        self.layer = nn.Linear(GRIDSIZE * 2 * ARRAYWIDTH * 2, 3)
    def forward(self, arr):
        activation = self.layer(self)
        output = F.relu(activation)
        return output

class CNN(nn.Module):
    #TODO: add configurability e.g. num_layers, num_neurons in each layer
    def __init__(self):
        super(CNN, self).__init__()
        self.name = "BubbleCNN"
        
        # Input has 4 neurons:
        #   - Horizontal distance to 1st and 2nd pipes
        #   - Vertical distance to 1st and 2nd lower pipes
        # Output has 2 neurons (each represents the Q-value of 1 action)
        self.conv1 = nn.Conv2d(8, 8, 5) #the output should be 18*28
        self.pool = nn.MaxPool2d(2,2)#should be 9*14
        
        self.fc1 = nn.Linear(8*9*14, 200)
        self.fc2 = nn.Linear(200, 20)
        
    def forward(self, network_inp):
        x = F.relu(self.conv1(network_inp))
        x = self.pool(x)
        x = x.view(-1, 8*9*14)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CNNAgent():
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
        self.model = baselineANN() if is_baseline else CNN()
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
    
    def Action(self,env,score,not_lose,is_train=False):
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
            if not_lose:
                if score.total > self.score.total:
                    cur_reward = score.total - self.score.total
                else:
                    cur_reward = -1
            else:
                cur_reward = -50
            
            prev_action = self.action
            
            self.replay_mem.append((prev_state,prev_action,cur_state,cur_reward))
            if len(self.replay_mem) >= self.replay_mem_size:
                self.replay_mem.pop(0)
            
            # Decide on next action
            if random.random() <= self.epsilon:
                #Exploration: With probability epsilon, choose random action
                self.action = random.randint(0,19)
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
        return env
    
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

# getting the current game state
def gameState(bubbleArray, ballcolor):
    dimension = 0
    state = np.ones((8, GRIDSIZE * 2, ARRAYWIDTH * 2)) * -1
    for colour in COLORLIST:
        counter = 0
        balls = 0
        if ballcolor == colour:
            state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH] = 1
            state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH + 1] = 1
            state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH] = 1
            state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH + 1] = 1
            balls = balls + 1
        for row in range(GRIDSIZE):
            for column in range(len(bubbleArray[0])):
                if bubbleArray[row][column] != BLANK and bubbleArray[row][column].color == colour:
                    if counter % 2 == 0:
                        state[dimension][row * 2][(2 * column)] = 1
                        state[dimension][row * 2][(2 * column) + 1] = 1
                        state[dimension][row * 2 + 1][(2 * column)] = 1
                        state[dimension][row * 2 + 1][(2 * column) + 1] = 1
                    elif counter % 2 != 0:
                        state[dimension][row * 2][2 * column + 1] = 1
                        state[dimension][row * 2][2 * column + 2] = 1
                        state[dimension][row * 2 + 1][2 * column + 1] = 1
                        state[dimension][row * 2 + 1][2 * column + 2] = 1
                    balls = balls + 1
            counter = counter + 1
        for row in range(GRIDSIZE * 2):
            for column in range(len(bubbleArray[0]) * 2):
                if state[dimension][row][column] > 0:
                    state[dimension][row][column] = 1/(float(balls) * 4.)
                if state[dimension][row][column] < 0:
                    state[dimension][row][column] = -1 * 1/float((GRIDSIZE * 2 * ARRAYWIDTH * 2) - 4. * balls)
        dimension = dimension + 1

    balls = 1 # shooting ball
    counter = 0
    state[dimension] = np.ones((GRIDSIZE * 2, ARRAYWIDTH * 2))
    state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH] = -1
    state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH + 1] = -1
    state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH] = -1
    state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH + 1] = -1
    for row in range(GRIDSIZE):
        for column in range(len(bubbleArray[0])):
            if bubbleArray[row][column] != BLANK:
                balls = balls + 1
                if counter % 2 == 0:
                    state[dimension][row * 2][(2 * column)] = -1
                    state[dimension][row * 2][(2 * column) + 1] = -1
                    state[dimension][row * 2 + 1][(2 * column)] = -1
                    state[dimension][row * 2 + 1][(2 * column) + 1] = -1
                elif counter % 2 != 0:
                    state[dimension][row * 2][2 * column + 1] = -1
                    state[dimension][row * 2][2 * column + 2] = -1
                    state[dimension][row * 2 + 1][2 * column + 1] = -1
                    state[dimension][row * 2 + 1][2 * column + 2] = -1
        counter = counter + 1
    for row in range(GRIDSIZE * 2):
        for column in range(len(bubbleArray[0]) * 2):
            if state[dimension][row][column] > 0:
                state[dimension][row][column] = 1/(float(balls) * 4.)
            if state[dimension][row][column] < 0:
                state[dimension][row][column] = -1 * 1/(float((GRIDSIZE * 2 * ARRAYWIDTH * 2) - 4. * balls))
    dimension = dimension + 1
    for n in range(8 - len(COLORLIST) - 1):
        state[dimension] = np.zeros((GRIDSIZE * 2, ARRAYWIDTH * 2))
        dimension = dimension + 1
    return state


if __name__ == '__main__':
    main()