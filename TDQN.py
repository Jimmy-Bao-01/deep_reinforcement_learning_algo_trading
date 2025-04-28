import random

import pandas as pd

from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import dataAugmentation
import tradingEnv


# Default parameters related to the DQN algorithm
gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1

# Default parameters related to the Experience Replay mechanism
capacity = 100000
batchSize = 32
experiencesRequired = 1000

# Default parameters related to the Deep Neural Network
numberOfNeurons = 512
dropout = 0.2

# Default parameters related to the Epsilon-Greedy exploration technique
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000
epsilon = 0.1

# Default parameters regarding the sticky actions RL generalization technique
alpha = 0.1

# Default parameters related to preprocessing
filterOrder = 5

# Default paramters related to the clipping of both the gradient and the RL rewards
gradientClipping = 1
rewardClipping = 1

# Default parameter related to the L2 Regularization 
L2Factor = 0.000001

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0



###############################################################################
############################### Class ReplayMemory ############################
###############################################################################

class ReplayMemory:
    """
    GOAL: Implementing the replay memory required for the Experience Replay
          mechanism of the DQN Reinforcement Learning algorithm.
    
    VARIABLES:  - memory: Data structure storing the experiences.
                                
    METHODS:    - __init__: Initialization of the memory data structure.
                - push: Insert a new experience into the replay memory.
                - sample: Sample a batch of experiences from the replay memory.
                - __len__: Return the length of the replay memory.
                - reset: Reset the replay memory.
    """

    def __init__(self, capacity=capacity):
        """
        GOAL: Initializating the replay memory data structure.
        
        INPUTS: - capacity: Capacity of the data structure, specifying the
                            maximum number of experiences to be stored
                            simultaneously.
        
        OUTPUTS: /
        """

        self.memory = deque(maxlen=capacity)
    

    def push(self, state, action, reward, nextState, done):
        """
        GOAL: Insert a new experience into the replay memory. An experience
              is composed of a state, an action, a reward, a next state and
              a termination signal.
        
        INPUTS: - state: RL state of the experience to be stored.
                - action: RL action of the experience to be stored.
                - reward: RL reward of the experience to be stored.
                - nextState: RL next state of the experience to be stored.
                - done: RL termination signal of the experience to be stored.
        
        OUTPUTS: /
        """

        self.memory.append((state, action, reward, nextState, done))


    def sample(self, batchSize):
        """
        GOAL: Sample a batch of experiences from the replay memory.
        
        INPUTS: - batchSize: Size of the batch to sample.
        
        OUTPUTS: - state: RL states of the experience batch sampled.
                 - action: RL actions of the experience batch sampled.
                 - reward: RL rewards of the experience batch sampled.
                 - nextState: RL next states of the experience batch sampled.
                 - done: RL termination signals of the experience batch sampled.
        """

        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done


    def __len__(self):
        """
        GOAL: Return the capicity of the replay memory, which is the maximum number of
              experiences which can be simultaneously stored in the replay memory.
        
        INPUTS: /
        
        OUTPUTS: - length: Capacity of the replay memory.
        """

        return len(self.memory)


    def reset(self):
        """
        GOAL: Reset (empty) the replay memory.
        
        INPUTS: /
        
        OUTPUTS: /
        """

        self.memory = deque(maxlen=capacity)


def DQN(numberOfInputs, numberOfOutputs, numberOfNeurons=numberOfNeurons, dropout=dropout):
    """
    GOAL: Implement a Deep Q-Network (DQN) using a functional approach.
    
    INPUTS:
    - numberOfInputs: Number of input features.
    - numberOfOutputs: Number of possible actions (Q-values output).
    - numberOfNeurons: Number of neurons per hidden layer (default: 128).
    - dropout: Dropout probability for regularization (default: 0.1).
    
    OUTPUTS:
    - A PyTorch Sequential model representing the Deep Q-Network.
    """
    model = nn.Sequential(
        nn.Linear(numberOfInputs, numberOfNeurons, bias=False),
        nn.BatchNorm1d(numberOfNeurons),
        nn.LeakyReLU(),
        nn.Dropout(dropout),

        nn.Linear(numberOfNeurons, numberOfNeurons, bias=False),
        nn.BatchNorm1d(numberOfNeurons),
        nn.LeakyReLU(),
        nn.Dropout(dropout),

        nn.Linear(numberOfNeurons, numberOfNeurons, bias=False),
        nn.BatchNorm1d(numberOfNeurons),
        nn.LeakyReLU(),
        nn.Dropout(dropout),

        nn.Linear(numberOfNeurons, numberOfNeurons, bias=False),
        nn.BatchNorm1d(numberOfNeurons),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        
        nn.Linear(numberOfNeurons, numberOfOutputs, bias=False)
    )
    
    # Initialize weights using Xavier initialization
    for layer in model:
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
    
    return model



random.seed(0)


device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')

# Default parameters related to the Epsilon-Greedy exploration technique
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000

# Default parameters regarding the sticky actions RL generalization technique
alpha = 0.1

# Variables defining the default observation and state spaces
stateLength = 30
observationSpace = 1 + (stateLength-1)*4
actionSpace = 2

def TDQN(stockName, startingDate, splittingDate):

    dataTraining = pd.read_csv('data/'+stockName+'_'+startingDate+'_'+splittingDate+'.csv').set_index('Date')
    dataTrainingAugmented = dataAugmentation.generate(dataTraining.copy())[0]
    # dataTest = pd.read_csv('data'+stockName+'_'+splittingDate+'_'+endingDate+'.csv').set_index('Date')
    dataTraining = tradingEnv.reset_data(dataTraining)

    # 1. Initialize - Set the two Deep Neural Networks of the DQN algorithm (policy/main and target)
    replayMemory = ReplayMemory(capacity=capacity)  # Experience replay memory
    main_network = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(device)   # Main DQN (θ), with Xavier init
    target_network = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(device) # Target DQN (θ⁻)
    target_network.load_state_dict(main_network.state_dict())  # Initialize target network with main network
    main_network.eval()
    target_network.eval()

    # Set the Deep Learning optimizer
    optimizer = optim.Adam(main_network.parameters(), lr=learningRate, weight_decay=L2Factor)  # Adam optimizer

    # Initialization of the iterations counter
    iterations = 0

    # 2. Training Loop
    print("Training progression (hardware selected => " + str(device) + "):")
    for episode in tqdm(range(50)):
        
        # Set the initial RL variables
        numberOfShares = 0
        transactionCosts = 0.1/100
        previousAction = 0
        done = 0
        t = stateLength
        epsilon = 0.1
        
        coefficients = tradingEnv.getNormalizationCoefficients(dataTrainingAugmented)
        dataTrainingAugmented = tradingEnv.reset_data(dataTrainingAugmented)
        initial_state = tradingEnv.initState(dataTrainingAugmented, stateLength)
        startingPoint = random.randrange(len(dataTrainingAugmented.index))
        initial_state, t, done = tradingEnv.setStartingPoint(startingPoint, dataTrainingAugmented, stateLength)
        state = tradingEnv.processState(initial_state, coefficients)
        
        while done == 0:

            # Epsilon-Greedy Policy
            action, _, _, iterations  = tradingEnv.chooseActionEpsilonGreedy(state, previousAction, main_network, iterations)  # random action
            # Interact with the environment with the chosen action
            nextState, reward, info, numberOfShares, done = tradingEnv.step(action, dataTrainingAugmented, t, numberOfShares, transactionCosts, stateLength, epsilon, done)
            
            # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
            reward = tradingEnv.processReward(reward)
            nextState = tradingEnv.processState(nextState, coefficients)
            replayMemory.push(state, action, reward, nextState, done)

            # Trick for better exploration
            otherAction = int(not bool(action))
            otherReward = tradingEnv.processReward(info['Reward'])
            otherNextState = tradingEnv.processState(info['State'], coefficients)
            otherDone = info['Done']
            replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

            # Execute the DQN learning procedure - Update the policy/main network and the target network
            main_network, target_network, optimizer = tradingEnv.learning(iterations, done, replayMemory, main_network, target_network, optimizer)  # Learning step
            # tradingEnv.learning(iterations, done, replayMemory, main_network, target_network, optimizer)  # Learning step
            
            # Update the RL state
            state = nextState
            previousAction = action
            t += 1
            
    return dataTrainingAugmented, main_network

def testing(dataTraining, dataTesting, main_network):
    """
    GOAL: Test the RL agent trading policy on a new trading environment
            in order to assess the trading strategy performance.
    
    INPUTS: - trainingEnv: Training RL environment (known).
            - testingEnv: Unknown trading RL environment.
            - rendering: Enable the trading environment rendering.
            - showPerformance: Enable the printing of a table summarizing
                                the trading strategy performance.
    
    OUTPUTS: - testingEnv: Trading environment backtested.
    """

    # Initialization of some RL variables
    dataTestingSmoothed = dataAugmentation.lowPassFilter(dataTesting.copy())
    dataTraining = dataAugmentation.lowPassFilter(dataTraining.copy())
    dataTestingSmoothed = tradingEnv.reset_data(dataTestingSmoothed)
    
    coefficients = tradingEnv.getNormalizationCoefficients(dataTraining)
    initial_state = tradingEnv.initState(dataTestingSmoothed, stateLength)
    state = tradingEnv.processState(initial_state, coefficients)
    dataTesting = tradingEnv.reset_data(dataTesting)
    
    QValues0 = []
    QValues1 = []
    done = 0
    numberOfShares = 0
    numberOfSharesTest = 0
    transactionCosts = 0.1/100
    t = stateLength
    
    # Interact with the environment until the episode termination
    while done == 0:

        # Choose an action according to the RL policy and the current RL state
        action, _, QValues = tradingEnv.chooseAction(state, main_network)
            
        # Interact with the environment with the chosen action
        nextState, _, _, numberOfShares, done = tradingEnv.step(action, dataTestingSmoothed, t, numberOfShares, transactionCosts, stateLength, epsilon, done)
        _, _, _, numberOfSharesTest, _ = tradingEnv.step(action, dataTesting, t, numberOfSharesTest, transactionCosts, stateLength, epsilon, done)
        # Update the new state
        state = tradingEnv.processState(nextState, coefficients)

        # Storing of the Q values
        QValues0.append(QValues[0])
        QValues1.append(QValues[1])
        
        t += 1
        
    return dataTesting, QValues0, QValues1
