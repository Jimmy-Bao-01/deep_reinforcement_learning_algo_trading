import random
import math
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None

import torch
import torch.nn.functional as F


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

money = 100000

rewardClipping = 1

gradientClipping = 1
batchSize = 32  # Size of the batch used for the training of the DQN

# Default parameters related to the DQN algorithm
gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1

GPUNumber = 0
device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')

def reset_data(data):
    """
    GOAL: Reset the trading environment to its initial state.
    
    INPUTS: - data: DataFrame containing the trading data.
    
    OUTPUTS: - data: DataFrame with the reset values.
    """

    # Reset the trading environment
    data['Position'] = 0
    data['Action'] = 0
    data['Holdings'] = 0.
    data['Cash'] = float(money)
    data['Money'] = data['Holdings'] + data['Cash']
    data['Returns'] = 0.
    data['NumberOfShares'] = 0

    return data

def initState(data, stateLength):
    """
    GOAL: Setting an arbitrary starting point regarding the trading activity.
            This technique is used for better generalization of the RL agent.
    
    INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
    
    OUTPUTS: /
    """

    # Set the RL variables common to every OpenAI gym environments
    state = [data['Close'][0: stateLength].tolist(),
                    data['Low'][0: stateLength].tolist(),
                    data['High'][0: stateLength].tolist(),
                    data['Volume'][0: stateLength].tolist(),
                    [0]]
    return state

def getNormalizationCoefficients(data):
    """
    GOAL: Retrieve the coefficients required for the normalization
            of input data.
    
    INPUTS: - tradingEnv: RL trading environement to process.
    
    OUTPUTS: - coefficients: Normalization coefficients.
    """

    # Retrieve the available trading data
    # tradingData = tradingEnv
    closePrices = data['Close'].tolist()
    lowPrices = data['Low'].tolist()
    highPrices = data['High'].tolist()
    volumes = data['Volume'].tolist()

    # Retrieve the coefficients required for the normalization
    coefficients = []
    margin = 1
    # 1. Close price => returns (absolute) => maximum value (absolute)
    returns = [abs((closePrices[i]-closePrices[i-1])/closePrices[i-1]) for i in range(1, len(closePrices))]
    coeffs = (0, np.max(returns)*margin)
    coefficients.append(coeffs)
    # 2. Low/High prices => Delta prices => maximum value
    deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(len(lowPrices))]
    coeffs = (0, np.max(deltaPrice)*margin)
    coefficients.append(coeffs)
    # 3. Close/Low/High prices => Close price position => no normalization required
    coeffs = (0, 1)
    coefficients.append(coeffs)
    # 4. Volumes => minimum and maximum values
    coeffs = (np.min(volumes)/margin, np.max(volumes)*margin)
    coefficients.append(coeffs)
    
    return coefficients

def setStartingPoint(startingPoint, data, stateLength):
    """
    GOAL: Setting an arbitrary starting point regarding the trading activity.
            This technique is used for better generalization of the RL agent.
    
    INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
    
    OUTPUTS: /
    """

    # Setting a custom starting point
    t = np.clip(startingPoint, stateLength, len(data.index))

    state = [data['Close'][t - stateLength : t].tolist(),
                data['Low'][t - stateLength : t].tolist(),
                data['High'][t - stateLength : t].tolist(),
                data['Volume'][t - stateLength : t].tolist(),
                [data['Position'][t - 1]]]
    if(t == data.shape[0]):
        done = 1
    else:
        done = 0
    return state, t, done

def processState(state, coefficients):
    """
    GOAL: Process the RL state returned by the environment
            (appropriate format and normalization).
    
    INPUTS: - state: RL state returned by the environment.
    
    OUTPUTS: - state: Processed RL state.
    """

    # Normalization of the RL state
    closePrices = [state[0][i] for i in range(len(state[0]))]
    lowPrices = [state[1][i] for i in range(len(state[1]))]
    highPrices = [state[2][i] for i in range(len(state[2]))]
    volumes = [state[3][i] for i in range(len(state[3]))]

    # 1. Close price => returns => MinMax normalization
    returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1] for i in range(1, len(closePrices))]
    if coefficients[0][0] != coefficients[0][1]:
        state[0] = [((x - coefficients[0][0])/(coefficients[0][1] - coefficients[0][0])) for x in returns]
    else:
        state[0] = [0 for x in returns]
    # 2. Low/High prices => Delta prices => MinMax normalization
    deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(1, len(lowPrices))]
    if coefficients[1][0] != coefficients[1][1]:
        state[1] = [((x - coefficients[1][0])/(coefficients[1][1] - coefficients[1][0])) for x in deltaPrice]
    else:
        state[1] = [0 for x in deltaPrice]
    # 3. Close/Low/High prices => Close price position => No normalization required
    closePricePosition = []
    for i in range(1, len(closePrices)):
        deltaPrice = abs(highPrices[i]-lowPrices[i])
        if deltaPrice != 0:
            item = abs(closePrices[i]-lowPrices[i])/deltaPrice
        else:
            item = 0.5
        closePricePosition.append(item)
    if coefficients[2][0] != coefficients[2][1]:
        state[2] = [((x - coefficients[2][0])/(coefficients[2][1] - coefficients[2][0])) for x in closePricePosition]
    else:
        state[2] = [0.5 for x in closePricePosition]
    # 4. Volumes => MinMax normalization
    volumes = [volumes[i] for i in range(1, len(volumes))]
    if coefficients[3][0] != coefficients[3][1]:
        state[3] = [((x - coefficients[3][0])/(coefficients[3][1] - coefficients[3][0])) for x in volumes]
    else:
        state[3] = [0 for x in volumes]
    
    # Process the state structure to obtain the appropriate format
    state = [item for sublist in state for item in sublist]

    return state

def chooseAction(state, mainNetwork):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.
        
        INPUTS: - state: RL state returned by the environment.
        
        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            tensorState = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            QValues = mainNetwork(tensorState).squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues

def chooseActionEpsilonGreedy(state, previousAction, mainNetwork, iterations):
    """
    GOAL: Choose a valid RL action from the action space according to the
            RL policy as well as the current RL state observed, following the 
            Epsilon Greedy exploration mechanism.

    INPUTS: - state: RL state returned by the environment.
            - previousAction: Previous RL action executed by the agent.

    OUTPUTS: - action: RL action chosen from the action space.
                - Q: State-action value function associated.
                - QValues: Array of all the Qvalues outputted by the
                        Deep Neural Network.
    """
    epsilonValue = lambda iteration: epsilonEnd + (epsilonStart - epsilonEnd) * math.exp(-1 * iteration / epsilonDecay)
    # EXPLOITATION -> RL policy
    if(random.random() > epsilonValue(iterations)):
        # Sticky action (RL generalization mechanism)
        if(random.random() > alpha):
            action, Q, QValues = chooseAction(state, mainNetwork)
        else:
            action = previousAction
            Q = 0
            QValues = [0, 0]

    # EXPLORATION -> Random
    else:
        action = random.randrange(actionSpace)
        Q = 0
        QValues = [0, 0]

    # Increment the iterations counter (for Epsilon Greedy)
    iterations += 1

    return action, Q, QValues, iterations

def computeLowerBound(cash, numberOfShares, price, transactionCosts, epsilon):
    """
    GOAL: Compute the lower bound of the complete RL action space, 
            i.e. the minimum number of share to trade.
    
    INPUTS: - cash: Value of the cash owned by the agent.
            - numberOfShares: Number of shares owned by the agent.
            - price: Last price observed.
    
    OUTPUTS: - lowerBound: Lower bound of the RL action space.
    """

    # Computation of the RL action lower bound
    deltaValues = - cash - numberOfShares * price * (1 + epsilon) * (1 + transactionCosts)
    if deltaValues < 0:
        lowerBound = deltaValues / (price * (2 * transactionCosts + (epsilon * (1 + transactionCosts))))
    else:
        lowerBound = deltaValues / (price * epsilon * (1 + transactionCosts))
    return lowerBound

def step(action, data, t, numberOfShares, transactionCosts, stateLength, epsilon, done):
    """
    GOAL: Transition to the next trading time step based on the
            trading position decision made (either long or short).
    
    INPUTS: - action: Trading decision (1 = long, 0 = short).    
    
    OUTPUTS: - state: RL state to be returned to the RL agent.
                - reward: RL reward to be returned to the RL agent.
                - done: RL episode termination signal (boolean).
                - info: Additional information returned to the RL agent.
    """

    # Stting of some local variables
    # t = t
    numberOfSharesMain = numberOfShares
    customReward = False

    # CASE 1: LONG POSITION
    if(action == 1):
        data['Position'][t] = 1
        # Case a: Long -> Long
        if(data['Position'][t - 1] == 1):
            data['Cash'][t] = data['Cash'][t - 1]
            data['Holdings'][t] = numberOfSharesMain * data['Close'][t]
        # Case b: No position -> Long
        elif(data['Position'][t - 1] == 0):
            numberOfSharesMain = math.floor(data['Cash'][t - 1]/(data['Close'][t] * (1 + transactionCosts)))
            data['Cash'][t] = data['Cash'][t - 1] - numberOfSharesMain * data['Close'][t] * (1 + transactionCosts)
            data['Holdings'][t] = numberOfSharesMain * data['Close'][t]
            data['Action'][t] = 1
        # Case c: Short -> Long
        else:
            data['Cash'][t] = data['Cash'][t - 1] - numberOfSharesMain * data['Close'][t] * (1 + transactionCosts)
            numberOfSharesMain = math.floor(data['Cash'][t]/(data['Close'][t] * (1 + transactionCosts)))
            data['Cash'][t] = data['Cash'][t] - numberOfSharesMain * data['Close'][t] * (1 + transactionCosts)
            data['Holdings'][t] = numberOfSharesMain * data['Close'][t]
            data['Action'][t] = 1

    # CASE 2: SHORT POSITION
    elif(action == 0):
        data['Position'][t] = -1
        # Case a: Short -> Short
        if(data['Position'][t - 1] == -1):
            lowerBound = computeLowerBound(data['Cash'][t - 1], -numberOfSharesMain, data['Close'][t-1], transactionCosts, epsilon)
            if lowerBound <= 0:
                data['Cash'][t] = data['Cash'][t - 1]
                data['Holdings'][t] =  - numberOfSharesMain * data['Close'][t]
            else:
                numberOfSharesToBuy = min(math.floor(lowerBound), numberOfSharesMain)
                numberOfSharesMain -= numberOfSharesToBuy
                data['Cash'][t] = data['Cash'][t - 1] - numberOfSharesToBuy * data['Close'][t] * (1 + transactionCosts)
                data['Holdings'][t] =  - numberOfSharesMain * data['Close'][t]
                customReward = True
        # Case b: No position -> Short
        elif(data['Position'][t - 1] == 0):
            numberOfSharesMain = math.floor(data['Cash'][t - 1]/(data['Close'][t] * (1 + transactionCosts)))
            data['Cash'][t] = data['Cash'][t - 1] + numberOfSharesMain * data['Close'][t] * (1 - transactionCosts)
            data['Holdings'][t] = - numberOfSharesMain * data['Close'][t]
            data['Action'][t] = -1
        # Case c: Long -> Short
        else:
            data['Cash'][t] = data['Cash'][t - 1] + numberOfSharesMain * data['Close'][t] * (1 - transactionCosts)
            numberOfSharesMain = math.floor(data['Cash'][t]/(data['Close'][t] * (1 + transactionCosts)))
            data['Cash'][t] = data['Cash'][t] + numberOfSharesMain * data['Close'][t] * (1 - transactionCosts)
            data['Holdings'][t] = - numberOfSharesMain * data['Close'][t]
            data['Action'][t] = -1

    # CASE 3: PROHIBITED ACTION
    else:
        raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

    # Update the total amount of money owned by the agent, as well as the return generated
    data['Money'][t] = data['Holdings'][t] + data['Cash'][t]
    data['Returns'][t] = (data['Money'][t] - data['Money'][t-1])/data['Money'][t-1]
    data['NumberOfShares'][t] = numberOfSharesMain
    
    # Set the RL reward returned to the trading agent
    if not customReward:
        reward = data['Returns'][t]
    else:
        reward = (data['Close'][t-1] - data['Close'][t])/data['Close'][t-1]

    # Transition to the next trading time step
    t_state = t + 1
    state = [data['Close'][t_state - stateLength : t_state].tolist(),
                    data['Low'][t_state - stateLength : t_state].tolist(),
                    data['High'][t_state - stateLength : t_state].tolist(),
                    data['Volume'][t_state - stateLength : t_state].tolist(),
                    [data['Position'][t_state - 1]]]
    
    if(t_state == data.shape[0]):
        done = 1  

    # Same reasoning with the other action (exploration trick)
    otherAction = int(not bool(action))
    customReward = False
    if(otherAction == 1):
        otherPosition = 1
        if(data['Position'][t - 1] == 1):
            otherCash = data['Cash'][t - 1]
            otherHoldings = numberOfShares * data['Close'][t]
        elif(data['Position'][t - 1] == 0):
            numberOfShares = math.floor(data['Cash'][t - 1]/(data['Close'][t] * (1 + transactionCosts)))
            otherCash = data['Cash'][t - 1] - numberOfShares * data['Close'][t] * (1 + transactionCosts)
            otherHoldings = numberOfShares * data['Close'][t]
        else:
            otherCash = data['Cash'][t - 1] - numberOfShares * data['Close'][t] * (1 + transactionCosts)
            numberOfShares = math.floor(otherCash/(data['Close'][t] * (1 + transactionCosts)))
            otherCash = otherCash - numberOfShares * data['Close'][t] * (1 + transactionCosts)
            otherHoldings = numberOfShares * data['Close'][t]
    else:
        otherPosition = -1
        if(data['Position'][t - 1] == -1):
            lowerBound = computeLowerBound(data['Cash'][t - 1], -numberOfShares, data['Close'][t-1], transactionCosts, epsilon)
            if lowerBound <= 0:
                otherCash = data['Cash'][t - 1]
                otherHoldings =  - numberOfShares * data['Close'][t]
            else:
                numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
                numberOfShares -= numberOfSharesToBuy
                otherCash = data['Cash'][t - 1] - numberOfSharesToBuy * data['Close'][t] * (1 + transactionCosts)
                otherHoldings =  - numberOfShares * data['Close'][t]
                customReward = True
        elif(data['Position'][t - 1] == 0):
            numberOfShares = math.floor(data['Cash'][t - 1]/(data['Close'][t] * (1 + transactionCosts)))
            otherCash = data['Cash'][t - 1] + numberOfShares * data['Close'][t] * (1 - transactionCosts)
            otherHoldings = - numberOfShares * data['Close'][t]
        else:
            otherCash = data['Cash'][t - 1] + numberOfShares * data['Close'][t] * (1 - transactionCosts)
            numberOfShares = math.floor(otherCash/(data['Close'][t] * (1 + transactionCosts)))
            otherCash = otherCash + numberOfShares * data['Close'][t] * (1 - transactionCosts)
            otherHoldings = - numberOfSharesMain * data['Close'][t]
    otherMoney = otherHoldings + otherCash
    
    if not customReward:
        otherReward = (otherMoney - data['Money'][t-1])/data['Money'][t-1]
    else:
        otherReward = (data['Close'][t-1] - data['Close'][t])/data['Close'][t-1]
    otherState = [data['Close'][t_state - stateLength : t_state].tolist(),
                    data['Low'][t_state - stateLength : t_state].tolist(),
                    data['High'][t_state - stateLength : t_state].tolist(),
                    data['Volume'][t_state - stateLength : t_state].tolist(),
                    [otherPosition]]
    info = {'State' : otherState, 'Reward' : otherReward, 'Done' : done}

    # Return the trading environment feedback to the RL trading agent
    return state, reward, info, numberOfSharesMain, done

def processReward(reward):
    """
    GOAL: Process the RL reward returned by the environment by clipping
            its value. Such technique has been shown to improve the stability
            the DQN algorithm.
    
    INPUTS: - reward: RL reward returned by the environment.
    
    OUTPUTS: - reward: Process RL reward.
    """

    return np.clip(reward, -rewardClipping, rewardClipping)

targetNetworkUpdate = 1000

def updateTargetNetwork(iterations, mainNetwork, targetNetwork):
    """
    GOAL: Update the target network weights with the main network weights
            every targetNetworkUpdate iterations.
    
    INPUTS: - iterations: Number of iterations executed.
            - mainNetwork: Main DQN network.
            - targetNetwork: Target DQN network.
    
    OUTPUTS: /
    """

    if(iterations % targetNetworkUpdate == 0):
        targetNetwork.load_state_dict(mainNetwork.state_dict())
    return targetNetwork
        
def learning(iterations, done, replayMemory, mainNetwork, targetNetwork, optimizer, batchSize=batchSize):
    """
    GOAL: Sample a batch of past experiences and learn from it
            by updating the Reinforcement Learning policy.
    
    INPUTS: batchSize: Size of the batch to sample from the replay memory.
    
    OUTPUTS: /
    """
    
    # Check that the replay memory is filled enough
    if (len(replayMemory) >= batchSize):
        # Set the Deep Neural Network in training mode
        mainNetwork.train()

        # Sample a batch of experiences from the replay memory
        state, action, reward, nextState, done = replayMemory.sample(batchSize)
        # print(done)

        # Initialization of Pytorch tensors for the RL experience elements
        state = torch.tensor(np.array(state), dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        nextState = torch.tensor(nextState, dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.float, device=device)

        # Compute the current Q values returned by the policy network
        currentQValues = mainNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute the next Q values returned by the target network
        with torch.no_grad():
            nextActions = torch.max(mainNetwork(nextState), 1)[1]
            nextQValues = targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
            expectedQValues = reward + gamma * nextQValues * (1 - done)

        # Compute the Huber loss
        loss = F.smooth_l1_loss(currentQValues, expectedQValues)

        # Computation of the gradients
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(mainNetwork.parameters(), gradientClipping)

        # Perform the Deep Neural Network optimization
        optimizer.step()

        # If required, update the target deep neural network (update frequency)
        targetNetwork = updateTargetNetwork(iterations, mainNetwork, targetNetwork)

        # Set back the Deep Neural Network in evaluation mode
        mainNetwork.eval()
        
    return mainNetwork, targetNetwork, optimizer
