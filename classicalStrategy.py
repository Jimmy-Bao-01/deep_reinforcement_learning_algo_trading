import math
import numpy as np
from tqdm import tqdm

import tradingEnv
import tradingPerformance

stateLength = 30
bounds = [1, 30]
step = 2
epsilon = 0.1
transactionCost = 0.1/100

def buy_and_hold(data):
    """
    GOAL: Buy and hold strategy for a given trading environment.
    
    INPUTS: - tradingEnv: Trading environment to evaluate.
            - initialInvestment: Initial investment amount.
    
    OUTPUTS: - finalValue: Final value of the investment.
    """
    
    data = tradingEnv.reset_data(data)
    done = 0
    action = 1
    t = stateLength
    numberOfShares = 0

    while done == 0:
        _, _, _, numberOfShares, done = tradingEnv.step(action, data, t, numberOfShares, transactionCost, stateLength, epsilon, done)
        t += 1
    
    return data

def sell_and_hold(data):
    """
    GOAL: Buy and hold strategy for a given trading environment.
    
    INPUTS: - tradingEnv: Trading environment to evaluate.
            - initialInvestment: Initial investment amount.
    
    OUTPUTS: - finalValue: Final value of the investment.
    """
    
    data = tradingEnv.reset_data(data)
    done = 0
    action = 0
    t = stateLength
    numberOfShares = 0

    while done == 0:
        _, _, _, numberOfShares, done = tradingEnv.step(action, data, t, numberOfShares, transactionCost, stateLength, epsilon, done)
        t += 1
    
    return data

def MA_TF(dataTraining, dataTesting, parameters=[5, 10]):
    """
    GOAL: Buy and hold strategy for a given trading environment.
    
    INPUTS: - tradingEnv: Trading environment to evaluate.
            - initialInvestment: Initial investment amount.
    
    OUTPUTS: - finalValue: Final value of the investment.
    """
    
    def chooseAction(state, parameters):
        """
        GOAL: Make a decision regarding the next trading position
              (long=1 and short=0) based on the moving averages.
        
        INPUTS: - state: State of the trading environment.      
        
        OUTPUTS: - action: Trading position decision (long=1 and short=0).
        """

        # Processing of the trading environment state
        state_TF = state[0]

        # Computation of the two moving averages
        shortAverage = np.mean(state_TF[-parameters[0]:])
        longAverage = np.mean(state_TF[-parameters[1]:])

        # Comparison of the two moving averages
        if(shortAverage >= longAverage):
            # Long position
            return 1
        else:
            # Short position
            return 0
    
    def training(data, trainingParameters=[bounds, step]):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters, by simulating many combinations of parameters.
        
        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase simulations.   
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Trading environment associated with the best
                                trading strategy parameters backtested.
        """

        # Compute the dimension of the parameter search space
        bounds = trainingParameters[0]
        step = trainingParameters[1]
        dimension = math.ceil((bounds[1] - bounds[0])/step)

        # Initialize some variables required for the simulations
        data = tradingEnv.reset_data(data)
        results = np.zeros((dimension, dimension))
        bestShort = 0
        bestLong = 0
        bestPerformance = -100
        i = 0
        j = 0
        count = 1

        # Loop through all the parameters combinations included in the parameter search space
        print("Training in progress (MA_TF)...")
        for shorter in tqdm(range(bounds[0], bounds[1], step)):
            for longer in range(bounds[0], bounds[1], step):

                # Obvious restriction on the parameters
                if(shorter < longer):

                    # Apply the trading strategy with the current combination of parameters
                    parameters = [shorter, longer]
                    done = 0
                    t = stateLength
                    numberOfShares = 0
                    state = tradingEnv.initState(data, stateLength)
                    while done == 0:
                        nextState, _, _, numberOfShares, done = tradingEnv.step(chooseAction(state, parameters), data, t, numberOfShares, transactionCost, \
                                stateLength, epsilon, done)
                        state = nextState
                        t += 1

                    # Retrieve the performance associated with this simulation (Sharpe Ratio)
                    performance = tradingPerformance.computeSharpeRatio(data)
                    results[i][j] = performance

                    # Track the best performance and parameters
                    if(performance > bestPerformance):
                        bestShort = shorter
                        bestLong = longer
                        bestPerformance = performance
                    
                    # Reset of the trading environment
                    data = tradingEnv.reset_data(data)
                    count += 1

                j += 1
            i += 1
            j = 0

        # Execute once again the strategy associated with the best parameters simulated
        parameters = [bestShort, bestLong]
        return parameters

    parameters = training(dataTraining)
    dataTesting = tradingEnv.reset_data(dataTesting)
    state = tradingEnv.initState(dataTesting, stateLength)
    done = 0
    t = stateLength
    numberOfShares = 0

    while done == 0:
            
        nextState, _, _, numberOfShares, done = tradingEnv.step(chooseAction(state, parameters), dataTesting, t, numberOfShares, transactionCost,\
                stateLength, epsilon, done)
        state = nextState
        t += 1
    
    return dataTesting

def MA_MR(dataTraining, dataTesting, parameters=[5, 10]):
    """
    GOAL: Buy and hold strategy for a given trading environment.
    
    INPUTS: - tradingEnv: Trading environment to evaluate.
            - initialInvestment: Initial investment amount.
    
    OUTPUTS: - finalValue: Final value of the investment.
    """
    
    def chooseAction(state, parameters):
        """
        GOAL: Make a decision regarding the next trading position
              (long=1 and short=0) based on the moving averages.
        
        INPUTS: - state: State of the trading environment.      
        
        OUTPUTS: - action: Trading position decision (long=1 and short=0).
        """

        # Processing of the trading environment state
        state_TF = state[0]

        # Computation of the two moving averages
        shortAverage = np.mean(state_TF[-parameters[0]:])
        longAverage = np.mean(state_TF[-parameters[1]:])

        # Comparison of the two moving averages
        if(shortAverage <= longAverage):
            # Long position
            return 1
        else:
            # Short position
            return 0
    
    def training(data, trainingParameters=[bounds, step]):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters, by simulating many combinations of parameters.
        
        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase simulations.   
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Trading environment associated with the best
                                trading strategy parameters backtested.
        """

        # Compute the dimension of the parameter search space
        bounds = trainingParameters[0]
        step = trainingParameters[1]
        dimension = math.ceil((bounds[1] - bounds[0])/step)

        # Initialize some variables required for the simulations
        data = tradingEnv.reset_data(data)
        results = np.zeros((dimension, dimension))
        bestShort = 0
        bestLong = 0
        bestPerformance = -100
        i = 0
        j = 0
        count = 1

        # Loop through all the parameters combinations included in the parameter search space
        print("Training in progress (MA_MR)...")
        for shorter in tqdm(range(bounds[0], bounds[1], step)):
            for longer in range(bounds[0], bounds[1], step):

                # Obvious restriction on the parameters
                if(shorter < longer):

                    # Apply the trading strategy with the current combination of parameters
                    parameters = [shorter, longer]
                    done = 0
                    t = stateLength
                    numberOfShares = 0
                    state = tradingEnv.initState(data, stateLength)
                    while done == 0:
                        nextState, _, _, numberOfShares, done = tradingEnv.step(chooseAction(state, parameters), data, t, numberOfShares, transactionCost, \
                                stateLength, epsilon, done)
                        state = nextState
                        t += 1

                    # Retrieve the performance associated with this simulation (Sharpe Ratio)
                    performance = tradingPerformance.computeSharpeRatio(data)
                    results[i][j] = performance

                    # Track the best performance and parameters
                    if(performance > bestPerformance):
                        bestShort = shorter
                        bestLong = longer
                        bestPerformance = performance
                    
                    # Reset of the trading environment
                    data = tradingEnv.reset_data(data)
                    count += 1

                j += 1
            i += 1
            j = 0

        # Execute once again the strategy associated with the best parameters simulated
        parameters = [bestShort, bestLong]
        return parameters

    parameters = training(dataTraining)
    dataTesting = tradingEnv.reset_data(dataTesting)
    state = tradingEnv.initState(dataTesting, stateLength)
    done = 0
    t = stateLength
    numberOfShares = 0

    while done == 0:
            
        nextState, _, _, numberOfShares, done = tradingEnv.step(chooseAction(state, parameters), dataTesting, t, numberOfShares, transactionCost,\
                stateLength, epsilon, done)
        state = nextState
        t += 1
    
    return dataTesting