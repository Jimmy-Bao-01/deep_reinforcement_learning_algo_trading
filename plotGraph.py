import matplotlib as plt
import pandas as pd


splittingDate = '2018-01-01'

def render(data, stockName):
    fig = plt.figure(figsize=(15, 11))
    ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Date')
    ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Date')
    
    data['Close'].plot(ax=ax1, color='blue', lw=2.)
    ax1.plot(data[data['Action'] == 1].index, data['Close'][data['Action'] == 1], '^', markersize=5, color='g')
    ax1.plot(data[data['Action'] == -1].index, data['Close'][data['Action'] == -1], 'v', markersize=5, color='r')
    
    data['Money'].plot(ax=ax2, color='blue', lw=2.)
    ax2.plot(data[data['Action'] == 1].index, data['Money'][data['Action'] == 1], '^', markersize=5, color='g')
    ax2.plot(data[data['Action'] == -1].index, data['Money'][data['Action'] == -1], 'v', markersize=5, color='r')
    
    ax1.legend(['Price', 'Long', 'Short'], loc='best')
    ax2.legend(['Capital', 'Long', 'Short'], loc='best')
    
    plt.savefig(f'figures/{stockName}_test_rendering.png')
    
def plotEntireTrading(trainingEnv, testingEnv):
    """
    GOAL: Plot the entire trading activity, with both the training
            and testing phases rendered on the same graph for
            comparison purposes.
    
    INPUTS: - trainingEnv: Trading environment for training.
            - testingEnv: Trading environment for testing.
    
    OUTPUTS: /
    """

    # Artificial trick to assert the continuity of the Money curve
    ratio = trainingEnv['Money'][-1]/testingEnv['Money'][0]
    testingEnv['Money'] = ratio * testingEnv['Money']

    # Concatenation of the training and testing trading dataframes
    dataframes = [trainingEnv, testingEnv]
    data = pd.concat(dataframes)

    # Set the Matplotlib figure and subplots
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
    ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

    # Plot the first graph -> Evolution of the stock market price
    trainingEnv['Close'].plot(ax=ax1, color='blue', lw=2)
    testingEnv['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_') 
    ax1.plot(data.loc[data['Action'] == 1.0].index, 
                data['Close'][data['Action'] == 1.0],
                '^', markersize=5, color='green')   
    ax1.plot(data.loc[data['Action'] == -1.0].index, 
                data['Close'][data['Action'] == -1.0],
                'v', markersize=5, color='red')
    
    # Plot the second graph -> Evolution of the trading capital
    trainingEnv['Money'].plot(ax=ax2, color='blue', lw=2)
    testingEnv['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_') 
    ax2.plot(data.loc[data['Action'] == 1.0].index, 
                data['Money'][data['Action'] == 1.0],
                '^', markersize=5, color='green')   
    ax2.plot(data.loc[data['Action'] == -1.0].index, 
                data['Money'][data['Action'] == -1.0],
                'v', markersize=5, color='red')

    # Plot the vertical line seperating the training and testing datasets
    ax1.axvline(pd.Timestamp(splittingDate), color='black', linewidth=2.0)
    ax2.axvline(pd.Timestamp(splittingDate), color='black', linewidth=2.0)
    
    # Generation of the two legends and plotting
    ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
    ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
    # plt.savefig(''.join(['Figures/', str(trainingEnv.marketSymbol), '_TrainingTestingRendering', '.png'])) 
    plt.show()