import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt

def computePnL(data):

    # Compute the PnL
    PnL = data["Money"][-1] - data["Money"][0]
    return PnL


def computeAnnualizedReturn(data):

    # Compute the cumulative return over the entire trading horizon
    cumulativeReturn = data['Returns'].cumsum()
    cumulativeReturn = cumulativeReturn[-1]
    
    # Compute the time elapsed (in days)
    start = data.index[0].to_pydatetime()
    end = data.index[-1].to_pydatetime()     
    timeElapsed = end - start
    timeElapsed = timeElapsed.days

    # Compute the Annualized Return
    if(cumulativeReturn > -1):
        annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365/timeElapsed)) - 1)
    else:
        annualizedReturn = -100
    return annualizedReturn


def computeAnnualizedVolatility(data):
    
    # Compute the Annualized Volatility (252 trading days in 1 trading year)
    annualizedVolatily = 100 * np.sqrt(252) * data['Returns'].std()
    return annualizedVolatily


def computeSharpeRatio(data, riskFreeRate=0):

    # Compute the expected return
    expectedReturn = data['Returns'].mean()
    
    # Compute the returns volatility
    volatility = data['Returns'].std()
    
    # Compute the Sharpe Ratio (252 trading days in 1 year)
    if expectedReturn != 0 and volatility != 0:
        sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
    else:
        sharpeRatio = 0
    return sharpeRatio


def computeSortinoRatio(data, riskFreeRate=0):
    
    # Compute the expected return
    expectedReturn = np.mean(data['Returns'])
    
    # Compute the negative returns volatility
    negativeReturns = [returns for returns in data['Returns'] if returns < 0]
    volatility = np.std(negativeReturns)
    
    # Compute the Sortino Ratio (252 trading days in 1 year)
    if expectedReturn != 0 and volatility != 0:
        sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
    else:
        sortinoRatio = 0
    return sortinoRatio


def computeMaxDrawdown(data, plotting=False):

    # Compute both the Maximum Drawdown and Maximum Drawdown Duration
    capital = data['Money'].values
    through = np.argmax(np.maximum.accumulate(capital) - capital)
    if through != 0:
        peak = np.argmax(capital[:through])
        maxDD = 100 * (capital[peak] - capital[through])/capital[peak]
        maxDDD = through - peak
    else:
        maxDD = 0
        maxDDD = 0
        return maxDD, maxDDD

    # Plotting of the Maximum Drawdown if required
    if plotting:
        plt.figure(figsize=(10, 4))
        plt.plot(data['Money'], lw=2, color='Blue')
        plt.plot([data.iloc[[peak]].index, data.iloc[[through]].index],
                    [capital[peak], capital[through]], 'o', color='Red', markersize=5)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.savefig(''.join(['Figures/', 'MaximumDrawDown', '.png']))
        #plt.show()

    # Return of the results
    return maxDD, maxDDD


def computeProfitability(data):
    
    # Initialization of some variables
    good = 0
    bad = 0
    profit = 0
    loss = 0
    index = next((i for i in range(len(data.index)) if data['Action'][i] != 0), None)
    if index == None:
        profitability = 0
        averageProfitLossRatio = 0
        return profitability, averageProfitLossRatio
    money = data['Money'][index]

    # Monitor the success of each trade over the entire trading horizon
    for i in range(index+1, len(data.index)):
        if(data['Action'][i] != 0):
            delta = data['Money'][i] - money
            money = data['Money'][i]
            if(delta >= 0):
                good += 1
                profit += delta
            else:
                bad += 1
                loss -= delta

    # Special case of the termination trade
    delta = data['Money'][-1] - money
    if(delta >= 0):
        good += 1
        profit += delta
    else:
        bad += 1
        loss -= delta

    # Compute the Profitability
    profitability = 100 * good/(good + bad)
        
    # Compute the ratio average Profit/Loss  
    if(good != 0):
        profit /= good
    if(bad != 0):
        loss /= bad
    if(loss != 0):
        averageProfitLossRatio = profit/loss
    else:
        averageProfitLossRatio = float('Inf')

    return profitability, averageProfitLossRatio
    

def computeSkewness(data):

    # Compute the Skewness of the returns
    skewness = data["Returns"].skew()
    return skewness
    

def computePerformance(data):

    # Compute the entire set of performance indicators
    PnL = computePnL(data)
    annualizedReturn = computeAnnualizedReturn(data)
    annualizedVolatily = computeAnnualizedVolatility(data)
    profitability, averageProfitLossRatio = computeProfitability(data)
    sharpeRatio = computeSharpeRatio(data)
    sortinoRatio = computeSortinoRatio(data)
    maxDD, maxDDD = computeMaxDrawdown(data)
    skewness = computeSkewness(data)

    # Generate the performance table
    performanceTable = [["Profit & Loss (P&L)", "{0:.0f}".format(PnL)], 
                        ["Annualized Return", "{0:.2f}".format(annualizedReturn) + '%'],
                        ["Annualized Volatility", "{0:.2f}".format(annualizedVolatily) + '%'],
                        ["Sharpe Ratio", "{0:.3f}".format(sharpeRatio)],
                        ["Sortino Ratio", "{0:.3f}".format(sortinoRatio)],
                        ["Maximum Drawdown", "{0:.2f}".format(maxDD) + '%'],
                        ["Maximum Drawdown Duration", "{0:.0f}".format(maxDDD) + ' days'],
                        ["Profitability", "{0:.2f}".format(profitability) + '%'],
                        ["Ratio Average Profit/Loss", "{0:.3f}".format(averageProfitLossRatio)],
                        ["Skewness", "{0:.3f}".format(skewness)]]
    
    return performanceTable


def displayPerformance(data, name):

    # Generation of the performance table
    performanceTable = computePerformance(data)
    
    # Display the table in the console (Tabulate for the beauty of the print operation)
    headers = ["Performance Indicator", name]
    tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
    print(tabulation)
