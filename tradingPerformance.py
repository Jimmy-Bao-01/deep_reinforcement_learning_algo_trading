"""
Goal: Accurately estimating the performance of a trading strategy.
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt



###############################################################################
######################### Class PerformanceEstimator ##########################
###############################################################################


"""
GOAL: Accurately estimating the performance of a trading strategy, by
        computing many different performance indicators.
    
VARIABLES: - data: Trading activity data from the trading environment.
            - PnL: Profit & Loss (performance indicator).
            - annualizedReturn: Annualized Return (performance indicator).
            - annualizedVolatily: Annualized Volatility (performance indicator).
            - profitability: Profitability (performance indicator).
            - averageProfitLossRatio: Average Profit/Loss Ratio (performance indicator).
            - sharpeRatio: Sharpe Ratio (performance indicator).
            - sortinoRatio: Sortino Ratio (performance indicator).
            - maxDD: Maximum Drawdown (performance indicator).
            - maxDDD: Maximum Drawdown Duration (performance indicator).
            - skewness: Skewness of the returns (performance indicator).
        
METHODS:   -  __init__: Object constructor initializing some class variables. 
            - computePnL: Compute the P&L.
            - computeAnnualizedReturn: Compute the Annualized Return.
            - computeAnnualizedVolatility: Compute the Annualized Volatility.
            - computeProfitability: Computate both the Profitability and the Average Profit/Loss Ratio.
            - computeSharpeRatio: Compute the Sharpe Ratio.
            - computeSortinoRatio: Compute the Sortino Ratio.
            - computeMaxDrawdown: Compute both the Maximum Drawdown and Maximum Drawdown Duration.
            - computeSkewness: Compute the Skewness of the returns.
            - computePerformance: Compute all the performance indicators.
            - displayPerformance: Display the entire set of performance indicators in a table.
"""

def computePnL(data):
    """
    GOAL: Compute the Profit & Loss (P&L) performance indicator, which
            quantifies the money gained or lost during the trading activity.
    
    INPUTS: /
    
    OUTPUTS:    - PnL: Profit or loss (P&L) performance indicator.
    """
    
    # Compute the PnL
    PnL = data["Money"][-1] - data["Money"][0]
    return PnL


def computeAnnualizedReturn(data):
    """
    GOAL: Compute the yearly average profit or loss (in %), called
            the Annualized Return performance indicator.
    
    INPUTS: /
    
    OUTPUTS:    - annualizedReturn: Annualized Return performance indicator.
    """
    
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
    """
    GOAL: Compute the Yearly Voltility of the returns (in %), which is
            a measurement of the risk associated with the trading activity.
    
    INPUTS: /
    
    OUTPUTS:    - annualizedVolatily: Annualized Volatility performance indicator.
    """
    
    # Compute the Annualized Volatility (252 trading days in 1 trading year)
    annualizedVolatily = 100 * np.sqrt(252) * data['Returns'].std()
    return annualizedVolatily


def computeSharpeRatio(data, riskFreeRate=0):
    """
    GOAL: Compute the Sharpe Ratio of the trading activity, which is one of
            the most suited performance indicator as it balances the brute
            performance and the risk associated with a trading activity.
    
    INPUTS:     - riskFreeRate: Return of an investment with a risk null.
    
    OUTPUTS:    - sharpeRatio: Sharpe Ratio performance indicator.
    """
    
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
    """
    GOAL: Compute the Sortino Ratio of the trading activity, which is similar
            to the Sharpe Ratio but does no longer penalize positive risk.
    
    INPUTS:     - riskFreeRate: Return of an investment with a risk null.
    
    OUTPUTS:    - sortinoRatio: Sortino Ratio performance indicator.
    """
    
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
    """
    GOAL: Compute both the Maximum Drawdown and the Maximum Drawdown Duration
            performance indicators of the trading activity, which are measurements
            of the risk associated with the trading activity.
    
    INPUTS: - plotting: Boolean enabling the maximum drawdown plotting.
    
    OUTPUTS:    - maxDD: Maximum Drawdown performance indicator.
                - maxDDD: Maximum Drawdown Duration performance indicator.
    """

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
    """
    GOAL: Compute both the percentage of trades that resulted
            in profit (Profitability), and the ratio between the
            average profit and the average loss (AverageProfitLossRatio).
    
    INPUTS: /
    
    OUTPUTS:    - profitability: Percentage of trades that resulted in profit.
                - averageProfitLossRatio: Ratio between the average profit
                                            and the average loss.
    """
    
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
    """
    GOAL: Compute the skewness of the returns, which is
            a measurement of the degree of distorsion
            from the symmetrical bell curve.
    
    INPUTS: /
    
    OUTPUTS:    - skewness: Skewness performance indicator.
    """
    
    # Compute the Skewness of the returns
    skewness = data["Returns"].skew()
    return skewness
    

def computePerformance(data):
    """
    GOAL: Compute the entire set of performance indicators.
    
    INPUTS: /
    
    OUTPUTS:    - performanceTable: Table summarizing the performance of 
                                    a trading strategy.
    """

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
    """
    GOAL: Compute and display the entire set of performance indicators
            in a table.
    
    INPUTS: - name: Name of the element (strategy or stock) analysed.
    
    OUTPUTS:    - performanceTable: Table summarizing the performance of 
                                    a trading activity.
    """
    
    # Generation of the performance table
    performanceTable = computePerformance(data)
    
    # Display the table in the console (Tabulate for the beauty of the print operation)
    headers = ["Performance Indicator", name]
    tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
    print(tabulation)
