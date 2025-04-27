from TDQN import TDQN, testing
import pandas as pd

import warnings
from pandas.errors import SettingWithCopyWarning  # <-- Add this line

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

stocks = {
    'Dow Jones' : 'DIA',
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ',
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Meta' : 'META',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Nokia' : 'NOK',
    'Philips' : 'PHIA.AS',
    'Siemens' : 'SIE.DE',
    'Baidu' : 'BIDU',
    'Alibaba' : 'BABA',
    'Tencent' : '0700.HK',
    'Sony' : '6758.T',
    'JPMorgan Chase' : 'JPM',
    'HSBC' : 'HSBC',
    'CCB' : '0939.HK',
    'ExxonMobil' : 'XOM',
    'Shell' : 'SHEL',
    'PetroChina' : '0857.HK',
    'Tesla' : 'TSLA',
    'Volkswagen' : 'VOW3.DE',
    'Toyota' : '7203.T',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T',
    # 'Twitter' : 'TWTR' # No accessible now
}

def simulation(stockName, startDate, splitingDate, endDate, displayTraining=False):
    """
    Simulate trading using a TDQN agent.

    Parameters:
    - stockName: The name of the stock to trade.
    - startDate: The start date for the simulation.
    - endDate: The end date for the simulation.
    - initialCash: The initial amount of cash for trading.
    - commission: The commission fee for each trade.
    - numEpisodes: The number of episodes to run the simulation.

    Returns:
    - None
    """
    
    stockName = stocks[stockName]
    
    if displayTraining:
        print('Training TDQN agent on', stockName)
        dataTest = pd.read_csv('data/'+stockName+'_'+startDate+'_'+splitingDate+'.csv').set_index('Date')
    else:
        print('Testing TDQN agent on', stockName)
        dataTest = pd.read_csv('data/'+stockName+'_'+splitingDate+'_'+endDate+'.csv').set_index('Date')
    
    # Train the TDQN agent
    TrainingData, MainNetwork = TDQN(stockName, startDate, splitingDate)
    
    # Test the agent
    TestedData, QValues0, QValues1 = testing(TrainingData, dataTest, MainNetwork)
    
    return TestedData, QValues0, QValues1
