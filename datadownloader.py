"""
Goal: Downloading stock market data from Yahoo Finance and saving it as CSV files.
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import yfinance as yf
import os
import pandas as pd

###############################################################################
################################ Global variables #############################
###############################################################################

 # Dictionary listing the 30 stocks considered as testbench
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

# Variables defining the default trading horizon
start_date = "2012-01-01"
start_validation_date = "2016-01-01"
splitting_date = '2018-01-01'
end_date = '2020-01-01'

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.columns = stock_data.columns.droplevel(1)
    # stock_data.set_index(stock_data['Date'], inplace=True)
    return stock_data

def dataframeToCSV(df, name):
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Format the filename
    filename = name + ".csv"

    # Save the DataFrame as CSV
    df.to_csv(filename, index=True)  # Ensure index (dates) is kept
    
def getDailyData(marketSymbol, startingDate, endingDate):
    """
    GOAL: Downloding daily stock market data from the Yahoo Finance API. 
    
    INPUTS:     - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
        
    OUTPUTS:    - data: Pandas dataframe containing the stock market data.
    """
    
    data = pd.read_csv('data/'+marketSymbol+'_'+startingDate+'_'+endingDate+'.csv')
    return data

def CSVToDataframe(name):
    """
    GOAL: Loading a CSV file into a dataframe.
    
    INPUTS:     - name: Name of the CSV file.   
        
    OUTPUTS:    - dataframe: Pandas dataframe loaded.
    """
    
    path = name + '.csv'
    return pd.read_csv(path,
                        header=0,
                        index_col='Date',
                        parse_dates=True)


if __name__ == "__main__":
    for stock in stocks.values():
        filename_train = 'data/'+stock+'_'+start_validation_date+'_'+splitting_date+'.csv'
        filename_test = 'data/'+stock+'_'+splitting_date+'_'+end_date+'.csv'
        if stock == "AAPL":
            filename_split = 'data/'+stock+'_'+start_validation_date+'_'+splitting_date+'.csv'
            val_data = get_stock_data(stock, start_validation_date, splitting_date)
            dataframeToCSV(val_data, filename_split)
        training_data = get_stock_data(stock, start_date, splitting_date)
        dataframeToCSV(training_data, filename_train)
        testing_data = get_stock_data(stock, splitting_date, end_date)
        dataframeToCSV(testing_data, filename_test)
