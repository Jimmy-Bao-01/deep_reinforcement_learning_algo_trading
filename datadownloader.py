import yfinance as yf
import pandas as pd
import numpy as np
import os


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.columns = stock_data.columns.droplevel(1)
    # stock_data.drop(columns=["Adj Close"], inplace=True)
    return stock_data

def save_yfinance_data_to_csv(df, start_date, end_date, ticker):
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Format the filename
    filename = f"data/{ticker}_{start_date}_{end_date}.csv"

    # Save the DataFrame as CSV
    df.to_csv(filename, index=True)  # Ensure index (dates) is kept

if __name__ == "__main__":
    
    tickers = ["AAPL"]
    stocks = {
    'Dow Jones' : 'DIA',
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ',
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Facebook' : 'FB',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Twitter' : 'TWTR',
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
    'Shell' : 'RDSA.AS',
    'PetroChina' : 'PTR',
    'Tesla' : 'TSLA',
    'Volkswagen' : 'VOW3.DE',
    'Toyota' : '7203.T',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T'
    }
    start_date = "2012-01-01"
    start_validation_date = "2016-01-01"
    splitting_date = '2018-01-01'
    end_date = '2020-01-01'
    
    for ticker in stocks.values():
        if ticker == "AAPL":
            val_data = get_stock_data(ticker, start_validation_date, splitting_date)
            save_yfinance_data_to_csv(val_data, start_validation_date, splitting_date, ticker)
        training_data = get_stock_data(ticker, start_date, splitting_date)
        save_yfinance_data_to_csv(training_data, start_date, splitting_date, ticker)
        testing_data = get_stock_data(ticker, splitting_date, end_date)
        save_yfinance_data_to_csv(testing_data, splitting_date, end_date, ticker)
