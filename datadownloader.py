import yfinance as yf
import pandas as pd
import numpy as np
import os


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.columns = stock_data.columns.droplevel(1)
    stock_data.drop(columns=["Adj Close"], inplace=True)
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
    start_date = "2012-01-01"
    start_validation_date = "2016-01-01"
    splitting_date = "2017-12-31"
    end_date = "2019-12-31"
    
    for ticker in tickers:
        if ticker == "AAPL":
            val_data = get_stock_data(ticker, start_validation_date, splitting_date)
            save_yfinance_data_to_csv(val_data, start_validation_date, splitting_date, ticker)
        training_data = get_stock_data(ticker, start_date, splitting_date)
        save_yfinance_data_to_csv(training_data, start_date, splitting_date, ticker)
        testing_data = get_stock_data(ticker, splitting_date, end_date)
        save_yfinance_data_to_csv(testing_data, splitting_date, end_date, ticker)
