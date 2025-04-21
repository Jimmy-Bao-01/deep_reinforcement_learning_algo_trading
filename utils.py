import pandas as pd
import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe Ratio of a series of returns.
    
    Parameters:
    - returns: pd.Series or np.ndarray of returns
    - risk_free_rate: float, the risk-free rate (default is 0.0)
    
    Returns:
    - Sharpe Ratio: float
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)