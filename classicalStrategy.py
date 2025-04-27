import math
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod

# from tradingPerformance import PerformanceEstimator



def buy_and_hold(tradingEnv, initialInvestment=100000):
    """
    GOAL: Buy and hold strategy for a given trading environment.
    
    INPUTS: - tradingEnv: Trading environment to evaluate.
            - initialInvestment: Initial investment amount.
    
    OUTPUTS: - finalValue: Final value of the investment.
    """
    
    # Calculate the final value of the investment
    finalValue = initialInvestment * (tradingEnv['Close'].iloc[-1] / tradingEnv['Close'].iloc[0])
    
    return finalValue