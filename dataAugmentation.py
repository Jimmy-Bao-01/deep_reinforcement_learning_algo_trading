import copy
import numpy as np

# Default ranges for the parameters of the data augmentation techniques 
shiftRange = [0]
stretchRange = [1]
filterRange = [5]
noiseRange = [0]
    
def shiftTimeSeries(tradingEnv, shiftMagnitude=0):
    """
    GOAL: Generate a new trading environment by simply shifting up or down
            the volume time series.
    
    INPUTS: - tradingEnv: Original trading environment to augment.
            - shiftMagnitude: Magnitude of the shift.
    
    OUTPUTS: - newTradingEnv: New trading environment generated.
    """

    # Creation of the new trading environment
    newTradingEnv = copy.deepcopy(tradingEnv)

    # Constraint on the shift magnitude
    if shiftMagnitude < 0:
        minValue = np.min(tradingEnv.data['Volume'])
        shiftMagnitude = max(-minValue, shiftMagnitude)
    
    # Shifting of the volume time series
    newTradingEnv['Volume'] += shiftMagnitude

    # Return the new trading environment generated
    return newTradingEnv


def streching(tradingEnv, factor=1):
    """
    GOAL: Generate a new trading environment by stretching
            or contracting the original price time series, by 
            multiplying the returns by a certain factor.
    
    INPUTS: - tradingEnv: Original trading environment to augment.
            - factor: Stretching/contraction factor.
    
    OUTPUTS: - newTradingEnv: New trading environment generated.
    """

    # Creation of the new trading environment
    newTradingEnv = copy.deepcopy(tradingEnv)

    # Application of the stretching/contraction operation
    returns = newTradingEnv['Close'].pct_change() * factor
    for i in range(1, len(newTradingEnv.index)):
        newTradingEnv['Close'].iloc[i] = newTradingEnv['Close'].iloc[i-1] * (1 + returns.iloc[i])
        newTradingEnv['Low'].iloc[i] = newTradingEnv['Close'].iloc[i] * tradingEnv['Low'].iloc[i] / tradingEnv['Close'].iloc[i]
        newTradingEnv['High'].iloc[i] = newTradingEnv['Close'].iloc[i] * tradingEnv['High'].iloc[i] / tradingEnv['Close'].iloc[i]
        newTradingEnv['Open'].iloc[i] = newTradingEnv['Close'].iloc[i-1]

    # Return the new trading environment generated
    return newTradingEnv


def noiseAddition(tradingEnv, stdev=1):
    """
    GOAL: Generate a new trading environment by adding some gaussian
            random noise to the original time series.
    
    INPUTS: - tradingEnv: Original trading environment to augment.
            - stdev: Standard deviation of the generated white noise.
    
    OUTPUTS: - newTradingEnv: New trading environment generated.
    """

    # Creation of a new trading environment
    newTradingEnv = copy.deepcopy(tradingEnv)

    # Generation of the new noisy time series
    for i in range(1, len(newTradingEnv.index)):
        # Generation of artificial gaussian random noises
        price = newTradingEnv['Close'].iloc[i]
        volume = newTradingEnv['Volume'].iloc[i]
        priceNoise = np.random.normal(0, stdev*(price/100))
        volumeNoise = np.random.normal(0, stdev*(volume/100))

        # Addition of the artificial noise generated
        newTradingEnv['Close'].iloc[i] *= (1 + priceNoise/100)
        newTradingEnv['Low'].iloc[i] *= (1 + priceNoise/100)
        newTradingEnv['High'].iloc[i] *= (1 + priceNoise/100)
        newTradingEnv['Volume'].iloc[i] *= (1 + volumeNoise/100)
        newTradingEnv['Open'].iloc[i] = newTradingEnv['Close'].iloc[i-1]

    # Return the new trading environment generated
    return newTradingEnv


def lowPassFilter(tradingEnv, order=5):
    """
    GOAL: Generate a new trading environment by filtering
            (low-pass filter) the original time series.
    
    INPUTS: - tradingEnv: Original trading environment to augment.
            - order: Order of the filtering operation.
    
    OUTPUTS: - newTradingEnv: New trading environment generated.
    """

    # Creation of a new trading environment
    newTradingEnv = copy.deepcopy(tradingEnv)

    # Application of a filtering (low-pass) operation
    newTradingEnv['Close'] = newTradingEnv['Close'].rolling(window=order).mean()
    newTradingEnv['Low'] = newTradingEnv['Low'].rolling(window=order).mean()
    newTradingEnv['High'] = newTradingEnv['High'].rolling(window=order).mean()
    newTradingEnv['Volume'] = newTradingEnv['Volume'].rolling(window=order).mean()
    for i in range(order):
        newTradingEnv['Close'].iloc[i] = tradingEnv['Close'].iloc[i]
        newTradingEnv['Low'].iloc[i] = tradingEnv['Low'].iloc[i]
        newTradingEnv['High'].iloc[i] = tradingEnv['High'].iloc[i]
        newTradingEnv['Volume'].iloc[i] = tradingEnv['Volume'].iloc[i]
    newTradingEnv['Open'] = newTradingEnv['Close'].shift(1)
    newTradingEnv['Open'].iloc[0] = tradingEnv['Open'].iloc[0]

    # Return the new trading environment generated
    return newTradingEnv


def generate(tradingEnv):
    """
    Generate a set of new trading environments based on the data
    augmentation techniques implemented.
    
    :param: - tradingEnv: Original trading environment to augment.
    
    :return: - tradingEnvList: List of trading environments generated
                                by data augmentation techniques.
    """

    # Application of the data augmentation techniques to generate the new trading environments
    tradingEnvList = []
    for shift in shiftRange:
        tradingEnvShifted = shiftTimeSeries(tradingEnv, shift)
        for stretch in stretchRange:
            tradingEnvStretched = streching(tradingEnvShifted, stretch)
            for order in filterRange:
                tradingEnvFiltered = lowPassFilter(tradingEnvStretched, order)
                for noise in noiseRange:
                    tradingEnvList.append(noiseAddition(tradingEnvFiltered, noise))
    return tradingEnvList
