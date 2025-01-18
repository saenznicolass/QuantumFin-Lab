import pandas as pd
import numpy as np
from typing import Tuple


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    
    gains = (delta.where(delta > 0, 0)).fillna(0)
    losses = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal line, and MACD Histogram"""
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    
    return macd, signal, hist


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX)"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate True Range
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate Directional Indicators
    pdi = 100 * pd.Series(pos_dm).rolling(window=period).mean() / atr
    ndi = 100 * pd.Series(neg_dm).rolling(window=period).mean() / atr
    
    # Calculate ADX
    dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    mid = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = mid + (std * num_std)
    lower = mid - (std * num_std)
    
    return upper, mid, lower

def calculate_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_factor: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels"""
    typical_price = (high + low + close) / 3
    mid = typical_price.rolling(window=period).mean()
    
    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    upper = mid + (atr * atr_factor)
    lower = mid - (atr * atr_factor)
    
    return upper, mid, lower


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'High', 'Low', 'Close' columns
    period : int
        Lookback period for ATR calculation
        
    Returns:
    --------
    pd.Series
        ATR values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    # True Range is the maximum of the three price ranges
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Average True Range
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    return atr