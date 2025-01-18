import pandas as pd
import numpy as np
from .base import TechnicalStrategy

class TripleMAStrategy(TechnicalStrategy):
    """Triple Moving Average Strategy Implementation"""
    
    def __init__(self, parameters: dict, data: pd.DataFrame):
        super().__init__(parameters, data)
        self.required_params = {
            'short_window': 5,
            'medium_window': 21,
            'long_window': 63,
            'min_trend_strength': 0.02,
            'position_size': 0.1
        }
        # Update default parameters with provided ones
        self.params = {**self.required_params, **parameters}

    def _generate_raw_signals(self) -> pd.DataFrame:
        """Generate raw signals based on triple MA crossover rules"""
        # Calculate EMAs
        close_prices = self.data['Close']
        short_ema = close_prices.ewm(span=self.params['short_window']).mean()
        medium_ema = close_prices.ewm(span=self.params['medium_window']).mean()
        long_ema = close_prices.ewm(span=self.params['long_window']).mean()
        
        # Initialize DataFrame
        signals = pd.DataFrame(index=self.data.index, columns=['signal'], data=0)
        
        # Calculate conditions
        long_condition = (
            (short_ema > medium_ema) & 
            (medium_ema > long_ema) & 
            (short_ema - long_ema > self.params['min_trend_strength'])
        )
        
        short_condition = (
            (short_ema < medium_ema) & 
            (medium_ema < long_ema) & 
            (long_ema - short_ema > self.params['min_trend_strength'])
        )
        
        # Assign signals using numpy where for efficiency
        signals['signal'] = np.where(
            long_condition, 1,
            np.where(short_condition, -1, 0)
        )
        
        return signals

class AdaptiveEMAStrategy(TechnicalStrategy):
    """Adaptive EMA Strategy with volatility-based adjustments"""
    
    def __init__(self, parameters: dict, data: pd.DataFrame):
        super().__init__(parameters, data)
        self.required_params = {
            'base_period': 21,
            'atr_period': 14,
            'volatility_factor': 2.0,
            'adaptation_rate': 0.1
        }
        self.params = {**self.required_params, **parameters}
    
    def calculate_atr(self) -> pd.Series:
        """Calculate Average True Range"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.params['atr_period']).mean()
        
        return atr

    def _generate_raw_signals(self) -> pd.DataFrame:
        """Generate raw signals for Adaptive EMA strategy"""
        # Loop-based adaptive EMA to avoid int conversion error:
        atr = self.calculate_atr()
        rolling_std = self.data['Close'].rolling(window=self.params['atr_period']).std().fillna(1)
        volatility_adjustment = (self.params['volatility_factor'] * atr / rolling_std).fillna(1.0)

        close = self.data['Close']
        adaptive_ema = pd.Series(index=close.index, dtype=float)
        adaptive_ema.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            period = max(2, int(self.params['base_period'] * volatility_adjustment.iloc[i]))
            alpha = 2.0 / (period + 1.0)
            adaptive_ema.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * adaptive_ema.iloc[i - 1]

        signals = pd.DataFrame(index=self.data.index, columns=['signal'], data=0)
        signals['signal'] = np.where(
            close > adaptive_ema, 1,
            np.where(close < adaptive_ema, -1, 0)
        )
        return signals