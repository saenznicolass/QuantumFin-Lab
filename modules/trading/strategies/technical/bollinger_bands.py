import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .base import TechnicalStrategy
from .indicators import calculate_atr

class BollingerBreakoutStrategy(TechnicalStrategy):
    """Enhanced Bollinger Bands Strategy Implementation"""
    
    def __init__(self, parameters: Dict, data: pd.DataFrame):
        super().__init__(parameters, data)
        self.required_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'volume_factor': 1.5,
            'atr_period': 14,
            'position_scale': 0.1
        }
        self.params = {**self.required_params, **parameters}
    
    def calculate_bollinger_bands(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        mid = self.data['Close'].rolling(window=self.params['bb_period']).mean()
        std = self.data['Close'].rolling(window=self.params['bb_period']).std()
        
        upper = mid + (std * self.params['bb_std'])
        lower = mid - (std * self.params['bb_std'])
        
        return upper, mid, lower
    
    def calculate_volume_signal(self) -> pd.Series:
        """Calculate volume-based confirmation signal"""
        volume = self.data['Volume']
        avg_volume = volume.rolling(window=self.params['bb_period']).mean()
        return volume > (avg_volume * self.params['volume_factor'])
    
    def _generate_raw_signals(self) -> pd.DataFrame:
        """Generate raw signals based on Bollinger Bands breakout"""
        upper, mid, lower = self.calculate_bollinger_bands()
        volume_confirm = self.calculate_volume_signal()
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=self.data.index, columns=['signal'], data=0)
        
        # Generate signals with volume confirmation
        long_condition = (
            (self.data['Close'] > upper) & 
            volume_confirm &
            (self.data['Close'].diff() > 0)
        )
        
        short_condition = (
            (self.data['Close'] < lower) & 
            volume_confirm &
            (self.data['Close'].diff() < 0)
        )
        
        # Assign signals using numpy where for efficiency
        signals['signal'] = np.where(
            long_condition, 1,
            np.where(short_condition, -1, 0)
        )
        
        return signals


class AdaptiveVolatilityStrategy(TechnicalStrategy):
    """Adaptive Volatility Breakout Strategy Implementation"""
    
    def __init__(self, parameters: Dict, data: pd.DataFrame):
        super().__init__(parameters, data)
        self.required_params = {
            'keltner_period': 20,
            'atr_factor': 2.0,
            'momentum_period': 10,
            'min_volatility': 0.01
        }
        self.params = {**self.required_params, **parameters}
    
    def calculate_keltner_channels(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        mid = typical_price.rolling(window=self.params['keltner_period']).mean()
        atr = calculate_atr(self.data, self.params['keltner_period'])
        
        upper = mid + (atr * self.params['atr_factor'])
        lower = mid - (atr * self.params['atr_factor'])
        
        return upper, mid, lower
    
    def calculate_momentum(self) -> pd.Series:
        """Calculate momentum indicator"""
        return self.data['Close'].diff(self.params['momentum_period'])
    
    def _generate_raw_signals(self) -> pd.DataFrame:
        """Generate raw signals based on adaptive volatility breakout"""
        upper, mid, lower = self.calculate_keltner_channels()
        momentum = self.calculate_momentum()
        atr = calculate_atr(self.data, self.params['keltner_period'])
        
        volatility_ratio = atr / atr.rolling(window=self.params['keltner_period']).mean()
        signals = pd.DataFrame(index=self.data.index, columns=['signal'], data=0)
        
        volatility_filter = volatility_ratio > self.params['min_volatility']
        
        long_condition = (
            (self.data['Close'] > upper) & 
            (momentum > 0) & 
            volatility_filter
        )
        short_condition = (
            (self.data['Close'] < lower) & 
            (momentum < 0) & 
            volatility_filter
        )
        
        position_scale = volatility_ratio.clip(0.5, 2.0)
        signals['signal'] = np.where(
            long_condition, 1 * position_scale,
            np.where(short_condition, -1 * position_scale, 0)
        )
        return signals
