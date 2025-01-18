import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .base import TechnicalStrategy
from .indicators import calculate_rsi, calculate_macd, calculate_adx


class RSIMACDStrategy(TechnicalStrategy):
    """RSI-MACD Confluence Strategy Implementation"""
    
    def __init__(self, parameters: Dict, data: pd.DataFrame):
        super().__init__(parameters, data)
        self.required_params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'min_trend_strength': 0.001
        }
        self.params = {**self.required_params, **parameters}
        
    def _generate_raw_signals(self) -> pd.DataFrame:
        """Generate raw trading signals based on RSI and MACD confluence"""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame with OHLCV columns")
            
        # Calculate indicators
        close_prices = self.data['Close']
        rsi = calculate_rsi(close_prices, self.params['rsi_period'])
        macd, signal, hist = calculate_macd(
            close_prices, 
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        
        # Initialize signals DataFrame with required columns
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0  # Initialize signal column
        
        # Generate conditions
        long_condition = (
            (rsi < self.params['rsi_oversold']) &  
            (macd > signal) &  
            (hist > self.params['min_trend_strength'])
        )
        
        short_condition = (
            (rsi > self.params['rsi_overbought']) &  
            (macd < signal) &  
            (hist < -self.params['min_trend_strength'])
        )
        
        # Assign signals using numpy where for efficiency
        signals['signal'] = np.where(
            long_condition, 1,
            np.where(short_condition, -1, 0)
        )
        
        return signals


class DynamicMomentumStrategy(TechnicalStrategy):
    """Dynamic Momentum Strategy with Adaptive RSI Implementation"""
    
    def __init__(self, parameters: Dict, data: pd.DataFrame):
        super().__init__(parameters, data)
        self.required_params = {
            'rsi_period': 14,
            'roc_period': 10,
            'adx_period': 14,
            'adx_threshold': 25,
            'band_factor': 1.5,
            'position_size': 0.1
        }
        self.params = {**self.required_params, **parameters}
    
    def calculate_adaptive_bands(self, rsi: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate adaptive RSI bands based on volatility"""
        rsi_std = rsi.rolling(window=self.params['rsi_period']).std()
        mid_line = 50
        band_width = self.params['band_factor'] * rsi_std
        
        upper_band = mid_line + band_width
        lower_band = mid_line - band_width
        
        # Ensure bands stay within RSI range (0-100)
        upper_band = upper_band.clip(50, 95)
        lower_band = lower_band.clip(5, 50)
        
        return upper_band, lower_band
    
    def calculate_roc(self, prices: pd.Series) -> pd.Series:
        """Calculate Rate of Change"""
        return ((prices - prices.shift(self.params['roc_period'])) / 
                prices.shift(self.params['roc_period'])) * 100
    
    def _generate_raw_signals(self) -> pd.DataFrame:
        """Generate raw signals based on dynamic momentum rules"""
        close_prices = self.data['Close']
        
        # Calculate indicators
        rsi = calculate_rsi(close_prices, self.params['rsi_period'])
        roc = self.calculate_roc(close_prices)
        adx = calculate_adx(self.data, self.params['adx_period'])
        
        # Calculate adaptive RSI bands
        upper_band, lower_band = self.calculate_adaptive_bands(rsi)
        
        # Align and drop rows with missing data
        df = pd.DataFrame({
            'Close': self.data['Close'],
            'RSI': rsi,
            'ROC': roc,
            'ADX': adx
        }, index=self.data.index).dropna()

        # Early return if no data
        if len(df) < 2:
            return pd.DataFrame(index=self.data.index, columns=['signal'], data=0)
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        trend_filter = df['ADX'] > self.params['adx_threshold']
        
        long_condition = (
            (df['RSI'] < lower_band.reindex(df.index)) &  
            (df['ROC'] > 0) &  
            trend_filter
        )
        
        short_condition = (
            (df['RSI'] > upper_band.reindex(df.index)) &  
            (df['ROC'] < 0) &  
            trend_filter
        )
        
        signals['signal'] = np.where(
            long_condition, 1,
            np.where(short_condition, -1, 0)
        )
        
        return signals
