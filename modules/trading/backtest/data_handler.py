from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from ..strategies.technical.indicators import (
    calculate_rsi, calculate_macd, calculate_adx,
    calculate_bollinger_bands, calculate_keltner_channels
)

class TechnicalDataHandler:
    """Handle market data for backtesting"""
    
    def __init__(self, data: pd.DataFrame):
        self.validate_data(data)
        self.data = self.preprocess_data(data)
        self.current_idx = 0
        
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be monotonically increasing")
            
        if data.isnull().any().any():
            raise ValueError("Data contains null values")
            
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data"""
        df = data.copy()
        
        # Ensure proper data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Add derived columns
        df['Returns'] = df['Close'].pct_change()
        df['TrueRange'] = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['High'] - df['Close'].shift()),
            'lc': abs(df['Low'] - df['Close'].shift())
        }).max(axis=1)
        
        df['ATR'] = df['TrueRange'].rolling(window=14).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Add gap information
        df['Gap'] = df['Open'] - df['Close'].shift()
        df['GapPct'] = df['Gap'] / df['Close'].shift()
        
        return df
    
    def calculate_indicators(self, params: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Calculate technical indicators based on provided parameters
        
        Parameters:
        -----------
        params : Dict[str, Dict]
            Dictionary of indicator parameters
            Example: {
                'rsi': {'period': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'period': 20, 'std': 2.0}
            }
        """
        results = {}
        
        for indicator, indicator_params in params.items():
            if indicator == 'rsi':
                results['rsi'] = calculate_rsi(
                    self.data['Close'],
                    period=indicator_params.get('period', 14)
                )
            
            elif indicator == 'macd':
                macd, signal, hist = calculate_macd(
                    self.data['Close'],
                    fast_period=indicator_params.get('fast', 12),
                    slow_period=indicator_params.get('slow', 26),
                    signal_period=indicator_params.get('signal', 9)
                )
                results['macd'] = pd.DataFrame({
                    'macd': macd,
                    'signal': signal,
                    'histogram': hist
                })
            
            elif indicator == 'bollinger':
                upper, mid, lower = calculate_bollinger_bands(
                    self.data['Close'],
                    period=indicator_params.get('period', 20),
                    num_std=indicator_params.get('std', 2.0)
                )
                results['bollinger'] = pd.DataFrame({
                    'upper': upper,
                    'middle': mid,
                    'lower': lower
                })
            
            elif indicator == 'keltner':
                upper, mid, lower = calculate_keltner_channels(
                    self.data['High'],
                    self.data['Low'],
                    self.data['Close'],
                    period=indicator_params.get('period', 20),
                    atr_factor=indicator_params.get('atr_factor', 2.0)
                )
                results['keltner'] = pd.DataFrame({
                    'upper': upper,
                    'middle': mid,
                    'lower': lower
                })
        
        self.indicators = results
        return results
    
    def get_analysis_ready_data(self, lookback: int = 0) -> pd.DataFrame:
        """
        Get data ready for analysis with all calculated indicators
        
        Parameters:
        -----------
        lookback : int
            Number of historical periods to include
        """
        result = self.data.copy()
        
        for indicator_name, indicator_data in self.indicators.items():
            if isinstance(indicator_data, pd.DataFrame):
                result = result.join(
                    indicator_data.add_prefix(f'{indicator_name}_')
                )
            else:
                result[indicator_name] = indicator_data
        
        if lookback > 0:
            result = result.tail(lookback)
        
        return result.dropna()
