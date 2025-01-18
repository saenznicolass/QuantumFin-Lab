from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class TechnicalStrategy(ABC):
    """Base class for all technical trading strategies"""
    
    def __init__(self, parameters: Dict, data: pd.DataFrame):
        self.params = parameters
        self.data = data.copy()
        self.reset_state()
        
    def reset_state(self):
        """Reset strategy state"""
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.last_signal = 0
        self.entry_price = 0
        self.entry_time = None
        
    @abstractmethod
    def _generate_raw_signals(self) -> pd.DataFrame:
        """Generate raw trading signals - to be implemented by subclasses"""
        pass
    
    def validate_signal_transition(self, current_position: int, new_signal: int, timestamp: pd.Timestamp) -> int:
        """
        Validate signal transitions based on current position
        Returns validated signal value
        """
        if current_position == new_signal:
            return 0  # No change if same position requested
            
        if current_position == 1:  # Currently long
            if new_signal < 0:  # Want to go short
                return -1  # Allow short after closing long
            return 0  # Exit to neutral
            
        elif current_position == -1:  # Currently short
            if new_signal > 0:  # Want to go long
                return 1  # Allow long after closing short
            return 0  # Exit to neutral
            
        else:  # Currently neutral
            return new_signal  # Can take any position
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate validated signals"""
        raw_signals = self._generate_raw_signals()
        
        # Basic validation and conversion
        if not isinstance(raw_signals, pd.DataFrame):
            raise ValueError("Raw signals must be a pandas DataFrame")
            
        # Initialize validated signals DataFrame
        signals = pd.DataFrame(index=raw_signals.index)
        signals['position'] = 0
        signals['signal'] = raw_signals['signal'].astype(float) if 'signal' in raw_signals.columns else 0.0
        signals['entry_price'] = 0.0
        signals['exit_price'] = 0.0
        
        current_position = 0
        
        for i in range(len(signals)):
            timestamp = signals.index[i]
            new_signal = signals['signal'].iloc[i]
            
            # Get validated signal
            validated_signal = self.validate_signal_transition(
                current_position, new_signal, timestamp)
            
            # Update position and prices
            if validated_signal != 0:
                signals.loc[timestamp, 'signal'] = validated_signal
                signals.loc[timestamp, 'entry_price'] = self.data.loc[timestamp, 'Close']
                
                if current_position != 0:  # If exiting a position
                    signals.loc[timestamp, 'exit_price'] = self.data.loc[timestamp, 'Close']
                    
                current_position = validated_signal
            
            signals.loc[timestamp, 'position'] = current_position
        
        # Ensure price columns are numeric
        signals['entry_price'] = signals['entry_price'].astype(float)
        signals['exit_price'] = signals['exit_price'].astype(float)
        
        return signals

    def run_strategy(self) -> Dict:
        """Execute strategy and return results"""
        self.reset_state()
        signals = self.generate_signals()
        
        # Ensure we include both signals and original data
        return {
            'data': self.data,
            'signals': signals,
            'parameters': self.params,
            'indicators': self._get_strategy_indicators()  # Optional method to add indicators
        }

    def _get_strategy_indicators(self) -> Dict:
        """Get strategy-specific indicators for plotting"""
        return {}  # Override in subclasses if needed
