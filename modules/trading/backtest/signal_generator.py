from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from ..strategies.technical.base import TechnicalStrategy

class SignalGenerator:
    """Generate and validate trading signals from technical strategies"""
    
    def __init__(self, strategy: TechnicalStrategy):
        """
        Initialize signal generator
        
        Parameters:
        -----------
        strategy : TechnicalStrategy
            Technical trading strategy instance
        """
        self.strategy = strategy
        self.signals = pd.DataFrame()
        self.metadata = {}
    
    def generate_entry_signals(self) -> pd.DataFrame:
        """Generate entry signals with proper state management"""
        strategy_results = self.strategy.run_strategy()
        
        if 'signals' not in strategy_results:
            raise ValueError("Strategy must return 'signals' in results dictionary")
            
        signals = strategy_results['signals'].copy()
        
        # Add additional validation for signal quality
        if signals['signal'].abs().sum() == 0:
            self.metadata['warning'] = "Strategy generated no signals"
            
        # Check for excessive trading
        signal_changes = signals['signal'].diff().abs().sum()
        if signal_changes > len(signals) * 0.1:  # More than 10% of bars have signals
            self.metadata['warning'] = "High frequency of signal changes detected"
            
        # Generate clean entry signals
        entries = pd.DataFrame(index=signals.index)
        entries['long_entry'] = ((signals['signal'] == 1) & 
                               (signals['signal'].shift(1) != 1)).astype(int)
        entries['short_entry'] = ((signals['signal'] == -1) & 
                                (signals['signal'].shift(1) != -1)).astype(int)
        
        # Add trade_id to track entry/exit pairs
        entries['trade_id'] = np.nan
        current_id = 0
        for i in range(len(entries)):
            if entries.iloc[i]['long_entry'] or entries.iloc[i]['short_entry']:
                entries.iloc[i, entries.columns.get_loc('trade_id')] = current_id
                current_id += 1
        
        # Add price levels for entry
        entries['entry_price'] = np.nan
        entries.loc[entries['long_entry'] == 1, 'entry_price'] = signals.loc[entries['long_entry'] == 1, 'entry_price']
        entries.loc[entries['short_entry'] == 1, 'entry_price'] = signals.loc[entries['short_entry'] == 1, 'entry_price']
        
        self.metadata.update({
            'n_long_signals': int(entries['long_entry'].sum()),
            'n_short_signals': int(entries['short_entry'].sum()),
            'avg_holding_period': self._calculate_avg_holding_period(signals)
        })
        
        return entries

    def generate_exit_signals(
        self,
        entries: pd.DataFrame,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None
    ) -> pd.DataFrame:
        """Generate exit signals with improved risk management"""
        signals = self.strategy.generate_signals()
        
        exits = pd.DataFrame(index=signals.index)
        exits['long_exit'] = ((signals['position'].shift(1) == 1) & 
                            (signals['position'] != 1)).astype(int)
        exits['short_exit'] = ((signals['position'].shift(1) == -1) & 
                             (signals['position'] != -1)).astype(int)
                             
        # Track high/low since entry for trailing stops
        if trailing_stop:
            highest_high = pd.Series(index=signals.index, dtype=float)
            lowest_low = pd.Series(index=signals.index, dtype=float)
            
            for i in range(1, len(signals)):
                if signals['position'].iloc[i-1] == 1:  # Long position
                    highest_high.iloc[i] = max(
                        signals['High'].iloc[i],
                        highest_high.iloc[i-1] if not pd.isna(highest_high.iloc[i-1]) else signals['High'].iloc[i]
                    )
                    # Add trailing stop exit
                    if (signals['Low'].iloc[i] < highest_high.iloc[i] * (1 - trailing_stop)):
                        exits.iloc[i]['long_exit'] = 1
                        
                elif signals['position'].iloc[i-1] == -1:  # Short position
                    lowest_low.iloc[i] = min(
                        signals['Low'].iloc[i],
                        lowest_low.iloc[i-1] if not pd.isna(lowest_low.iloc[i-1]) else signals['Low'].iloc[i]
                    )
                    # Add trailing stop exit
                    if (signals['High'].iloc[i] > lowest_low.iloc[i] * (1 + trailing_stop)):
                        exits.iloc[i]['short_exit'] = 1
        
        exits['exit_price'] = np.nan
        exits['exit_type'] = ''
        
        return exits
    
    def _apply_stops_and_targets(
        self, 
        exits: pd.DataFrame, 
        signals: pd.DataFrame,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ):
        """Apply stop loss and take profit levels"""
        price = self.strategy.data['Close']
        position = signals['position']
        entry_price = signals['entry_price']
        
        for i in range(1, len(exits)):
            if position.iloc[i-1] != 0 and entry_price.iloc[i-1] > 0:
                curr_price = price.iloc[i]
                entry = entry_price.iloc[i-1]
                
                if position.iloc[i-1] == 1:  # Long position
                    if (stop_loss and curr_price <= entry * (1 - stop_loss)) or \
                       (take_profit and curr_price >= entry * (1 + take_profit)):
                        exits.iloc[i, exits.columns.get_loc('long_exit')] = 1
                        
                elif position.iloc[i-1] == -1:  # Short position
                    if (stop_loss and curr_price >= entry * (1 + stop_loss)) or \
                       (take_profit and curr_price <= entry * (1 - take_profit)):
                        exits.iloc[i, exits.columns.get_loc('short_exit')] = 1

    def validate_signals(self, signals: pd.DataFrame) -> bool:
        """Validate signal consistency"""
        # Check for overlapping signals
        if ((signals['long_entry'] & signals['short_entry']).any() or
            (signals['long_exit'] & signals['short_exit']).any()):
            raise ValueError("Overlapping signals detected")
        
        # Validate entry/exit sequence
        long_position = False
        short_position = False
        
        for i in range(len(signals)):
            row = signals.iloc[i]
            
            # Entry validation
            if row['long_entry']:
                if long_position:
                    # Continuar sin lanzar excepción
                    continue
                long_position = True
            if row['short_entry']:
                if short_position:
                    continue
                short_position = True
            
            # Exit validation
            if row['long_exit']:
                if not long_position:
                    continue
                long_position = False
            if row['short_exit']:
                if not short_position:
                    continue
                short_position = False
        
        return True

    def _calculate_avg_holding_period(self, signals: pd.DataFrame) -> float:
        """Calculate average holding period of positions"""
        position_changes = signals['signal'].diff().abs() > 0
        holding_periods = []
        start_idx = None
        
        for i in range(len(signals)):
            if position_changes.iloc[i]:
                if start_idx is not None:
                    holding_periods.append(i - start_idx)
                start_idx = i
                
        return np.mean(holding_periods) if holding_periods else 0
