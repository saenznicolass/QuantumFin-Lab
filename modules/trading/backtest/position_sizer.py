from typing import Dict, Optional
import pandas as pd
import numpy as np
from ..strategies.technical.indicators import calculate_atr

class PositionSizer:
    """Position sizing and risk management component"""
    
    def __init__(self, parameters: Dict):
        """
        Initialize position sizer with risk parameters
        
        Parameters:
        -----------
        parameters : Dict
            Risk and position sizing parameters
            {
                'risk_per_trade': 0.02,  # 2% risk per trade
                'max_position_size': 0.2,  # 20% max position
                'atr_factor': 2.0,        # ATR multiplier
                'volatility_adjust': True, # Use volatility adjustment
                'portfolio_value': 100000  # Initial portfolio value
            }
        """
        self.params = parameters
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """Validate position sizing parameters"""
        required_params = {
            'risk_per_trade': 0.02,
            'max_position_size': 0.2,
            'portfolio_value': 100000
        }
        
        # Set defaults for missing parameters
        for param, default in required_params.items():
            if param not in self.params:
                self.params[param] = default
                
        if self.params['risk_per_trade'] > 0.15:  # Max 15% risk per trade
            self.params['risk_per_trade'] = 0.15
            
        if self.params['max_position_size'] > 0.25:  # Max 25% position size
            self.params['max_position_size'] = 0.25
    
    def calculate_position_size(
        self,
        price: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk parameters"""
        # Calculate risk amount in currency
        risk_amount = self.params['portfolio_value'] * self.params['risk_per_trade']
        
        # Calculate position size based on stop loss
        risk_per_unit = abs(price - stop_loss) if stop_loss > 0 else price * 0.02
        position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        # Apply maximum position size constraint
        max_position = (self.params['portfolio_value'] * 
                       self.params['max_position_size']) / price
        
        # Apply volatility adjustment if provided
        if volatility is not None and volatility > 0:
            vol_adjustment = 1.0 / (1.0 + volatility/price)
            position_size *= vol_adjustment
        
        return min(position_size, max_position)

    def adjust_for_volatility(
        self,
        position_sizes: pd.Series,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Adjust position sizes based on market volatility
        
        Parameters:
        -----------
        position_sizes : pd.Series
            Base position sizes
        data : pd.DataFrame
            OHLCV data for volatility calculation
        """
        # Calculate ATR-based volatility
        atr = calculate_atr(data, period=14)
        atr_pct = atr / data['Close']
        
        # Calculate volatility ratio
        vol_ratio = atr_pct / atr_pct.rolling(window=20).mean()
        
        # Adjust position sizes
        adjusted_sizes = position_sizes * (1 / vol_ratio).clip(0.5, 2.0)
        
        return adjusted_sizes
    
    def get_risk_metrics(self, positions: pd.DataFrame) -> Dict:
        """Calculate risk metrics for current positions"""
        metrics = {
            'total_exposure': positions.abs().sum(),
            'net_exposure': positions.sum(),
            'largest_position': positions.abs().max(),
            'number_of_positions': (positions != 0).sum()
        }
        
        # Calculate portfolio concentration
        if metrics['total_exposure'] > 0:
            metrics['concentration'] = metrics['largest_position'] / metrics['total_exposure']
        else:
            metrics['concentration'] = 0.0
            
        return metrics

    def get_current_exposure(self) -> float:
        """
        Temporary placeholder to avoid AttributeError.
        Implement logic here if real-time exposure tracking is needed.
        """
        return 0.0
