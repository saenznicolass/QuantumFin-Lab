from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .data_handler import TechnicalDataHandler
from .signal_generator import SignalGenerator
from .performance_analyzer import PerformanceAnalyzer
from .position_sizer import PositionSizer
from ..strategies.technical.base import TechnicalStrategy

class BacktestEngine:
    """Engine for running technical strategy backtests"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: TechnicalStrategy,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0001,  # 1 pip default slippage
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV market data
        strategy : TechnicalStrategy
            Trading strategy instance
        initial_capital : float
            Initial capital for trading
        commission : float
            Trading commission (e.g., 0.001 for 0.1%)
        slippage : float
            Slippage per trade (e.g., 0.0001 for 1 pip)
        risk_free_rate : float
            Annual risk-free rate for performance metrics
        """
        self.data_handler = TechnicalDataHandler(data)
        self.strategy = strategy
        self.signal_generator = SignalGenerator(strategy)
        self.position_sizer = PositionSizer({
            'risk_per_trade': 0.02,
            'max_position_size': 0.2,
            'portfolio_value': initial_capital
        })
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.results = None
        
    def _apply_slippage_and_commission(
        self, 
        price: float, 
        direction: int,
        position_size: float
    ) -> Tuple[float, float]:
        """
        Apply slippage and commission to a trade
        Returns executed price and commission cost
        """
        # Apply slippage in the direction of the trade
        executed_price = price * (1 + direction * self.slippage)
        
        # Calculate commission
        commission_cost = abs(position_size * executed_price * self.commission)
        
        return executed_price, commission_cost
        
    def run_backtest(
        self,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """Execute backtest for the strategy"""
        try:
            # Get strategy results
            strategy_results = self.strategy.run_strategy()
            
            # Generate trading signals
            entries = self.signal_generator.generate_entry_signals()
            exits = self.signal_generator.generate_exit_signals(entries, stop_loss, take_profit)
            
            # Validate signals
            self.signal_generator.validate_signals(pd.concat([entries, exits], axis=1))
            
            # Initialize results tracking
            portfolio_value = pd.Series(index=self.data_handler.data.index, data=self.initial_capital)
            positions = pd.Series(index=self.data_handler.data.index, data=0.0)
            cash = pd.Series(index=self.data_handler.data.index, data=self.initial_capital)
            trades = []
            
            # Run simulation
            for i in range(1, len(self.data_handler.data)):
                current_bar = self.data_handler.data.iloc[i]
                prev_bar = self.data_handler.data.iloc[i-1]
                
                # Process exits first
                if positions.iloc[i-1] != 0:
                    # Check for gap risk
                    gap_loss = False
                    if positions.iloc[i-1] > 0:
                        gap_loss = current_bar['Open'] < prev_bar['Close'] * (1 - stop_loss if stop_loss else 0)
                    else:
                        gap_loss = current_bar['Open'] > prev_bar['Close'] * (1 + stop_loss if stop_loss else np.inf)
                        
                    if gap_loss or exits.iloc[i]['long_exit'] or exits.iloc[i]['short_exit']:
                        # Execute at open price with slippage
                        exit_price, commission_cost = self._apply_slippage_and_commission(
                            current_bar['Open'],
                            -np.sign(positions.iloc[i-1]),
                            positions.iloc[i-1]
                        )
                        
                        # Calculate PnL
                        trade_pnl = positions.iloc[i-1] * (exit_price - prev_bar['Close'])
                        
                        # Update positions and cash
                        cash.iloc[i] = cash.iloc[i-1] + trade_pnl - commission_cost
                        positions.iloc[i] = 0
                        
                        # Record trade
                        trades.append({
                            'exit_date': self.data_handler.data.index[i],
                            'exit_price': exit_price,
                            'pnl': trade_pnl,
                            'commission': commission_cost,
                            'reason': 'gap_stop' if gap_loss else 'signal'
                        })
                    else:
                        # Mark-to-market open position
                        positions.iloc[i] = positions.iloc[i-1]
                        unrealized_pnl = positions.iloc[i] * (current_bar['Close'] - prev_bar['Close'])
                        cash.iloc[i] = cash.iloc[i-1] + unrealized_pnl
                
                # Process entries
                if positions.iloc[i] == 0:
                    if entries.iloc[i]['long_entry'] or entries.iloc[i]['short_entry']:
                        # Calculate position size
                        size = self.position_sizer.calculate_position_size(
                            current_bar['Open'],
                            current_bar['Open'] * (1 - stop_loss) if stop_loss else 0,
                            current_bar['High'] - current_bar['Low']
                        )
                        
                        direction = 1 if entries.iloc[i]['long_entry'] else -1
                        entry_price, commission_cost = self._apply_slippage_and_commission(
                            current_bar['Open'],
                            direction,
                            size
                        )
                        
                        positions.iloc[i] = direction * size
                        cash.iloc[i] = cash.iloc[i-1] - commission_cost
                        
                        trades.append({
                            'entry_date': self.data_handler.data.index[i],
                            'entry_price': entry_price,
                            'position': positions.iloc[i],
                            'commission': commission_cost
                        })
                
                # Update portfolio value
                portfolio_value.iloc[i] = cash.iloc[i] + (positions.iloc[i] * current_bar['Close'])
            
            # Calculate returns using actual executed prices
            returns = portfolio_value.pct_change().fillna(0)
            
            # Store results
            self.results = pd.DataFrame({
                'portfolio_value': portfolio_value,
                'positions': positions,
                'cash': cash,
                'returns': returns
            })
            
            # Calculate performance metrics with improved risk metrics
            analyzer = PerformanceAnalyzer(self.results, self.risk_free_rate)
            performance_report = analyzer.generate_report()
            
            # Ensure signals have the required columns
            signals = pd.concat([entries, exits], axis=1)
            if isinstance(signals, pd.DataFrame) and 'signal' not in signals.columns:
                if 'position' in signals.columns:
                    signals['signal'] = signals['position']
                else:
                    signals['signal'] = 0

            # Discard redundant signals if already holding the same position
            current_position = 0
            final_signals = []
            for dt, signal in signals['signal'].items():
                if (current_position == 1 and signal == 1) or (current_position == -1 and signal == -1):
                    continue
                if signal != 0:
                    current_position = signal
                final_signals.append((dt, signal))
            signals = pd.Series(dict(final_signals))

            return {
                'data': self.data_handler.data,
                'results': self.results,
                'trades': pd.DataFrame(trades),
                'performance': performance_report,
                'strategy_params': self.strategy.params,
                'signals': signals,
                'strategy_results': strategy_results  # Include original strategy results
            }
            
        except Exception as e:
            raise ValueError(f"Error running strategy: {str(e)}")
    
    def optimize_parameters(
        self,
        param_grid: Dict[str, List],
        metric: str = 'sharpe_ratio'
    ) -> Dict:
        """
        Optimize strategy parameters using grid search
        
        Parameters:
        -----------
        param_grid : Dict[str, List]
            Dictionary of parameters to optimize and their possible values
        metric : str
            Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
        """
        best_score = float('-inf')
        best_params = None
        results = []
        
        # Generate parameter combinations
        from itertools import product
        param_combinations = [
            dict(zip(param_grid.keys(), values))
            for values in product(*param_grid.values())
        ]
        
        # Test each combination
        for params in param_combinations:
            # Update strategy parameters
            self.strategy.params.update(params)
            
            # Run backtest
            backtest_result = self.run_backtest()
            score = backtest_result['performance']['Risk-Adjusted'][metric]
            
            results.append({
                'parameters': params,
                'score': score,
                'performance': backtest_result['performance']
            })
            
            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_results': pd.DataFrame(results)
        }
