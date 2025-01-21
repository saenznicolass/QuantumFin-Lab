from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .data_handler import TechnicalDataHandler
from .signal_generator import SignalGenerator
from .performance_analyzer import PerformanceAnalyzer
from .position_sizer import PositionSizer
from ..strategies.technical.base import TechnicalStrategy
from ...risk.metrics.drawdown import analyze_drawdowns  # Simplificar importaciones

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
        self.position_sizer = None
        
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
            # Validar que position_sizer esté configurado
            if self.position_sizer is None:
                raise ValueError("PositionSizer must be configured before running backtest")
            
            # Get strategy results
            strategy_results = self.strategy.run_strategy()
            
            # Generate trading signals
            entries = self.signal_generator.generate_entry_signals()
            exits = self.signal_generator.generate_exit_signals(entries, stop_loss, take_profit)
            
            # Validate signals
            self.signal_generator.validate_signals(pd.concat([entries, exits], axis=1))
            
            # Initialize results tracking for ALL dates
            all_dates = self.data_handler.data.index
            portfolio_value = pd.Series(index=all_dates, data=self.initial_capital)
            positions = pd.Series(index=all_dates, data=0.0)
            cash = pd.Series(index=all_dates, data=self.initial_capital)
            trades = []
            prev_entry_price = 0
            
            # Initialize trade tracking
            current_trade_id = 0
            trades = []
            open_position = {
                'trade_id': None,
                'entry_date': None,
                'entry_price': None,
                'position_size': 0,
                'entry_signal': None,
                'commission': 0
            }
            
            # Run simulation
            for i in range(1, len(all_dates)):
                current_date = all_dates[i]
                current_bar = self.data_handler.data.loc[current_date]
                prev_bar = self.data_handler.data.loc[all_dates[i-1]]
                
                # Default to previous values if no action
                positions.iloc[i] = positions.iloc[i-1]
                cash.iloc[i] = cash.iloc[i-1]
                
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
                            abs(positions.iloc[i-1])
                        )
                        
                        # Calculate PnL and record complete trade
                        trade_pnl = positions.iloc[i-1] * (exit_price - open_position['entry_price']) - commission_cost
                        
                        trades.append({
                            'trade_id': open_position['trade_id'],
                            'entry_date': open_position['entry_date'],
                            'entry_signal': open_position['entry_signal'],
                            'entry_price': open_position['entry_price'],
                            'exit_date': current_date,
                            'exit_price': exit_price,
                            'exit_signal': 'Gap Stop' if gap_loss else ('Long Exit' if positions.iloc[i-1] > 0 else 'Short Exit'),
                            'position_size': abs(positions.iloc[i-1]),
                            'pnl': trade_pnl,
                            'commission': open_position['commission'] + commission_cost
                        })
                        
                        # Reset position tracking
                        positions.iloc[i] = 0
                        cash.iloc[i] = cash.iloc[i-1] + trade_pnl
                        open_position = {
                            'trade_id': None,
                            'entry_date': None,
                            'entry_price': None,
                            'position_size': 0,
                            'entry_signal': None,
                            'commission': 0
                        }
                    else:
                        # Mark-to-market open position sin afectar el cash
                        positions.iloc[i] = positions.iloc[i-1]
                        unrealized_pnl = positions.iloc[i] * (current_bar['Close'] - prev_bar['Close'])
                        cash.iloc[i] = cash.iloc[i-1]
                        portfolio_value.iloc[i] = cash.iloc[i] + unrealized_pnl
                
                # Process entries for new positions only when no position is open
                if positions.iloc[i] == 0 and open_position['trade_id'] is None:
                    if entries.iloc[i]['long_entry'] or entries.iloc[i]['short_entry']:
                        # Calculate position size using position_sizer params
                        size = self.position_sizer.calculate_position_size(
                            current_bar['Open'],
                            current_bar['Open'] * (1 - (stop_loss if stop_loss else 0.02)),  # Default 2% stop
                            current_bar['High'] - current_bar['Low']  # Volatility estimate
                        )
                        
                        direction = 1 if entries.iloc[i]['long_entry'] else -1
                        entry_price, commission_cost = self._apply_slippage_and_commission(
                            current_bar['Open'],
                            direction,
                            size
                        )
                        
                        # Record entry
                        open_position = {
                            'trade_id': current_trade_id,
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'position_size': size,
                            'entry_signal': 'Long Entry' if direction > 0 else 'Short Entry',
                            'commission': commission_cost
                        }
                        current_trade_id += 1
                        
                        positions.iloc[i] = direction * size
                        cash.iloc[i] = cash.iloc[i-1] - commission_cost
                
                # Update portfolio value for ALL dates, even without trades
                if positions.iloc[i] != 0:
                    unrealized_pnl = positions.iloc[i] * (current_bar['Close'] - open_position['entry_price'])
                    portfolio_value.iloc[i] = cash.iloc[i] + unrealized_pnl
                else:
                    portfolio_value.iloc[i] = cash.iloc[i]

            # Use DrawdownAnalyzer for all drawdown calculations
            drawdown_analyzer = analyze_drawdowns(portfolio_value)
            drawdown_report = drawdown_analyzer.get_drawdown_report()

            # Create comprehensive results DataFrame
            self.results = pd.DataFrame({
                'portfolio_value': portfolio_value,
                'positions': positions,
                'cash': cash,
                'returns': portfolio_value.pct_change().fillna(0),
                'drawdown': drawdown_report['drawdown_series'],
                'historical_peak': drawdown_analyzer.metrics['running_peak'],
                'cumulative_returns': (1 + portfolio_value.pct_change().fillna(0)).cumprod() - 1
            })

            # Calculate performance metrics
            trades_df = pd.DataFrame(trades)
            analyzer = PerformanceAnalyzer(self.results, self.risk_free_rate)
            performance_report = analyzer.generate_report(trades_df)
            
            # Add detailed drawdown analysis to performance report
            performance_report['Drawdown Analysis'] = {
                'summary': drawdown_report['summary_metrics'],
                'recovery_stats': drawdown_report['recovery_statistics'],
                'worst_drawdowns': drawdown_analyzer.calculate_worst_drawdowns(n=5).to_dict('records'),
                'timeframe_analysis': drawdown_analyzer.analyze_drawdown_by_timeframes(),
                'underwater_periods': drawdown_report['underwater_periods'],
                'drawdown_series': drawdown_report['drawdown_series']
            }

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
                'trades': trades_df,
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
