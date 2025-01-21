from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ...risk.metrics.drawdown import analyze_drawdowns, calculate_calmar_ratio

class PerformanceAnalyzer:
    """Analyze and report trading strategy performance"""
    
    def __init__(self, results: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer
        
        Parameters:
        -----------
        results : pd.DataFrame
            Backtest results containing portfolio values, positions and trades
        risk_free_rate : float
            Annual risk-free rate for risk-adjusted metrics
        """
        self.results = results
        self.rf_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        self.metrics = {}
        print(results.head())

    def calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate trading-specific performance metrics"""
        if trades_df.empty:
            return {}
            
        # Group trades by trade_id for accurate metrics
        if 'trade_id' in trades_df.columns:
            complete_trades = trades_df.dropna(subset=['trade_id']).copy()
            metrics = {}
            
            if len(complete_trades) > 0:
                trade_summary = complete_trades.groupby('trade_id').agg({
                    'entry_date': 'first',
                    'exit_date': 'last',
                    'pnl': 'sum',
                    'commission': 'sum'
                }).reset_index()
                
                trade_summary['holding_period'] = (
                    pd.to_datetime(trade_summary['exit_date']) - 
                    pd.to_datetime(trade_summary['entry_date'])
                ).dt.days
                
                winning_trades = trade_summary[trade_summary['pnl'] > 0]
                losing_trades = trade_summary[trade_summary['pnl'] < 0]
                
                metrics.update({
                    'total_trades': len(trade_summary),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(trade_summary),
                    'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) 
                                  if len(losing_trades) > 0 else float('inf'),
                    'avg_winner': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
                    'avg_loser': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
                    'avg_holding_period': trade_summary['holding_period'].mean(),
                    'total_commission': trade_summary['commission'].sum(),
                    'net_profit': trade_summary['pnl'].sum(),
                    'avg_profit_per_trade': trade_summary['pnl'].mean()
                })

        # Calculate equity curve metrics using COMPLETE date range
        equity_metrics = self.calculate_equity_metrics(self.results['portfolio_value'])
        metrics.update(equity_metrics)
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        gross_profits = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    def calculate_max_consecutive(self, series: pd.Series, value: bool) -> int:
        """Calculate maximum consecutive occurrences of a value"""
        # Convert to strings for easier grouping
        groups = (series != series.shift()).cumsum()
        consecutive = series.groupby(groups).transform('size') * series.astype(int)
        return consecutive[series == value].max() if not series.empty else 0
        
    def calculate_equity_metrics(self, portfolio_values: pd.Series) -> Dict:
        """Calculate equity curve related metrics"""
        if portfolio_values.empty:
            return {}
            
        returns = portfolio_values.pct_change().fillna(0)
        
        # Calculate annualized metrics
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - self.rf_rate) / ann_vol if ann_vol > 0 else 0
        
        # Calculate downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (ann_return - self.rf_rate) / downside_vol if downside_vol > 0 else 0
        
        # Get drawdown metrics from centralized analyzer
        drawdown_analyzer = analyze_drawdowns(portfolio_values)
        drawdown_metrics = drawdown_analyzer.get_complete_drawdown_metrics()
        
        return {
            'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'max_drawdown_duration': drawdown_metrics['max_drawdown_duration'],
            'recovery_factor': drawdown_metrics['recovery_factor'],
            'downside_volatility': downside_vol,
            'drawdown_metrics': drawdown_metrics
        }

    def generate_report(self, trades_df: pd.DataFrame) -> Dict:
        """Generate comprehensive performance report"""
        self.calculate_trading_metrics(trades_df)
        
        report = {
            'Returns': {
                'total_return': self.metrics.get('total_return', 0),
                'annualized_return': self.metrics.get('annualized_return', 0),
                'max_drawdown': self.metrics.get('max_drawdown', 0),
                'max_drawdown_duration': self.metrics.get('max_drawdown_duration', 0)
            },
            'Risk Metrics': {
                'annualized_volatility': self.metrics.get('annualized_volatility', 0),
                'downside_volatility': self.metrics.get('downside_volatility', 0),
                'max_drawdown': self.metrics.get('max_drawdown', 0)
            },
            'Risk-Adjusted': {
                'sharpe_ratio': self.metrics.get('sharpe_ratio', 0),
                'sortino_ratio': self.metrics.get('sortino_ratio', 0),
                'recovery_factor': self.metrics.get('recovery_factor', 0),
            },
            'Trading Stats': {
                'total_trades': self.metrics.get('total_trades', 0),
                'winning_trades': self.metrics.get('winning_trades', 0),
                'losing_trades': self.metrics.get('losing_trades', 0),
                'win_rate': self.metrics.get('win_rate', 0),
                'avg_win': self.metrics.get('avg_winner', 0),
                'avg_loss': abs(self.metrics.get('avg_loser', 0)),
                'net_profit': self.metrics.get('net_profit', 0),  # Añadido
                'total_commission': self.metrics.get('total_commission', 0),  # Añadido
                'profit_factor': self.metrics.get('profit_factor', 0),
                'avg_profit_per_trade': self.metrics.get('avg_profit_per_trade', 0),
                'avg_holding_period': self.metrics.get('avg_holding_period', 0),
                'expectancy': self.calculate_expectancy()
            },
            'Drawdown Analysis': {
                'underwater_periods': self.metrics.get('underwater_periods', 0),
                'avg_recovery_time': self.metrics.get('avg_recovery_time', 0),
                'max_drawdown': self.metrics.get('max_drawdown', 0),
                'max_drawdown_duration': self.metrics.get('max_drawdown_duration', 0)
            }
        }

        # Remove old drawdown calculations and use centralized analyzer
        drawdown_analyzer = analyze_drawdowns(self.results['portfolio_value'])
        drawdown_report = drawdown_analyzer.get_drawdown_report()
        
        report['Risk Metrics'].update({
            'calmar_ratio': calculate_calmar_ratio(self.results['portfolio_value']),
            'max_drawdown': drawdown_report['max_drawdown'],
            'avg_recovery_time': drawdown_report['recovery_statistics']['avg_recovery_time']
        })
        
        report['Drawdown Analysis'] = drawdown_report

        return report
    
    def calculate_expectancy(self) -> float:
        """
        Calculate trading expectancy (average reward-to-risk ratio)
        using pre-calculated metrics
        
        Expectancy = (Win Rate * R-Multiple) - Loss Rate
        where R-Multiple = Average Win / Average Loss
        
        Returns:
        --------
        float
            Expectancy value representing expected return per dollar risked
        """
        # Use pre-calculated metrics
        win_rate = self.metrics.get('win_rate', 0)
        avg_win = self.metrics.get('avg_winner', 0)
        avg_loss = abs(self.metrics.get('avg_loser', 0))
        
        # Calculate loss rate from win rate
        loss_rate = 1 - win_rate
        # Avoid division by zero
        if avg_loss == 0:
            return 0.0
            
        # Calculate R-multiple
        r_multiple = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate * r_multiple) - loss_rate
        return expectancy
