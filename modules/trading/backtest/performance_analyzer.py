from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

class PerformanceAnalyzer:
    """Analyze and report trading strategy performance"""
    
    def __init__(self, results: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer
        
        Parameters:
        -----------
        results : pd.DataFrame
            Backtest results containing returns and positions
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio calculation
        """
        self.results = results
        self.rf_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        self.metrics = {}
        
    def calculate_returns(self) -> Dict:
        """Calculate return-based performance metrics"""
        returns = self.results['returns']
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': (1 + returns).prod() ** (252/len(returns)) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean(),
            'avg_winner': returns[returns > 0].mean(),
            'avg_loser': returns[returns < 0].mean(),
        }
        
        # Risk-adjusted metrics
        excess_returns = returns - self.daily_rf
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(excess_returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(excess_returns)
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based performance metrics"""
        returns = self.results['returns']
        
        metrics = {
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'worst_day': returns.min(),
            'best_day': returns.max(),
        }
        
        # Calculate rolling metrics
        roll_vol = returns.rolling(window=21).std() * np.sqrt(252)
        metrics['avg_volatility'] = roll_vol.mean()
        metrics['max_volatility'] = roll_vol.max()
        
        self.metrics.update(metrics)
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        var = self.calculate_var(returns, confidence)
        return -returns[returns <= -var].mean()
    
    def calculate_sharpe_ratio(self, excess_returns: pd.Series) -> float:
        """Calculate Sharpe Ratio"""
        if len(excess_returns) < 2:
            return 0.0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def calculate_sortino_ratio(self, excess_returns: pd.Series) -> float:
        """Calculate Sortino Ratio"""
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        downside_std = np.sqrt(252) * negative_returns.std()
        return excess_returns.mean() * np.sqrt(252) / downside_std
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        # Calculate all metrics
        self.calculate_returns()
        self.calculate_risk_metrics()
        
        # Format metrics for reporting
        report = {}
        for category, metrics in {
            'Returns': ['total_return', 'annual_return', 'volatility', 'max_drawdown'],
            'Risk Metrics': ['var_95', 'cvar_95', 'skewness', 'kurtosis'],
            'Risk-Adjusted': ['sharpe_ratio', 'sortino_ratio'],
            'Trading Stats': ['win_rate', 'avg_winner', 'avg_loser']
        }.items():
            report[category] = {
                metric: self.metrics.get(metric) 
                for metric in metrics 
                if metric in self.metrics
            }
        
        return report
