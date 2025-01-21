import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DrawdownAnalyzer:
    """Centralized class for drawdown analysis and visualization"""
    
    def __init__(self, equity_curve: pd.Series):
        self.equity_curve = equity_curve
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive drawdown metrics"""
        running_peak = self.equity_curve.expanding().max()
        drawdown_series = (self.equity_curve - running_peak) / running_peak
        
        # Calculate drawdown periods
        is_drawdown = drawdown_series < 0
        drawdown_periods = self._calculate_drawdown_periods(drawdown_series)
        underwater_periods = self._calculate_underwater_periods()
        
        return {
            'drawdown_series': drawdown_series,
            'max_drawdown': drawdown_series.min(),
            'current_drawdown': drawdown_series.iloc[-1],
            'drawdown_periods': drawdown_periods,
            'underwater_periods': underwater_periods,
            'running_peak': running_peak
        }
    
    def _calculate_drawdown_periods(self, drawdown_series: pd.Series) -> list:
        """Calculate drawdown period statistics"""
        is_drawdown = drawdown_series < 0
        periods = []
        current_duration = 0
        start_idx = None
        
        for i, (date, in_dd) in enumerate(is_drawdown.items()):
            if in_dd:
                if start_idx is None:
                    start_idx = i
                current_duration += 1
            elif start_idx is not None:
                periods.append({
                    'start_date': drawdown_series.index[start_idx],
                    'end_date': date,
                    'duration': current_duration,
                    'depth': drawdown_series[start_idx:i].min()
                })
                start_idx = None
                current_duration = 0
        
        return periods
    
    def _calculate_underwater_periods(self) -> Dict:
        """Calculate underwater period statistics"""
        peak = self.equity_curve.iloc[0]
        underwater_start = None
        periods = []
        
        for date, value in self.equity_curve.items():
            if value > peak:
                peak = value
                if underwater_start:
                    periods.append(self._create_underwater_period(underwater_start, date, peak))
                    underwater_start = None
            elif value < peak and not underwater_start:
                underwater_start = date
        
        if underwater_start:
            periods.append(
                self._create_underwater_period(
                    underwater_start, 
                    self.equity_curve.index[-1],
                    peak
                )
            )
        
        return self._summarize_underwater_periods(periods)
    
    def _create_underwater_period(self, start: pd.Timestamp, end: pd.Timestamp, peak: float) -> Dict:
        """Create an underwater period record"""
        return {
            'start': start,
            'end': end,
            'duration': (end - start).days,
            'depth': min(self.equity_curve[start:end]) / peak - 1
        }
    
    def _summarize_underwater_periods(self, periods: list) -> Dict:
        """Summarize underwater period statistics"""
        if not periods:
            return {
                'total_periods': 0,
                'avg_duration': 0,
                'avg_depth': 0,
                'max_duration': 0,
                'max_depth': 0,
                'periods': []
            }
        
        return {
            'total_periods': len(periods),
            'avg_duration': np.mean([p['duration'] for p in periods]),
            'avg_depth': np.mean([p['depth'] for p in periods]),
            'max_duration': max([p['duration'] for p in periods]),
            'max_depth': min([p['depth'] for p in periods]),
            'periods': periods
        }
    
    def plot_drawdown_analysis(self) -> go.Figure:
        """Create unified drawdown visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value & Drawdown', 'Underwater Analysis'),
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        # Portfolio value with underwater periods highlighted
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve,
                name="Portfolio Value",
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add underwater period markers
        for period in self.metrics['underwater_periods']['periods']:
            fig.add_vrect(
                x0=period['start'],
                x1=period['end'],
                fillcolor="rgba(255,0,0,0.1)",
                layer="below",
                line_width=0,
                row=1, col=1
            )
        
        # Drawdown series
        fig.add_trace(
            go.Scatter(
                x=self.metrics['drawdown_series'].index,
                y=self.metrics['drawdown_series'],
                name="Drawdown",
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title="Drawdown Analysis",
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def get_summary_metrics(self) -> Dict:
        """Get summary metrics for display"""
        return {
            'max_drawdown': self.metrics['max_drawdown'],
            'current_drawdown': self.metrics['current_drawdown'],
            'total_underwater_periods': self.metrics['underwater_periods']['total_periods'],
            'avg_underwater_duration': self.metrics['underwater_periods']['avg_duration'],
            'max_underwater_duration': self.metrics['underwater_periods']['max_duration']
        }

    def calculate_recovery_statistics(self) -> Dict:
        """Calculate detailed recovery statistics"""
        is_drawdown = self.metrics['drawdown_series'] < 0
        recovery_periods = []
        current_period = 0
        
        for is_down in is_drawdown:
            if is_down:
                current_period += 1
            elif current_period > 0:
                recovery_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            recovery_periods.append(current_period)
            
        return {
            'recovery_periods': recovery_periods,
            'avg_recovery_time': np.mean(recovery_periods) if recovery_periods else 0,
            'max_recovery_time': max(recovery_periods) if recovery_periods else 0,
            'total_recoveries': len(recovery_periods)
        }

    def get_drawdown_report(self) -> Dict:
        """Generate comprehensive drawdown report"""
        recovery_stats = self.calculate_recovery_statistics()
        
        return {
            'max_drawdown': self.metrics['max_drawdown'],
            'current_drawdown': self.metrics['current_drawdown'],
            'underwater_periods': self.metrics['underwater_periods'],
            'drawdown_series': self.metrics['drawdown_series'],
            'recovery_statistics': recovery_stats,
            'summary_metrics': self.get_summary_metrics()
        }

    def get_underwater_chart_data(self) -> Tuple[pd.Series, pd.Series]:
        """Get data for underwater chart visualization"""
        return (
            self.equity_curve,
            self.metrics['drawdown_series']
        )

    def analyze_drawdown_by_timeframes(self, timeframes: List[str] = ['ME', 'QE', 'YE']) -> Dict:
        """Analyze drawdown statistics across different timeframes"""
        results = {}
        for timeframe in timeframes:
            resampled = self.equity_curve.resample(timeframe).last()
            timeframe_dd = DrawdownAnalyzer(resampled)
            results[timeframe] = timeframe_dd.get_summary_metrics()
        return results

    def calculate_worst_drawdowns(self, n: int = 5) -> pd.DataFrame:
        """Get the N worst drawdown periods"""
        drawdown_periods = pd.DataFrame(self.metrics['drawdown_periods'])
        if drawdown_periods.empty:
            return pd.DataFrame()
            
        return drawdown_periods.sort_values('depth').head(n)

    def calculate_max_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days"""
        drawdown = self.metrics['drawdown_series']
        is_drawdown = drawdown < 0
        if not is_drawdown.any():
            return 0
            
        # Calculate durations
        drawdown_periods = []
        start = None
        for date, in_drawdown in is_drawdown.items():
            if in_drawdown and start is None:
                start = date
            elif not in_drawdown and start is not None:
                drawdown_periods.append((date - start).days)
                start = None
                
        return max(drawdown_periods, default=0)

    def calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown)"""
        net_profit = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        max_dd = abs(self.metrics['max_drawdown'])
        return net_profit / max_dd if max_dd != 0 else float('inf')

    def get_complete_drawdown_metrics(self) -> Dict:
        """Get comprehensive drawdown metrics including duration and recovery"""
        report = self.get_drawdown_report()
        report.update({
            'max_drawdown_duration': self.calculate_max_drawdown_duration(),
            'recovery_factor': self.calculate_recovery_factor(),
            'drawdown_metrics': self.get_summary_metrics(),
            'timeframe_analysis': self.analyze_drawdown_by_timeframes()
        })
        return report

# Mantener estas funciones de conveniencia para compatibilidad
def analyze_drawdowns(equity_curve: pd.Series) -> DrawdownAnalyzer:
    """Convenience function to create DrawdownAnalyzer instance"""
    return DrawdownAnalyzer(equity_curve)

def get_rolling_drawdown(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling maximum drawdown"""
    rolling_max = returns.rolling(window=window).max()
    rolling_drawdown = (returns - rolling_max) / rolling_max
    return rolling_drawdown

def calculate_calmar_ratio(portfolio_value: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar Ratio (annualized return / max drawdown)"""
    
    returns = portfolio_value.pct_change().dropna()
    analyzer = analyze_drawdowns(portfolio_value)
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    max_dd = abs(analyzer.metrics['max_drawdown'])
    return ann_return / max_dd if max_dd != 0 else float('inf')
