import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional
from ..metrics.drawdown import analyze_drawdowns

def run_backtest(
    data: pd.DataFrame,
    weights: Dict[str, float],
    rebalance_frequency: str = "ME",
    transaction_costs: float = 0.001,
    benchmark: Optional[str] = None
) -> pd.DataFrame:
    """
    Run portfolio backtest with rebalancing and transaction costs.
    
    :param data: DataFrame of asset prices
    :param weights: Dictionary of weights {asset: weight}
    :param rebalance_frequency: Rebalancing frequency ('ME', 'QE', '6M', 'Y')
    :param transaction_costs: Transaction costs as decimal
    :param benchmark: Optional benchmark ticker (e.g. '^GSPC' for S&P 500)
    :return: DataFrame with backtest results
    """
    # Convert weights to series if dict
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Initialize portfolio
    portfolio_value = 100  # Start with $100
    current_weights = weights.copy()
    portfolio_values = []
    dates = []
    portfolio_returns = []
    rebalance_dates = pd.date_range(start=returns.index[0], end=returns.index[-1], freq=rebalance_frequency)
    
    # Get benchmark data if specified
    benchmark_values = None
    if benchmark:
        try:
            benchmark_data = yf.download(benchmark, start=returns.index[0], end=returns.index[-1], auto_adjust=True)
            benchmark_values = benchmark_data['Close'] / benchmark_data['Close'].iloc[0] * 100
        except:
            print(f"Warning: Could not fetch benchmark data for {benchmark}")
    
    # Run backtest
    for date in returns.index:
        # Calculate daily returns
        daily_return = (returns.loc[date] * current_weights).sum()
        
        # Update portfolio value
        portfolio_value *= (1 + daily_return)
        
        # Store results
        portfolio_values.append(portfolio_value)
        dates.append(date)
        portfolio_returns.append(daily_return)
        
        # Rebalance if needed
        if date in rebalance_dates:
            # Calculate current weights
            asset_values = portfolio_value * current_weights
            current_total = asset_values.sum()
            current_weights = asset_values / current_total
            
            # Calculate rebalancing costs
            weight_diffs = abs(current_weights - weights)
            turnover = weight_diffs.sum() / 2
            cost = turnover * transaction_costs
            
            # Apply transaction costs
            portfolio_value *= (1 - cost)
            
            # Reset to target weights
            current_weights = weights.copy()

    # Create results DataFrame
    results = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'returns': portfolio_returns
    }, index=returns.index)
    
    # Add benchmark if available
    if benchmark_values is not None:
        benchmark_values = benchmark_values.reindex(returns.index, method='ffill')
        results['benchmark_value'] = benchmark_values
    
    return results

def calculate_backtest_metrics(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics from backtest results.
    
    :param backtest_results: DataFrame with backtest results
    :return: DataFrame with metrics
    """
    portfolio_returns = backtest_results['returns']
    
    # Calculate metrics
    total_return = (backtest_results['portfolio_value'].iloc[-1] / backtest_results['portfolio_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    
    # Drawdown calculations
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Usar DrawdownAnalyzer para análisis de drawdown
    drawdown_analyzer = analyze_drawdowns(backtest_results['portfolio_value'])
    drawdown_metrics = drawdown_analyzer.get_complete_drawdown_metrics()
    
    # Calculate benchmark-relative metrics if available
    benchmark_metrics = {}
    if 'benchmark_value' in backtest_results.columns:
        benchmark_returns = backtest_results['benchmark_value'].pct_change().dropna()
        # Align portfolio and benchmark returns
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio_returns = aligned.iloc[:, 0]
        benchmark_returns = aligned.iloc[:, 1]
        
        # Calculate beta
        covar = np.cov(portfolio_returns, benchmark_returns)[0,1]
        var = np.var(benchmark_returns)
        beta = covar / var if var != 0 else np.nan
        # Calculate alpha
        alpha = annual_return - beta * (benchmark_returns.mean() * 252)
        # Information ratio
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        info_ratio = (annual_return - benchmark_returns.mean() * 252) / tracking_error if tracking_error != 0 else np.nan
        
        benchmark_metrics.update({
            'Beta': {'Value': beta, 'Format': 'decimal'},
            'Alpha': {'Value': alpha, 'Format': 'percentage'},
            'Information Ratio': {'Value': info_ratio, 'Format': 'decimal'},
            'Tracking Error': {'Value': tracking_error, 'Format': 'percentage'}
        })
    
    # Create metrics dictionary
    metrics = {
        'Total Return': {'Value': total_return, 'Format': 'percentage'},
        'Annualized Return': {'Value': annual_return, 'Format': 'percentage'},
        'Annual Volatility': {'Value': annual_vol, 'Format': 'percentage'},
        'Sharpe Ratio': {'Value': sharpe, 'Format': 'decimal'},
        'Maximum Drawdown': {'Value': max_drawdown, 'Format': 'percentage'},
        'Drawdown Analysis': drawdown_metrics,
    }
    
    # Add benchmark metrics if available
    metrics.update(benchmark_metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).T.reset_index()
    metrics_df.columns = ['Metric', 'Value', 'Format']
    
    # Add formatted values
    metrics_df['Formatted Value'] = metrics_df.apply(
        lambda x: f"{x['Value']:.2%}" if x['Format'] == 'percentage' and pd.notnull(x['Value'])
        else f"{x['Value']:.2f}" if x['Format'] == 'decimal' and pd.notnull(x['Value'])
        else "N/A", axis=1
    )
    
    return metrics_df
