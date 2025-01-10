import pandas as pd
import numpy as np

def run_backtest(prices: pd.DataFrame, 
                weights: dict,
                rebalance_frequency: str = 'M',
                transaction_costs: float = 0.001):
    """
    Run a portfolio backtest with periodic rebalancing
    
    :param prices: DataFrame of asset prices
    :param weights: dict of target weights
    :param rebalance_frequency: pandas frequency string ('M' for monthly)
    :param transaction_costs: percentage cost per trade (one-way)
    :return: DataFrame with backtest results
    """
    # Convert weights to series
    target_weights = pd.Series(weights)
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Initialize backtest DataFrame
    backtest_results = pd.DataFrame(index=returns.index)
    backtest_results['portfolio_value'] = 1.0
    current_weights = target_weights.copy()
    portfolio_value = 1.0
    
    # Group by rebalancing frequency
    for period_start, period_data in returns.groupby(pd.Grouper(freq=rebalance_frequency)):
        if period_data.empty:
            continue
            
        # Calculate current weights after drift
        if len(period_data) > 0:
            period_returns = (1 + period_data).prod() - 1
            drifted_weights = current_weights * (1 + period_returns)
            drifted_weights = drifted_weights / drifted_weights.sum()
            
            # Calculate rebalancing costs (round-trip)
            weight_changes = np.abs(drifted_weights - target_weights).sum()
            costs = weight_changes * transaction_costs  # Round-trip costs
            
            # Apply costs at the beginning of the period
            portfolio_value *= (1 - costs)
            backtest_results.loc[period_data.index[0], 'costs'] = costs
        
        # Calculate returns for the period
        period_portfolio_returns = period_data.mul(current_weights).sum(axis=1)
        backtest_results.loc[period_data.index, 'returns'] = period_portfolio_returns
        
        # Update portfolio value
        for date in period_data.index:
            portfolio_value *= (1 + period_portfolio_returns[date])
            backtest_results.loc[date, 'portfolio_value'] = portfolio_value
        
        # Update weights for next period
        current_weights = target_weights.copy()
    
    # Fill missing values
    backtest_results['costs'] = backtest_results['costs'].fillna(0)
    backtest_results['returns_after_costs'] = backtest_results['returns']
    
    # Adjust returns after costs on rebalancing dates
    rebalancing_dates = backtest_results[backtest_results['costs'] > 0].index
    backtest_results.loc[rebalancing_dates, 'returns_after_costs'] -= backtest_results.loc[rebalancing_dates, 'costs']
    
    return backtest_results

def calculate_backtest_metrics(backtest_results: pd.DataFrame,
                             risk_free_rate: float = 0.02):
    """
    Calculate performance metrics from backtest results
    """
    returns = backtest_results['returns_after_costs']
    
    # Calculate proper annualized return from total return
    total_return = backtest_results['portfolio_value'].iloc[-1] - 1
    n_years = len(returns) / 252  # Convert days to years
    ann_return = (1 + total_return) ** (1 / n_years) - 1  # Geometric mean annual return
    
    # Rest of calculations
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Win rate and other metrics
    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
    
    metrics = {
        'Annual Return': {'Value': ann_return, 'Format': 'percentage'},
        'Annual Volatility': {'Value': ann_vol, 'Format': 'percentage'},
        'Sharpe Ratio': {'Value': sharpe, 'Format': 'decimal'},
        'Max Drawdown': {'Value': max_drawdown, 'Format': 'percentage'},
        'Win Rate': {'Value': win_rate, 'Format': 'percentage'},
        'Avg Win': {'Value': avg_win, 'Format': 'percentage'},
        'Avg Loss': {'Value': avg_loss, 'Format': 'percentage'},
        'Profit Factor': {'Value': profit_factor, 'Format': 'decimal'},
        'Total Return': {'Value': total_return, 'Format': 'percentage'},
        'Total Costs': {'Value': backtest_results['costs'].sum(), 'Format': 'percentage'}
    }
    
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Metric'})
    metrics_df['Formatted Value'] = metrics_df.apply(
        lambda x: f"{x['Value']:.2%}" if x['Format'] == 'percentage' and pd.notnull(x['Value']) 
        else f"{x['Value']:.2f}" if x['Format'] == 'decimal' and pd.notnull(x['Value'])
        else "N/A", axis=1
    )
    
    return metrics_df
