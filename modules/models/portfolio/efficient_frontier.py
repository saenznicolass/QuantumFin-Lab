# modules/models/portfolio/efficient_frontier.py

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from typing import Tuple, List
import plotly.graph_objects as go

def generate_efficient_frontier_points(
    mu,
    cov_matrix,
    weight_bounds,
    allow_short: bool = True,
    points: int = 50,
    risk_free_rate: float = 0.05
) -> pd.DataFrame:
    """
    Generate a set of (volatility, return) points for the efficient frontier.
    We do this by sweeping over target returns from min to max.

    :param mu: Expected returns (Series)
    :param cov_matrix: Covariance matrix (DataFrame)
    :param weight_bounds: (min_weight, max_weight) or (None, None)
    :param allow_short: Whether short selling is permitted
    :param points: Number of discrete points to sample along return range
    :param risk_free_rate: Used for possible Sharpe ratio calculations
    :return: DataFrame with columns ['Return', 'Volatility'] representing frontier
    """
    if mu.empty or cov_matrix.empty:
        return pd.DataFrame(columns=['Return', 'Volatility'])

    # Convert bounds if short is allowed
    if allow_short and weight_bounds == (0.0, 1.0):
        weight_bounds = (None, None)

    min_ret = mu.min()
    max_ret = mu.max()
    ret_range = np.linspace(min_ret, max_ret, points)

    results = []
    for target in ret_range:
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=weight_bounds)
        try:
            ef.efficient_return(target_return=target)
            # no need to store weights here
            ret, vol, _ = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results.append({"Return": ret, "Volatility": vol})
        except:
            # Some target returns might be infeasible
            pass

    return pd.DataFrame(results)

def calculate_efficient_frontier(returns: pd.DataFrame, 
                               min_vol_target: float = None,
                               max_vol_target: float = None,
                               num_portfolios: int = 1000) -> pd.DataFrame:
    """
    Calculate the efficient frontier points.
    
    :param returns: DataFrame of asset returns
    :param min_vol_target: minimum volatility target
    :param max_vol_target: maximum volatility target
    :param num_portfolios: number of portfolios to simulate
    :return: DataFrame with portfolio weights and metrics
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)
    
    # Generate random portfolios
    np.random.seed(42)
    weights_list = []
    returns_list = []
    volatility_list = []
    sharpe_list = []
    
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        
        weights_list.append(weights)
        returns_list.append(portfolio_return)
        volatility_list.append(portfolio_std)
        sharpe_list.append(sharpe_ratio)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Return': returns_list,
        'Volatility': volatility_list,
        'Sharpe': sharpe_list
    })
    
    # Add individual asset weights
    for i, asset in enumerate(returns.columns):
        results[f'Weight_{asset}'] = [w[i] for w in weights_list]
    
    return results

def plot_efficient_frontier_3d(frontier_df: pd.DataFrame) -> go.Figure:
    """
    Create a 3D visualization of the efficient frontier.
    
    :param frontier_df: DataFrame with portfolio metrics
    :return: Plotly figure object
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=frontier_df['Volatility'],
        y=frontier_df['Return'],
        z=frontier_df['Sharpe'],
        mode='markers',
        marker=dict(
            size=6,
            color=frontier_df['Sharpe'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title='Efficient Frontier 3D Visualization',
        scene=dict(
            xaxis_title='Volatility',
            yaxis_title='Return',
            zaxis_title='Sharpe Ratio'
        ),
        width=800,
        height=800
    )
    
    return fig

def get_optimal_portfolio(frontier_df: pd.DataFrame, 
                         optimization_goal: str = 'sharpe') -> pd.Series:
    """
    Get the optimal portfolio based on the specified goal.
    
    :param frontier_df: DataFrame with portfolio metrics
    :param optimization_goal: 'sharpe', 'min_vol', or 'max_return'
    :return: Series with optimal portfolio weights and metrics
    """
    if optimization_goal == 'sharpe':
        idx = frontier_df['Sharpe'].idxmax()
    elif optimization_goal == 'min_vol':
        idx = frontier_df['Volatility'].idxmin()
    elif optimization_goal == 'max_return':
        idx = frontier_df['Return'].idxmax()
    else:
        raise ValueError("Invalid optimization goal")
        
    return frontier_df.loc[idx]
