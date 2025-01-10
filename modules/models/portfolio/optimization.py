# modules/models/portfolio/optimization.py

import numpy as np
import pandas as pd
import streamlit as st

from pypfopt import EfficientFrontier, risk_models, expected_returns
import plotly.graph_objects as go

def run_portfolio_optimization(data: pd.DataFrame, 
                               allow_short=True,
                               optimization_objective="Max Sharpe Ratio",
                               required_return=None,
                               max_weight=1.0,
                               min_weight=0.0,
                               risk_free_rate=0.0):
    """
    Performs portfolio optimization (unconstrained + constrained) using
    the PyPortfolioOpt library and stores results in session state.

    :param data: DataFrame of historical prices (columns = tickers, rows = date)
    :param allow_short: bool indicating if short selling is allowed
    :param optimization_objective: str among ["Max Sharpe Ratio", "Min Volatility", "Target Return"]
    :param required_return: float, the target return in decimal (e.g., 0.05 for 5%)
    :param max_weight: float, maximum weight allowed per asset
    :param min_weight: float, minimum weight allowed per asset
    :param risk_free_rate: float, the risk-free rate
    :return: dict with unconstrained weights, constrained weights, 
             and their performance metrics
    """
    # Basic checks
    if data.empty or len(data.columns) < 2:
        raise ValueError("Data is empty or only one asset selected. Need >= 2 assets.")

    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(data)
    S  = risk_models.sample_cov(data)
    # Ensure symmetric matrix
    S  = (S + S.T) / 2
    valid_assets = mu.index

    # Weight bounds
    weight_bounds_unconstrained = (None, None) if allow_short else (0, 1)
    weight_bounds_constrained = (min_weight, max_weight)

    # ============ 1) UNCONSTRAINED OPTIMIZATION ==============
    ef_unconstrained = EfficientFrontier(mu, S, weight_bounds=weight_bounds_unconstrained)
    if optimization_objective == "Max Sharpe Ratio":
        ef_unconstrained.max_sharpe(risk_free_rate=risk_free_rate)
    elif optimization_objective == "Min Volatility":
        ef_unconstrained.min_volatility()
    elif optimization_objective == "Target Return" and required_return is not None:
        ef_unconstrained.efficient_return(target_return=required_return)
    else:
        ef_unconstrained.max_sharpe(risk_free_rate=risk_free_rate)

    weights_unconstrained = ef_unconstrained.clean_weights()
    performance_unconstrained = ef_unconstrained.portfolio_performance(
        verbose=False, risk_free_rate=risk_free_rate
    )

    # ============ 2) CONSTRAINED OPTIMIZATION ==============
    ef_constrained = EfficientFrontier(mu, S, weight_bounds=weight_bounds_constrained)
    if optimization_objective == "Max Sharpe Ratio":
        ef_constrained.max_sharpe(risk_free_rate=risk_free_rate)
    elif optimization_objective == "Min Volatility":
        ef_constrained.min_volatility()
    elif optimization_objective == "Target Return" and required_return is not None:
        ef_constrained.efficient_return(target_return=required_return)
    else:
        ef_constrained.max_sharpe(risk_free_rate=risk_free_rate)

    weights_constrained = ef_constrained.clean_weights()
    performance_constrained = ef_constrained.portfolio_performance(
        verbose=False, risk_free_rate=risk_free_rate
    )

    # Create DataFrames for performance metrics
    metrics_data = {
        'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
        'Unconstrained': list(performance_unconstrained),
        'Constrained': list(performance_constrained),
        'Format': ['percentage', 'percentage', 'decimal']
    }
    
    performance_df = pd.DataFrame(metrics_data)
    
    for col in ['Unconstrained', 'Constrained']:
        performance_df[f'Formatted {col}'] = performance_df.apply(
            lambda x: f"{x[col]:.2%}" if x['Format'] == 'percentage' and pd.notnull(x[col]) 
            else f"{x[col]:.2f}" if x['Format'] == 'decimal' and pd.notnull(x[col])
            else "N/A", axis=1
        )

    # Create DataFrames for weights
    weights_data = {
        'Asset': list(valid_assets),
        'Unconstrained': [weights_unconstrained.get(asset, 0.0) for asset in valid_assets],
        'Constrained': [weights_constrained.get(asset, 0.0) for asset in valid_assets],
        'Format': ['percentage'] * len(valid_assets)
    }
    
    weights_df = pd.DataFrame(weights_data)
    
    for col in ['Unconstrained', 'Constrained']:
        weights_df[f'Formatted {col}'] = weights_df.apply(
            lambda x: f"{x[col]:.2%}" if pd.notnull(x[col]) else "N/A", axis=1
        )

    # Store results in session state
    st.session_state['weights_constrained'] = weights_constrained
    st.session_state['weights_unconstrained'] = weights_unconstrained
    st.session_state['data'] = data
    st.session_state['mu'] = mu
    st.session_state['S'] = S
    st.session_state['performance_constrained'] = performance_constrained
    st.session_state['performance_unconstrained'] = performance_unconstrained
    st.session_state['valid_assets'] = valid_assets

    return {
        'mu': mu,
        'S': S,
        'valid_assets': valid_assets,
        'weights_df': weights_df,
        'performance_df': performance_df,
        'weights_unconstrained': weights_unconstrained,
        'performance_unconstrained': performance_unconstrained,
        'weights_constrained': weights_constrained,
        'performance_constrained': performance_constrained,
    }


def compute_efficient_frontier_points(mu, S, bounds, risk_free, n_points=50):
    """
    Helper to compute points on the efficient frontier by stepping
    through the feasible return range and calling 'efficient_return'.
    
    :param risk_free: The risk-free rate to use (from user input)
    """
    returns_range = np.linspace(mu.min(), mu.max(), n_points)
    frontier_points = []
    from pypfopt.efficient_frontier import EfficientFrontier

    for r_target in returns_range:
        ef_temp = EfficientFrontier(mu, S, weight_bounds=bounds)
        try:
            ef_temp.efficient_return(target_return=r_target)
            ret, vol, _ = ef_temp.portfolio_performance(verbose=False, risk_free_rate=risk_free)
            frontier_points.append((vol, ret))
        except:
            # Some target returns not feasible
            pass
    return frontier_points


@st.cache_data
def compute_cumulative_returns(historical_data: pd.DataFrame, weights: dict):
    """
    Given a DataFrame of prices and a dict of weights,
    compute the daily portfolio returns and then the cumulative returns.
    """
    w_series = pd.Series(weights)
    daily_returns = historical_data.pct_change().dropna()
    portfolio_returns = daily_returns.mul(w_series, axis=1).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns

def analyze_portfolio_metrics(data: pd.DataFrame, weights: dict):
    """
    Calculate additional portfolio metrics for analysis
    """
    if not isinstance(weights, dict):
        weights = dict(zip(data.columns, weights))
        
    returns = data.pct_change().dropna()
    portfolio_returns = returns.mul(pd.Series(weights)).sum(axis=1)
    
    # Calculate metrics
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    
    # Calculate drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown
    }

def plot_portfolio_weights(weights: dict, title: str = "Portfolio Weights"):
    """
    Create a pie chart visualization of portfolio weights
    """
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.3
    )])
    
    fig.update_layout(
        title=title,
        showlegend=True
    )
    
    return fig
