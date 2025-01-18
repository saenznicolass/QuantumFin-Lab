# modules/risk/analysis/tail_analysis.py

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go

def extreme_value_analysis(returns, weights, threshold_percentile=5):
    """
    Perform an extreme value analysis for portfolio returns below a certain percentile.

    :param returns: DataFrame of asset returns
    :param weights: dict or array of weights
    :param threshold_percentile: int, e.g. 5 for the bottom 5% tail
    :return: dict with tail stats
    """
    if isinstance(weights, dict):
        w_array = []
        for col in returns.columns:
            w_array.append(weights.get(col, 0.0))
        weights = np.array(w_array)

    portfolio_returns = returns.values.dot(weights)
    threshold = np.percentile(portfolio_returns, threshold_percentile)
    tail_returns = portfolio_returns[portfolio_returns <= threshold]

    tail_mean = tail_returns.mean() if len(tail_returns) else np.nan
    tail_vol = tail_returns.std() if len(tail_returns) else np.nan
    # 5% VaR in the tail, or we can default to a fixed 5% if desired
    tail_var = -np.percentile(tail_returns, threshold_percentile) if len(tail_returns) else np.nan
    tail_cvar = -tail_returns[tail_returns <= -tail_var].mean() if len(tail_returns) and tail_var and not np.isnan(tail_var) else np.nan

    return {
        'tail_var': tail_var,
        'tail_cvar': tail_cvar,
        'tail_mean': tail_mean,
        'tail_vol': tail_vol,
        'tail_data': tail_returns
    }

def analyze_tail_risk(returns, weights, threshold=0.05):
    """
    Analyze tail risk using a threshold approach (like EV theory).
    If threshold=0.05, we look at the bottom 5% returns.

    :param returns: DataFrame of asset returns
    :param weights: array or dict of portfolio weights
    :param threshold: float, e.g. 0.05
    :return: dict with tail risk metrics
    """
    if isinstance(weights, dict):
        w_array = []
        for col in returns.columns:
            w_array.append(weights.get(col, 0.0))
        weights = np.array(w_array)

    portfolio_returns = returns.values.dot(weights)
    cutoff = np.percentile(portfolio_returns, threshold * 100)
    tail_returns = portfolio_returns[portfolio_returns < cutoff]

    tail_mean = tail_returns.mean() if len(tail_returns) else np.nan
    tail_vol = tail_returns.std() if len(tail_returns) else np.nan
    tail_var = -np.percentile(tail_returns, 5) if len(tail_returns) else np.nan
    tail_cvar = -tail_returns[tail_returns <= -tail_var].mean() if len(tail_returns) and tail_var and not np.isnan(tail_var) else np.nan

    return {
        'tail_var': tail_var,
        'tail_cvar': tail_cvar,
        'tail_mean': tail_mean,
        'tail_vol': tail_vol,
        'tail_data': tail_returns
    }

def create_qq_plot(returns, weights):
    """
    Create a Q-Q plot for portfolio returns vs a Normal distribution.

    :param returns: DataFrame of asset returns
    :param weights: dict or array of portfolio weights
    :return: plotly Figure
    """
    if isinstance(weights, dict):
        w_array = []
        for col in returns.columns:
            w_array.append(weights.get(col, 0.0))
        weights = np.array(w_array)

    portfolio_returns = returns.values.dot(weights)
    sorted_returns = np.sort(portfolio_returns)
    p = np.linspace(0.01, 0.99, len(sorted_returns))
    theoretical_q = stats.norm.ppf(p, loc=np.mean(sorted_returns), scale=np.std(sorted_returns))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical_q,
        y=sorted_returns,
        mode='markers',
        name='Q-Q Plot'
    ))
    min_val = min(theoretical_q.min(), sorted_returns.min())
    max_val = max(theoretical_q.max(), sorted_returns.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Normal Dist',
        line=dict(dash='dash')
    ))
    fig.update_layout(
        title='Q-Q Plot of Portfolio Returns',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles'
    )
    return fig
