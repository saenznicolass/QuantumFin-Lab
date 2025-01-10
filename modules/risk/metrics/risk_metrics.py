# modules/risk/metrics/risk_metrics.py

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from modules.config.constants import TRADING_DAYS_PER_YEAR

def calculate_var(payoffs, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) given a distribution of payoffs.
    
    :param payoffs: 1D array or list of simulated payoffs or returns
    :param confidence_level: float (0 < confidence_level < 1), e.g. 0.95
    :return: VaR as a float
    """
    sorted_payoffs = np.sort(payoffs)
    index = int((1 - confidence_level) * len(sorted_payoffs))
    var = sorted_payoffs[index]
    return var

def calculate_risk_metrics(returns, weights, risk_free_rate=0.0):
    """
    Calculate various risk metrics for the portfolio (Annual Return, 
    Volatility, Sharpe, Sortino, Max Drawdown, Beta, etc.).
    
    :param returns: DataFrame of asset returns
    :param weights: dict or 1D array of portfolio weights
    :param risk_free_rate: float, annual risk-free rate
    :return: dict containing risk metrics
    """
    import yfinance as yf

    if isinstance(weights, dict):
        # Convert to array in same order as returns.columns
        w_array = []
        for col in returns.columns:
            w_array.append(weights.get(col, 0.0))
        weights = np.array(w_array)
    else:
        weights = np.array(weights)

    portfolio_returns = returns.dot(weights)  # Use Pandas dot for proper alignment
    portfolio_std = portfolio_returns.std()
    ann_factor = TRADING_DAYS_PER_YEAR
    ann_return = (1 + portfolio_returns.mean()) ** ann_factor - 1
    ann_vol = portfolio_std * np.sqrt(ann_factor)
    
    # Daily Sharpe
    daily_sharpe = (portfolio_returns.mean() - risk_free_rate / ann_factor) / portfolio_std if portfolio_std > 0 else np.nan
    
    # Annualized Sharpe
    sharpe_ratio = daily_sharpe * np.sqrt(ann_factor) if not np.isnan(daily_sharpe) else np.nan

    # Sortino Ratio calculation
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(ann_factor) if len(downside_returns) > 0 else np.nan
    
    sortino_ratio = (ann_return - risk_free_rate) / downside_std if downside_std and downside_std > 0 else np.nan

    # Maximum Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else np.nan

    # Beta calculation
    beta = np.nan
    try:
        market_data = yf.download('^GSPC', start=returns.index[0], end=returns.index[-1])['Adj Close']
        market_returns = market_data.pct_change().dropna()
        aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        covariance = aligned.cov().iloc[0,1]
        market_var = aligned.iloc[:,1].var()
        beta = covariance / market_var if market_var != 0 else np.nan
    except Exception:
        beta = np.nan

    metrics_data = {
        'Annual Return': {'Value': ann_return, 'Format': 'percentage'},
        'Annual Volatility': {'Value': ann_vol, 'Format': 'percentage'}, 
        'Sharpe Ratio': {'Value': sharpe_ratio, 'Format': 'decimal'},
        'Sortino Ratio': {'Value': sortino_ratio, 'Format': 'decimal'},
        'Max Drawdown': {'Value': max_drawdown, 'Format': 'percentage'},
        'Beta': {'Value': beta, 'Format': 'decimal'},
        'Daily VaR (95%)': {'Value': calculate_var(portfolio_returns, 0.95), 'Format': 'percentage'},
        'CVaR (95%)': {'Value': calculate_cvar(portfolio_returns, 0.95), 'Format': 'percentage'}
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = pd.DataFrame(metrics_data).T.reset_index().rename(columns={'index': 'Metric'})
    metrics_df['Formatted Value'] = metrics_df.apply(
        lambda x: f"{x['Value']:.2%}" if x['Format'] == 'percentage' and pd.notnull(x['Value']) 
        else f"{x['Value']:.2f}" if x['Format'] == 'decimal' and pd.notnull(x['Value'])
        else "N/A", axis=1
    )
    
    return metrics_df

def calculate_risk_contribution(weights, cov_matrix):
    """
    Calculate the risk contribution of each asset in the portfolio.
    
    :param weights: array of portfolio weights
    :param cov_matrix: covariance matrix of returns
    :return: array of risk contributions (each element is the fraction of total risk contributed by that asset)
    """
    weights = np.array(weights)
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    # Marginal risk: d(vol) / d(w_i)
    marginal_risk = (cov_matrix @ weights) / portfolio_vol
    risk_contribution = marginal_risk * weights
    return risk_contribution / portfolio_vol

def calculate_rolling_metrics_extended(returns, weights, window=252, risk_free_rate=0.02):
    """
    Calculate extended rolling risk metrics over a specified window.
    Rolling volatility, rolling Sharpe, rolling VaR, etc.
    
    :param returns: DataFrame of asset returns
    :param weights: dict or array of portfolio weights
    :param window: rolling window size (in trading days)
    :param risk_free_rate: float, annual risk-free rate
    :return: (rolling_vol, rolling_sharpe, rolling_var, rolling_sortino, rolling_max_dd) as Series
    """
    if isinstance(weights, dict):
        w_array = []
        for col in returns.columns:
            w_array.append(weights.get(col, 0.0))
        weights = np.array(w_array)
    else:
        weights = np.array(weights)

    portfolio_returns = returns.values.dot(weights)
    portfolio_series = pd.Series(portfolio_returns, index=returns.index)

    # Rolling Volatility
    rolling_vol = portfolio_series.rolling(window=window).std() * np.sqrt(252)

    # Rolling Sharpe
    rolling_sharpe = (portfolio_series.rolling(window=window).mean() * 252 - risk_free_rate) / rolling_vol

    # Rolling VaR (95% by default)
    rolling_var = portfolio_series.rolling(window=window).quantile(0.05)

    # Rolling Sortino
    def rolling_sortino_func(x):
        neg = x[x < 0]
        down_std = neg.std() * np.sqrt(252) if len(neg) > 0 else 0
        mean_ret = x.mean() * 252
        return (mean_ret - risk_free_rate) / down_std if down_std != 0 else np.nan

    rolling_sortino = portfolio_series.rolling(window=window).apply(rolling_sortino_func, raw=False)

    # Rolling Max Drawdown
    def rolling_max_dd_func(x):
        cum = (1 + x).cumprod()
        max_cum = cum.max()
        min_cum = cum.min()
        if max_cum != 0:
            return (min_cum - max_cum) / max_cum
        else:
            return np.nan

    rolling_max_dd = portfolio_series.rolling(window=window).apply(rolling_max_dd_func, raw=False)

    return rolling_vol, rolling_sharpe, rolling_var, rolling_sortino, rolling_max_dd

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR)
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_relative_metrics(portfolio_returns, benchmark_returns):
    """
    Calculate metrics relative to a benchmark
    """
    # Calculate beta
    covariance = np.cov(portfolio_returns, benchmark_returns)[0,1]
    benchmark_var = np.var(benchmark_returns)
    beta = covariance / benchmark_var
    
    # Calculate alpha
    portfolio_mean = portfolio_returns.mean()
    benchmark_mean = benchmark_returns.mean()
    alpha = portfolio_mean - beta * benchmark_mean
    
    # Calculate information ratio
    tracking_error = (portfolio_returns - benchmark_returns).std()
    information_ratio = (portfolio_mean - benchmark_mean) / tracking_error
    
    return {
        'Beta': beta,
        'Alpha': alpha,
        'Information Ratio': information_ratio,
        'Tracking Error': tracking_error
    }
