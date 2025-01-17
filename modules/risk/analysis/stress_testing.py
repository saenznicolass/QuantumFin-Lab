# modules/risk/analysis/stress_testing.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def monte_carlo_portfolio_simulation(
    returns, 
    weights, 
    n_simulations=10000, 
    time_horizon=252
):
    """
    Simulate portfolio returns using a basic lognormal assumption.
    
    :param returns: DataFrame of asset returns
    :param weights: dict or array of portfolio weights
    :param n_simulations: number of Monte Carlo runs
    :param time_horizon: number of days to project
    :param return_paths: bool, if True return the entire paths
    :return: array of simulated final returns or daily paths
    """
    if isinstance(weights, dict):
        w_array = []
        for col in returns.columns:
            w_array.append(weights.get(col, 0.0))
        weights = np.array(w_array)
    else:
        weights = np.array(weights)

    # Convert to log returns
    log_returns = np.log1p(returns)
    # Calculate daily mean and covariance
    daily_mean = log_returns.mean()
    daily_cov = log_returns.cov()

    # Generate random draws for each day in the time horizon
    # Assuming these are log returns
    simulated = np.random.multivariate_normal(daily_mean, daily_cov, (n_simulations, time_horizon))

    # Aggregate daily log returns for each simulation, then convert to final return
    portfolio_daily = simulated.dot(weights)
    # Sum log returns and exponentiate
    final_returns = np.exp(portfolio_daily.cumsum(axis=1)[:, -1]) - 1
    daily_paths = np.exp(portfolio_daily.cumsum(axis=1)) - 1
    
    # Simplified transformation
    final_values = (1 + portfolio_daily).cumprod(axis=1)
    
    # Add normalization
    final_values = final_values / final_values[:, 0].reshape(-1, 1)
    
    return final_returns, final_values, daily_paths

def run_stress_test_scenarios(portfolio_returns, scenarios_dict, confidence_level=0.95):
    """
    Apply each stress scenario (as a shock) to the portfolio returns 
    and compute stats like VaR, CVaR, max loss, etc.
    
    :param portfolio_returns: 1D array of daily portfolio returns
    :param scenarios_dict: dict of scenario -> shock_value (e.g. -0.2 for -20%)
    :param confidence_level: float (e.g., 0.95)
    :return: DataFrame with scenario results
    """
    results = []
    for scenario, shock in scenarios_dict.items():
        # Shock the returns
        shocked_returns = portfolio_returns * (1 + shock)
        # VaR
        var_s = np.percentile(shocked_returns, (1 - confidence_level) * 100)
        cvar_s = shocked_returns[shocked_returns <= var_s].mean()
        max_loss = shocked_returns.min() if len(shocked_returns) > 0 else np.nan
        # Some placeholder for "Recovery time" or any other metric
        # For demonstration, define "recovery_time" as how many days returns < 0
        recovery_time = np.sum(shocked_returns < 0)

        results.append({
            'Scenario': scenario,
            'Shock (%)': shock,
            'VaR': var_s,
            'CVaR': cvar_s,
            'Max Loss': max_loss,
            'Recovery Days': recovery_time
        })
    df_scenarios = pd.DataFrame(results)
    return df_scenarios

def plot_stress_test_scenarios(scenarios_df):
    """
    Create a bar chart of scenario shock impacts.
    
    :param scenarios_df: DataFrame with scenario results (Scenario, Shock(%), VaR, CVaR, etc.)
    :return: plotly Figure
    """
    fig = go.Figure()
    # Sort by shock if you wish:
    sorted_df = scenarios_df.sort_values(by='Shock (%)')

    fig.add_trace(go.Bar(
        x=sorted_df['Scenario'],
        y=sorted_df['Shock (%)'],
        name='Impact',
        marker_color='red',
        opacity=0.7
    ))
    fig.update_layout(
        title='Stress Test Scenario Impacts',
        xaxis_title='Scenario',
        yaxis_title='Shock (%)',
        yaxis_tickformat='%',
        barmode='group',
        hovermode='x unified'
    )
    return fig
