import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from modules.risk.analysis.tail_analysis import (
    extreme_value_analysis, create_qq_plot)
from modules.risk.metrics.risk_metrics import (
    calculate_risk_metrics, calculate_rolling_metrics_extended)
from modules.risk.analysis.factor_analysis import (
    perform_factor_analysis, calculate_factor_contribution)

def display_rolling_metrics(is_constrained, window):
    """Display rolling risk metrics visualization"""
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained
    returns = st.session_state.data.pct_change().dropna()
    
    rolling_vol, rolling_sharpe, rolling_var, rolling_sortino, rolling_max_dd = calculate_rolling_metrics_extended(
        returns, weights, window=window, risk_free_rate=st.session_state.risk_free_rate
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns.index, y=rolling_vol,
        name="Volatility", line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=returns.index, y=rolling_sharpe,
        name="Sharpe Ratio", line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=returns.index, y=rolling_max_dd,
        name="Max Drawdown", line=dict(color='red')
    ))
    fig.update_layout(
        title=f"Rolling Risk Metrics ({window}-day window)",
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def display_tail_analysis(is_constrained):
    """Display tail risk analysis and EVT"""
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained
    returns = st.session_state.data.pct_change().dropna()
    portfolio_returns = returns.dot(pd.Series(weights))
    
    # EVT Analysis
    tail_risk = extreme_value_analysis(returns, weights, threshold_percentile=5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tail VaR (5%)", f"{tail_risk['tail_var']:.2%}" if tail_risk['tail_var'] else "N/A")
        st.metric("Tail CVaR (5%)", f"{tail_risk['tail_cvar']:.2%}" if tail_risk['tail_cvar'] else "N/A")
    
    with col2:
        # Q-Q Plot
        qq_fig = create_qq_plot(returns, weights)
        st.plotly_chart(qq_fig, use_container_width=True)

def display_realized_risk_metrics(is_constrained):
    """Display realized risk metrics for the selected portfolio type"""
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained
    asset_returns = st.session_state.data.pct_change().dropna()
    
    risk_metrics = calculate_risk_metrics(
        asset_returns,
        weights,
        risk_free_rate=st.session_state.risk_free_rate
    )
    
    st.dataframe(
        risk_metrics[['Metric', 'Formatted Value']]
        .rename(columns={'Formatted Value': 'Value'})
        .set_index('Metric')
    )

def display_expected_risk_metrics(is_constrained):
    """Display expected risk metrics for the selected portfolio type"""
    # Use optimization results directly
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained
    mu = st.session_state.optimization_results['mu']
    
    # Usar el retorno esperado calculado por el optimizador
    if is_constrained:
        expected_return = st.session_state.optimization_results['performance_constrained'][0]
    else:
        expected_return = st.session_state.optimization_results['performance_unconstrained'][0]
    
    asset_returns = st.session_state.data.pct_change().dropna()
    risk_metrics = calculate_risk_metrics(
        asset_returns,
        weights,
        risk_free_rate=st.session_state.risk_free_rate,
        expected_return=expected_return  # Usar el valor del optimizador
    )
    
    # Filter for expected metrics only
    expected_metrics = risk_metrics[
        risk_metrics['Metric'].str.startswith('Expected')
    ]
    
    st.dataframe(
        expected_metrics[['Metric', 'Formatted Value']]
        .rename(columns={'Formatted Value': 'Value'})
        .set_index('Metric')
    )

def display_factor_analysis(is_constrained: bool):
    """Display factor analysis for the selected portfolio type"""
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained
    returns = st.session_state.data.pct_change().dropna()
    
    # Factor Analysis
    factor_loadings, factor_returns = perform_factor_analysis(returns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Factor Loadings")
        st.dataframe(factor_loadings.style.format("{:.4f}"))
        
    with col2:
        st.markdown("#### Factor Contributions")
        factor_contrib = calculate_factor_contribution(
            returns,
            weights
        )
        st.dataframe(factor_contrib.style.format({
            'Coefficient': '{:.4f}',
            'T-Stat': '{:.2f}',
            'P-Value': '{:.4f}'
        }))