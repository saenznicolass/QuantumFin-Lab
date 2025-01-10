# app/tabs/risk_management_tab.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Import from the new risk modules
from modules.risk.metrics.risk_metrics import (
    calculate_var,
    calculate_risk_metrics,
    calculate_risk_contribution,
    calculate_rolling_metrics_extended
)
from modules.risk.analysis.tail_analysis import (
    extreme_value_analysis,
    analyze_tail_risk,
    create_qq_plot
)
from modules.risk.analysis.stress_testing import (
    monte_carlo_portfolio_simulation,
    run_stress_test_scenarios,
    plot_stress_test_scenarios
)

def render_risk_management_tab():
    """
    Renders the Risk Management tab, including VaR/CVaR, 
    stress testing, tail risk analysis, rolling metrics, etc.
    """
    st.header("Risk Management")

    # Check if portfolio optimization has been run
    if ('weights_constrained' not in st.session_state or 
        'weights_unconstrained' not in st.session_state or 
        'data' not in st.session_state):
        st.error("Please run the Portfolio Optimization first to obtain optimized portfolio weights and data.")
        return

    # Let user choose which weights to use, default to unconstrained
    weight_type = st.radio(
        "Select Portfolio Weights",
        ["Unconstrained", "Constrained"],
        index=0,  # Default to unconstrained
        help="Choose which optimized portfolio weights to analyze"
    )

    # Select appropriate weights based on user choice
    weights = (st.session_state['weights_unconstrained'] 
              if weight_type == "Unconstrained" 
              else st.session_state['weights_constrained'])

    # Rest of the risk management tab code
    data = st.session_state['data']
    risk_free_rate = st.session_state.get('risk_free_rate', 0.05)

    # Convert prices to returns
    returns = data.pct_change().dropna()
    st.subheader("Basic Portfolio Risk Metrics")

    # 1) Calculate basic risk metrics
    risk_metrics_df = calculate_risk_metrics(returns, weights, risk_free_rate)

    # Show complete metrics table
    st.markdown("#### Complete Risk Metrics")
    st.dataframe(
        risk_metrics_df[['Metric', 'Formatted Value']]
        .rename(columns={'Formatted Value': 'Value'})
        .set_index('Metric')
    )

    # 2) Calculate VaR and CVaR
    st.subheader("Value at Risk (VaR) Analysis")
    confidence_level = st.slider("Confidence Level for VaR/CVaR", 0.80, 0.99, 0.95, 0.01)
    portfolio_returns = returns.values.dot(np.array([weights.get(c, 0.0) for c in returns.columns]))
    var_value = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    cvar_value = portfolio_returns[portfolio_returns <= var_value].mean() if len(portfolio_returns) > 0 else np.nan

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(f"VaR ({confidence_level:.0%})", f"{-var_value:.2%}")
    with col_b:
        st.metric(f"CVaR ({confidence_level:.0%})", f"{-cvar_value:.2%}" if not np.isnan(cvar_value) else "N/A")

    # Distribution histogram with VaR lines
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, name="Returns"))
    fig_dist.add_vline(
        x=var_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR ({confidence_level:.0%})"
    )
    if not np.isnan(cvar_value):
        fig_dist.add_vline(
            x=cvar_value,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"CVaR ({confidence_level:.0%})"
        )
    fig_dist.update_layout(
        title="Portfolio Return Distribution",
        xaxis_title="Return",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Monte Carlo Simulation")
    n_sim = st.number_input("Number of Simulations", 1000, 1000000, 10000, 1000)
    sim_returns = monte_carlo_portfolio_simulation(returns, weights, n_simulations=n_sim)
    sim_mean = sim_returns.mean()
    sim_std = sim_returns.std()
    sim_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
    sim_cvar = sim_returns[sim_returns <= sim_var].mean() if len(sim_returns) else np.nan

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Simulated Mean Return", f"{sim_mean:.2%}")
    col_m2.metric("Simulated Volatility", f"{sim_std:.2%}")
    col_m3.metric(f"Simulated VaR ({confidence_level:.0%})", f"{-sim_var:.2%}")

    col_m4, col_m5 = st.columns(2)
    col_m4.metric(f"Simulated CVaR ({confidence_level:.0%})", f"{-sim_cvar:.2%}" if not np.isnan(sim_cvar) else "N/A")

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Histogram(x=sim_returns, nbinsx=50, name="Simulated Returns"))
    fig_sim.update_layout(
        title=f"Distribution of {n_sim:,} Simulated Returns",
        xaxis_title="Return",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.subheader("Tail Risk / Extreme Value Analysis")
    tail_risk = extreme_value_analysis(returns, weights, threshold_percentile=5)
    col_t1, col_t2 = st.columns(2)
    col_t1.metric("Tail VaR (5%)", f"{tail_risk['tail_var']:.2%}" if tail_risk['tail_var'] else "N/A")
    col_t2.metric("Tail CVaR (5%)", f"{tail_risk['tail_cvar']:.2%}" if tail_risk['tail_cvar'] else "N/A")

    # Q-Q plot
    st.subheader("Return Distribution Analysis (Q-Q Plot)")
    qq_fig = create_qq_plot(returns, weights)
    st.plotly_chart(qq_fig, use_container_width=True)

    # Risk Contribution
    st.subheader("Risk Contribution Analysis")
    cov_matrix = returns.cov().values
    w_array = np.array([weights.get(c, 0.0) for c in returns.columns])
    rc = calculate_risk_contribution(w_array, cov_matrix)
    risk_contrib_df = pd.DataFrame({
        'Asset': returns.columns,
        'Weight': w_array,
        'Risk Contribution (%)': rc * 100
    })
    st.dataframe(risk_contrib_df.style.format({'Weight': '{:.2%}', 'Risk Contribution (%)': '{:.2f}%'}))

    # Rolling metrics
    st.subheader("Rolling Risk Metrics")
    rolling_window = st.number_input("Rolling Window (days)", 21, 500, 252, 1)
    rolling_vol, rolling_sharpe, rolling_var, rolling_sortino, rolling_max_dd = calculate_rolling_metrics_extended(returns, weights, window=rolling_window, risk_free_rate=risk_free_rate)

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=returns.index, y=rolling_vol, name="Volatility", line=dict(color='blue')
    ))
    fig_rolling.add_trace(go.Scatter(
        x=returns.index, y=rolling_sharpe, name="Sharpe Ratio", line=dict(color='green')
    ))
    fig_rolling.add_trace(go.Scatter(
        x=returns.index, y=rolling_max_dd, name="Max Drawdown", line=dict(color='red')
    ))
    fig_rolling.update_layout(
        title=f"Rolling Risk Metrics ({rolling_window}-day)",
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    # Stress Testing
    st.subheader("Stress Testing")
    # Example scenario shocks
    scenarios_dict = {
        "Market Crash (-20%)": -0.20,
        "Interest Rate Shock (+2%)": -0.05,
        "Volatility Spike (+50%)": -0.10,
        "Custom Scenario": 0.0  # user can override
    }
    custom_shock = st.slider("Custom Shock (%)", min_value=-50.0, max_value=50.0, value=0.0, step=1.0) / 100.0
    scenarios_dict["Custom Scenario"] = custom_shock

    scenario_df = run_stress_test_scenarios(portfolio_returns, scenarios_dict, confidence_level=confidence_level)
    st.dataframe(
        scenario_df.style
            .background_gradient(cmap='RdYlGn_r', subset=['Shock (%)', 'VaR', 'CVaR', 'Max Loss'])
            .format({'Shock (%)': '{:.1%}', 'VaR': '{:.2%}', 'CVaR': '{:.2%}', 'Max Loss': '{:.2%}'})
    )

    fig_stress = plot_stress_test_scenarios(scenario_df)
    st.plotly_chart(fig_stress, use_container_width=True)

    st.info("All risk metrics and analyses are for educational purposes only and are not investment advice.")
