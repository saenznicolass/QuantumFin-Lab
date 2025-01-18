from scipy import stats
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.risk.analysis.stress_testing import (
    run_stress_test_scenarios, monte_carlo_portfolio_simulation)
from modules.risk.backtesting.portfolio_backtester import (
    run_backtest, calculate_backtest_metrics)


def run_detailed_monte_carlo_analysis(is_constrained: bool, confidence_level: float):
    """Enhanced Monte Carlo analysis with additional metrics and visualizations"""
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained

    # Usar los parámetros del optimizador
    mu = st.session_state.optimization_results['mu']
    S = st.session_state.optimization_results['S']

    portfolio_type = "constrained" if is_constrained else "unconstrained"

    col1, col2 = st.columns(2)
    with col1:
        n_sim = st.number_input(
            "Number of Simulations",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            key=f"n_sim_{portfolio_type}"  # Añadido key único
        )
    with col2:
        time_horizon = st.number_input(
            "Time Horizon (days)",
            min_value=21,
            max_value=504,
            value=252,
            step=21,
            key=f"time_horizon_{portfolio_type}"  # Añadido key único
        )

    returns = st.session_state.data.pct_change().dropna()
    sim_returns, final_values, mc_paths = monte_carlo_portfolio_simulation(
        returns, weights,
        n_simulations=n_sim,
        time_horizon=time_horizon
    )

    # Calculate Monte Carlo statistics
    sim_mean = sim_returns.mean()
    sim_std = sim_returns.std()
    sim_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
    sim_cvar = sim_returns[sim_returns <= sim_var].mean()
    sim_skew = pd.Series(sim_returns).skew()
    sim_kurt = pd.Series(sim_returns).kurtosis()

    # Display summary statistics
    st.subheader("Monte Carlo Summary Statistics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Expected Return", f"{sim_mean:.2%}")
    with col2:
        st.metric("Volatility", f"{sim_std:.2%}")
    with col3:
        st.metric("VaR", f"{sim_var:.2%}")
    with col4:
        st.metric("CVaR", f"{sim_cvar:.2%}")
    with col5:
        st.metric("Skewness", f"{sim_skew:.2f}")
    with col6:
        st.metric("Kurtosis", f"{sim_kurt:.2f}")
    # Enhanced visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Distribution with fitted normal
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=sim_returns,
            nbinsx=50,
            name="MC Returns",
            histnorm='probability'
        ))

        # Add normal distribution fit
        x_range = np.linspace(sim_returns.min(), sim_returns.max(), 100)
        y_normal = stats.norm.pdf(x_range, sim_mean, sim_std)

        fig_dist.add_trace(go.Scatter(
            x=x_range,
            y=y_normal,
            name="Normal Fit",
            line=dict(color='red', dash='dash')
        ))

        fig_dist.add_vline(
            x=sim_var,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({confidence_level:.0%})"
        )

        fig_dist.update_layout(
            title="Distribution of Simulated Returns",
            xaxis_title="Return",
            yaxis_title="Probability",
            showlegend=True
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Path analysis with confidence bands
        final_values = final_values
        median_path = np.median(final_values, axis=0)
        lower_band = np.percentile(final_values, (1 - confidence_level) * 100, axis=0)
        upper_band = np.percentile(final_values, confidence_level * 100, axis=0)

        fig_paths = go.Figure()

        # Add 500 paths
        for i in range(min(500, mc_paths.shape[0])):
            fig_paths.add_trace(go.Scatter(
                y=final_values[i],
                mode='lines',
                opacity=0.1,
                line=dict(color='blue'),
                showlegend=False
            ))

        # Add median and confidence bands
        fig_paths.add_trace(go.Scatter(
            y=median_path,
            mode='lines',
            name='Median Path',
            line=dict(color='red', width=2)
        ))

        fig_paths.add_trace(go.Scatter(
            y=upper_band,
            mode='lines',
            name=f'{confidence_level:.0%} Confidence Band',
            line=dict(color='green', dash='dash')
        ))

        fig_paths.add_trace(go.Scatter(
            y=lower_band,
            mode='lines',
            name='Lower Band',
            line=dict(color='green', dash='dash'),
            fill='tonexty'
        ))

        fig_paths.update_layout(
            title=f"Monte Carlo Paths with {confidence_level:.0%} Confidence Bands",
            xaxis_title="Time (days)",
            yaxis_title="Portfolio Value",
            showlegend=True
        )
        st.plotly_chart(fig_paths, use_container_width=True)

def run_detailed_stress_analysis(is_constrained: bool, confidence_level: float):
    """Enhanced stress testing analysis"""
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained
    portfolio_returns = st.session_state.data.pct_change().dropna().dot(pd.Series(weights))

    portfolio_type = "constrained" if is_constrained else "unconstrained"

    # Custom scenario configuration
    st.subheader("Scenario Configuration")
    col1, col2 = st.columns(2)

    with col1:
        market_crash = st.slider(
            "Market Crash Scenario (%)",
            min_value=-50.0,
            max_value=0.0,
            value=-20.0,
            step=5.0,
            key=f"market_crash_{portfolio_type}"  # Añadido key único
        ) / 100.0

        moderate_decline = st.slider(
            "Moderate Decline Scenario (%)",
            min_value=-30.0,
            max_value=0.0,
            value=-10.0,
            step=2.5,
            key=f"moderate_decline_{portfolio_type}"  # Añadido key único
        ) / 100.0

    with col2:
        recovery = st.slider(
            "Recovery Scenario (%)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=2.5,
            key=f"recovery_{portfolio_type}"  # Añadido key único
        ) / 100.0

        custom_shock = st.slider(
            "Custom Scenario (%)",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=5.0,
            key=f"custom_shock_{portfolio_type}"  # Añadido key único
        ) / 100.0

    # Define scenarios
    scenarios = {
        "Market Crash": market_crash,
        "Moderate Decline": moderate_decline,
        "Slight Decline": -0.05,
        "Base Case": 0.0,
        "Recovery": recovery,
        "Custom Scenario": custom_shock
    }

    # Run stress test
    scenarios_df = run_stress_test_scenarios(portfolio_returns, scenarios)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stress Test Results")
        st.dataframe(
            scenarios_df.style
                .background_gradient(cmap='RdYlGn_r', subset=['Shock (%)', 'VaR', 'CVaR', 'Max Loss'])
                .format({
                    'Shock (%)': '{:.1%}',
                    'VaR': '{:.2%}',
                    'CVaR': '{:.2%}',
                    'Max Loss': '{:.2%}',
                    'Recovery Days': '{:.0f}'
                })
        )

    with col2:
        # Visualize stress test results
        fig = go.Figure()

        # Bar chart for different metrics
        for metric in ['VaR', 'CVaR', 'Max Loss']:
            fig.add_trace(go.Bar(
                name=metric,
                x=scenarios_df['Scenario'],
                y=scenarios_df[metric],
                text=[f"{v:.1%}" for v in scenarios_df[metric]],
                textposition='auto',
            ))

        fig.update_layout(
            title="Stress Test Metrics by Scenario",
            barmode='group',
            yaxis_tickformat='%',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # Additional analysis
    st.subheader("Recovery Analysis")
    fig_recovery = go.Figure()
    fig_recovery.add_trace(go.Bar(
        name="Expected Recovery Days",
        x=scenarios_df['Scenario'],
        y=scenarios_df['Recovery Days'],
        text=[f"{v:.0f}" for v in scenarios_df['Recovery Days']],
        textposition='auto',
    ))

    fig_recovery.update_layout(
        title="Expected Recovery Period by Scenario",
        yaxis_title="Trading Days",
        showlegend=False
    )

    st.plotly_chart(fig_recovery, use_container_width=True)

def run_detailed_backtest(is_constrained: bool, rebalance_freq: str, transaction_costs: float, benchmark: str = None):
    """Run detailed backtest analysis for a specific portfolio type"""
    weights = st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained

    # Get optimization metrics for comparison
    if is_constrained:
        expected_return, expected_vol, expected_sharpe = st.session_state.optimization_results['performance_constrained']
    else:
        expected_return, expected_vol, expected_sharpe = st.session_state.optimization_results['performance_unconstrained']

    try:
        # Run backtest with same data used in optimization
        backtest_results = run_backtest(
            st.session_state.data,
            weights,
            rebalance_frequency=rebalance_freq,
            transaction_costs=transaction_costs,
            benchmark=benchmark
        )

        # Calculate metrics using the same risk-free rate as optimization
        metrics = calculate_backtest_metrics(backtest_results)

        # Detailed metrics and visualizations
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            st.markdown("#### Detailed Metrics")
            st.dataframe(
                metrics[['Metric', 'Formatted Value']]
                .rename(columns={'Formatted Value': 'Value'})
                .set_index('Metric')
            )

        with col2:
            st.markdown("#### Expected vs Realized Metrics")
            realized_return = metrics.loc[metrics['Metric'] == 'Annualized Return', 'Value'].iloc[0]
            diff_return = realized_return - expected_return
            st.metric(
                "Annual Return",
                f"{realized_return:.2%}",
                f"{diff_return:+.2%} vs Expected"
            )

            realized_vol = metrics.loc[metrics['Metric'] == 'Annual Volatility', 'Value'].iloc[0]
            diff_vol = realized_vol - expected_vol
            st.metric(
                "Volatility",
                f"{realized_vol:.2%}",
                f"{diff_vol:+.2%} vs Expected"
            )

            realized_sharpe = metrics.loc[metrics['Metric'] == 'Sharpe Ratio', 'Value'].iloc[0]
            diff_sharpe = realized_sharpe - expected_sharpe
            st.metric(
                "Sharpe Ratio",
                f"{realized_sharpe:.2f}",
                f"{diff_sharpe:+.2f} vs Expected"
            )

        with col3:
            st.markdown("#### Portfolio Evolution")
            fig = go.Figure()

            # Portfolio value
            fig.add_trace(go.Scatter(
                x=backtest_results.index,
                y=backtest_results['portfolio_value'],
                name="Portfolio Value"
            ))

            # Add benchmark if selected
            if benchmark and 'benchmark_value' in backtest_results.columns:
                fig.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['benchmark_value'],
                    name=f"Benchmark ({benchmark})",
                    line=dict(dash='dash')
                ))

            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Additional analysis
        st.markdown("#### Risk Analysis")
        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:
            # Drawdown plot
            fig_dd = go.Figure()
            drawdown = (backtest_results['portfolio_value'] / backtest_results['portfolio_value'].cummax() - 1)
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                fill='tozeroy',
                name="Drawdown"
            ))
            fig_dd.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                yaxis_tickformat='%'
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with risk_col2:
            # Rolling volatility
            fig_vol = go.Figure()
            rolling_vol = backtest_results['returns'].rolling(window=21).std() * np.sqrt(252)
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name="Rolling Volatility"
            ))
            fig_vol.update_layout(
                title="21-Day Rolling Volatility (Annualized)",
                xaxis_title="Date",
                yaxis_title="Volatility",
                yaxis_tickformat='%'
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        return
