
import streamlit as st
import plotly.graph_objects as go

from modules.models.portfolio.optimization import compute_cumulative_returns
from modules.risk.analysis.tail_analysis import (
    extreme_value_analysis, create_qq_plot)
from modules.visualization.streamlit.portfolio_run import (
    run_detailed_backtest, run_detailed_monte_carlo_analysis,
    run_detailed_stress_analysis)
from modules.visualization.streamlit.portfolio_display import (
    display_expected_risk_metrics, display_factor_analysis,
    display_realized_risk_metrics, display_rolling_metrics)


def add_portfolio_analysis_section():
    """Portfolio correlation and performance analysis"""
    st.subheader("Portfolio Analysis")

    if st.session_state.data is not None:
        # Correlation matrix
        corr_matrix = st.session_state.data.pct_change().corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu'
        ))
        fig_corr.update_layout(title="Asset Correlation Matrix")

    if 'optimization_results' not in st.session_state:
        st.info("Run portfolio optimization first to see analysis.")
        return

    # Performance over time
    st.subheader("Historical Performance")
    cumret_constrained = compute_cumulative_returns(
        st.session_state.data,
        st.session_state.weights_constrained
    )
    cumret_unconstrained = compute_cumulative_returns(
        st.session_state.data,
        st.session_state.weights_unconstrained
    )

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=cumret_constrained.index,
        y=cumret_constrained.values,
        name="Constrained Portfolio"
    ))
    fig_perf.add_trace(go.Scatter(
        x=cumret_unconstrained.index,
        y=cumret_unconstrained.values,
        name="Unconstrained Portfolio"
    ))
    fig_perf.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified'
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_corr, use_container_width=True)
    with col2:
        st.plotly_chart(fig_perf, use_container_width=True)

def add_comprehensive_risk_analysis():
    """Comprehensive risk analysis including realized and expected metrics"""
    st.subheader("Basic Risk Analysis")

    if 'optimization_results' not in st.session_state:
        st.info("Please run optimization first to see risk analysis.")
        return

    # Constrained vs Unconstrained comparison for basic metrics
    const_tab, unconst_tab = st.tabs(["Constrained Portfolio", "Unconstrained Portfolio"])

    for is_constrained, tab in [(True, const_tab), (False, unconst_tab)]:
        with tab:
            # Risk Metrics Section
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns([0.8, 0.8, 2, 1.5])

            with metrics_col1:
                st.markdown("#### Expected Metrics")
                display_expected_risk_metrics(is_constrained)

            with metrics_col2:
                st.markdown("#### Realized Metrics")
                display_realized_risk_metrics(is_constrained)

            # Rolling Analysis Section
            with metrics_col3:
                st.markdown("#### Rolling Risk Plots")
                rolling_window = st.number_input(
                    "Rolling Window (days)",
                    21, 500, 30, 1,
                    key=f"rolling_window_{is_constrained}"
                )
                display_rolling_metrics(is_constrained, rolling_window)
            with metrics_col4:
                st.markdown("#### Tail Risk Analysis")
                tail_risk = extreme_value_analysis(
                    st.session_state.data.pct_change().dropna(),
                    st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained,
                    threshold_percentile=5
                )
                tail_col1, tail_col2 = st.columns(2)

                with tail_col1:
                    st.metric("Tail VaR (5%)", f"{tail_risk['tail_var']:.2%}" if tail_risk['tail_var'] else "N/A")
                with tail_col2:
                    st.metric("Tail CVaR (5%)", f"{tail_risk['tail_cvar']:.2%}" if tail_risk['tail_cvar'] else "N/A")

                # Q-Q Plot
                qq_fig = create_qq_plot(
                    st.session_state.data.pct_change().dropna(),
                    st.session_state.weights_constrained if is_constrained else st.session_state.weights_unconstrained
                )
                st.plotly_chart(qq_fig, use_container_width=True)


def add_monte_carlo_stress_section():
    """Monte Carlo simulation and stress testing analysis"""
    st.subheader("Monte Carlo & Stress Testing Analysis")

    # Global parameters
    confidence_level = st.slider(
        "Confidence Level for Risk Metrics",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01
    )

    tab1, tab2 = st.tabs(["Monte Carlo Analysis", "Stress Testing"])

    with tab1:
        mc_const_tab, mc_unconst_tab = st.tabs(["Constrained Portfolio", "Unconstrained Portfolio"])

        with mc_const_tab:
            run_detailed_monte_carlo_analysis(True, confidence_level)
        with mc_unconst_tab:
            run_detailed_monte_carlo_analysis(False, confidence_level)

    with tab2:
        stress_const_tab, stress_unconst_tab = st.tabs(["Constrained Portfolio", "Unconstrained Portfolio"])

        with stress_const_tab:
            run_detailed_stress_analysis(True, confidence_level)
        with stress_unconst_tab:
            run_detailed_stress_analysis(False, confidence_level)

def add_backtest_section():
    """Add backtesting analysis section"""
    st.subheader("Portfolio Backtesting")

    if 'weights_constrained' not in st.session_state:
        st.info("Please run optimization first to perform backtesting.")
        return

    # Backtesting Configuration
    st.markdown("### Backtesting Configuration")

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
            index=0
        )

        freq_map = {
            "Monthly": "M",
            "Quarterly": "Q",
            "Semi-Annual": "6M",
            "Annual": "Y"
        }

    with config_col2:
        transaction_costs = st.slider(
            "Transaction Costs (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        ) / 100.0

    with config_col3:
        benchmark = st.selectbox(
            "Benchmark",
            ["^GSPC", "^DJI", "^IXIC", "None"],
            index=0,
            help="Select benchmark for comparison"
        )

    # Run backtest for both constrained and unconstrained portfolios
    const_tab, unconst_tab = st.tabs(["Constrained Portfolio", "Unconstrained Portfolio"])

    with const_tab:
        run_detailed_backtest(
            True,
            freq_map[rebalance_freq],
            transaction_costs,
            benchmark if benchmark != "None" else None
        )

    with unconst_tab:
        run_detailed_backtest(
            False,
            freq_map[rebalance_freq],
            transaction_costs,
            benchmark if benchmark != "None" else None
        )

def add_factor_analysis_section():
    """Add factor analysis section to the portfolio tab"""
    st.subheader("Factor Analysis")

    if 'optimization_results' not in st.session_state:
        st.info("Please run optimization first to see factor analysis.")
        return

    returns = st.session_state.data.pct_change().dropna()

    # Usar los pesos optimizados para ambos análisis
    const_tab, unconst_tab = st.tabs(["Constrained Portfolio", "Unconstrained Portfolio"])

    with const_tab:
        display_factor_analysis(True)

    with unconst_tab:
        display_factor_analysis(False)
