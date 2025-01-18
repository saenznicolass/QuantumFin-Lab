import streamlit as st
import pandas as pd

from modules.config.settings import config
from modules.config.constants import OPTIONS_TICKERS, DEFAULT_TICKERS, TRADING_DAYS_PER_YEAR
from modules.data.data_manager import get_data
from modules.visualization.plotly.portfolio_charts import plot_efficient_frontier
from modules.models.portfolio.optimization import run_portfolio_optimization
from modules.risk.metrics.risk_metrics import calculate_risk_contribution
from modules.visualization.streamlit.portfolio_add import (
    add_backtest_section, add_comprehensive_risk_analysis,
    add_factor_analysis_section, add_monte_carlo_stress_section,
    add_portfolio_analysis_section)

def render_portfolio_tab():
    """Renders the Portfolio Optimization Dashboard"""

    # Professional header
    st.markdown("""
        <div style='text-align: center; background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1>Portfolio Optimization Suite</h1>
            <p>Portfolio management and optimization tools</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state if needed
    if 'data' not in st.session_state:
        st.session_state.data = None

    setup_tab, optimization_tab, analysis_tab = st.tabs(["Portfolio Setup", "Optimization", "Analysis & Insights"])

    with setup_tab:
        st.markdown("### Asset Selection & Configuration")

        # Asset selection
        tickers = st.multiselect(
            "Select Assets",
            options=OPTIONS_TICKERS,
            default=DEFAULT_TICKERS,
            help="Choose the assets to include in your portfolio"
        )

        # Date selection
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp.now() - pd.DateOffset(years=2),
                help="Select the start date for historical data"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp.now(),
                help="Select the end date for historical data"
            )
        with col3:
            # Risk-free rate input
            risk_free_rate = st.slider(
                "Annual Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(config.RISK_FREE_RATE * 100),
                step=0.1,
                help="Set the annual risk-free rate for calculations"
            ) / 100.0

        # Store risk-free rate in session state
        st.session_state['risk_free_rate'] = risk_free_rate

        # Trading constraints
        allow_short = st.checkbox(
            "Allow Short Selling",
            value=False,
            help="Enable short selling of assets"
        )
        if st.button("Update Market Data"):
            get_data.clear()
            st.session_state.data = None

        if st.session_state.data is None and tickers:
            with st.spinner("Loading market data..."):
                st.session_state.data = get_data(
                    tickers,
                    start=start_date,
                    end=end_date
                )

    with optimization_tab:
        if not tickers:
            st.warning("Please select at least one asset in the Portfolio Setup tab.")
            return

        # Load or refresh data
        if st.session_state.data is None:
            with st.spinner("Loading market data..."):
                st.session_state.data = get_data(
                    tickers,
                    start=start_date,
                    end=end_date
                )

        if st.session_state.data.empty or len(st.session_state.data.columns) < 2:
            st.error("Insufficient data for optimization. Please select at least two assets.")
            return

        # Optimization parameters
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Optimization Parameters")
            optimization_objective = st.selectbox(
                "Optimization Objective",
                ["Max Sharpe Ratio", "Min Volatility", "Target Return"],
                help="Choose the optimization strategy"
            )

            if optimization_objective == "Target Return":
                required_return = st.slider(
                    "Target Annual Return (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.5
                ) / 100.0
            else:
                required_return = None

            # Add confidence level selection for CVaR
            if optimization_objective == "Min CVaR":
                confidence_level = st.slider(
                    "CVaR Confidence Level",
                    min_value=0.90,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    help="Confidence level for CVaR calculation"
                )
            else:
                confidence_level = 0.95

        with col2:
            st.markdown("#### Portfolio Constraints")
            max_weight = st.slider(
                "Maximum Weight per Asset (%)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=5.0
            ) / 100.0

            min_weight = -1.0 if allow_short else 0.0
            if allow_short:
                min_weight = st.slider(
                    "Minimum Weight per Asset (%)",
                    min_value=-100.0,
                    max_value=0.0,
                    value=-15.0,
                    step=5.0
                ) / 100.0

        # Run optimization
        try:
            with st.spinner("Optimizing portfolio..."):
                optimization_results = run_portfolio_optimization(
                    data=st.session_state.data,
                    allow_short=allow_short,
                    optimization_objective=optimization_objective,
                    required_return=required_return,
                    max_weight=max_weight,
                    min_weight=min_weight,
                    risk_free_rate=risk_free_rate
                )

            # Store results in session state
            st.session_state.update({
                'optimization_results': optimization_results,
                'weights_unconstrained': optimization_results['weights_unconstrained'],
                'weights_constrained': optimization_results['weights_constrained']
            })

            # Compute and store portfolio_returns
            portfolio_returns = st.session_state.data.pct_change().dropna().dot(
                pd.Series(optimization_results['weights_constrained'], index=optimization_results['weights_constrained'].keys())
            )
            st.session_state.portfolio_returns = portfolio_returns

            # Calculate covariance matrix
            cov_matrix = st.session_state.data.pct_change().dropna().cov().values

            # Calculate risk contributions
            rc_unconstrained = calculate_risk_contribution(
                weights=pd.Series(optimization_results['weights_unconstrained']),
                cov_matrix=cov_matrix
            )
            rc_constrained = calculate_risk_contribution(
                weights=pd.Series(optimization_results['weights_constrained']),
                cov_matrix=cov_matrix
            )

            # Add risk contributions to the weights dataframe
            weights_df = optimization_results['weights_df']
            weights_df['Risk U. (%)'] = rc_unconstrained * 100
            weights_df['Risk C. (%)'] = rc_constrained * 100

            # Display optimization results
            st.markdown("### Optimization Results")
            col3, col4, col5 = st.columns([1.4, 2, 1])

            with col3:
                st.subheader("Portfolio Weights")
                st.dataframe(
                    weights_df[['Asset', 'Formatted Unconstrained', 'Formatted Constrained', 'Risk U. (%)', 'Risk C. (%)']]
                    .rename(columns={
                        'Formatted Unconstrained': 'Unconstrained',
                        'Formatted Constrained': 'Constrained'
                    }).set_index('Asset')
                )
            with col4:
                # Efficient Frontier Plot
                st.markdown("### Efficient Frontier")
                plot_efficient_frontier(
                    optimization_results['mu'],
                    optimization_results['S'],
                    (min_weight, max_weight),
                    allow_short,
                    optimization_results,
                    risk_free_rate
                )
            with col5:
                st.subheader("Performance Metrics")
                st.dataframe(
                    optimization_results['performance_df'][['Metric', 'Formatted Unconstrained', 'Formatted Constrained']]
                    .rename(columns={
                        'Formatted Unconstrained': 'Unconstrained',
                        'Formatted Constrained': 'Constrained'
                    }).set_index('Metric')
                )
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")

    with analysis_tab:
        overview_subtab, risk_subtab, mc_stress_subtab, backtest_subtab, factor_subtab = st.tabs([
            "Overview",
            "Risk Analysis",
            "Monte Carlo & Stress",
            "Backtesting",
            "Factor Analysis"
        ])

        with overview_subtab:
            add_portfolio_analysis_section()

        with risk_subtab:
            add_comprehensive_risk_analysis()

        with mc_stress_subtab:
            add_monte_carlo_stress_section()

        with backtest_subtab:
            add_backtest_section()

        with factor_subtab:
            add_factor_analysis_section()
