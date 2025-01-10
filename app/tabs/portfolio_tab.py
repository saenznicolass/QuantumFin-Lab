import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pypfopt import EfficientFrontier

from modules.config.settings import config
from modules.config.constants import OPTIONS_TICKERS, DEFAULT_TICKERS
from modules.data.data_manager import get_data
from modules.models.portfolio.optimization import (
    run_portfolio_optimization,
    compute_efficient_frontier_points,
    compute_cumulative_returns
)

from modules.risk.metrics.risk_metrics import calculate_risk_metrics

from modules.risk.analysis.stress_testing import run_stress_test_scenarios

from modules.risk.backtesting.portfolio_backtester import run_backtest, calculate_backtest_metrics

from modules.models.portfolio.efficient_frontier import (
    calculate_efficient_frontier,
    plot_efficient_frontier_3d,
    get_optimal_portfolio
)
from modules.risk.analysis.factor_analysis import (
    perform_factor_analysis,
    calculate_factor_contribution
)

def render_portfolio_tab():
    """Renders the Portfolio Optimization Dashboard"""
    
    # Professional header
    st.markdown("""
        <div style='text-align: center; background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1>Portfolio Optimization Suite</h1>
            <p>Advanced portfolio management and optimization tools</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state if needed
    if 'data' not in st.session_state:
        st.session_state.data = None
        
    # Main tabs
    setup_tab, optimization_tab, analysis_tab = st.tabs([
        "Portfolio Setup",
        "Optimization",
        "Analysis & Insights"
    ])

    # Portfolio Setup Tab
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
        col1, col2 = st.columns(2)
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
        
        # Data refresh - modified to include date range
        if st.button("Update Market Data"):
            get_data.clear()
            st.session_state.data = None

        # Update how we fetch data to include date range
        if st.session_state.data is None and tickers:
            with st.spinner("Loading market data..."):
                st.session_state.data = get_data(
                    tickers,
                    start=start_date,
                    end=end_date
                )

    # Optimization Tab
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
                value=40.0,
                step=5.0
            ) / 100.0

            min_weight = -1.0 if allow_short else 0.0
            if allow_short:
                min_weight = st.slider(
                    "Minimum Weight per Asset (%)",
                    min_value=-100.0,
                    max_value=0.0,
                    value=-40.0,
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

            # Display optimization results
            st.markdown("### Optimization Results")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Portfolio Weights")
                st.dataframe(
                    optimization_results['weights_df'][['Asset', 'Formatted Unconstrained', 'Formatted Constrained']]
                    .rename(columns={
                        'Formatted Unconstrained': 'Unconstrained',
                        'Formatted Constrained': 'Constrained'
                    }).set_index('Asset')
                )

            with col4:
                st.subheader("Performance Metrics")
                st.dataframe(
                    optimization_results['performance_df'][['Metric', 'Formatted Unconstrained', 'Formatted Constrained']]
                    .rename(columns={
                        'Formatted Unconstrained': 'Unconstrained',
                        'Formatted Constrained': 'Constrained'
                    }).set_index('Metric')
                )

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

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")

    # Analysis Tab
    with analysis_tab:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Risk Analysis",
            "Performance",
            "Backtesting",
            "Factor Analysis"
        ])
        
        with tab1:
            add_risk_analysis_section()
        
        with tab2:
            st.subheader("Correlation Analysis")
            if st.session_state.data is not None:
                corr_matrix = st.session_state.data.pct_change().corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu'
                ))
                fig.update_layout(title="Asset Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)

            if 'optimization_results' not in st.session_state:
                st.info("Run portfolio optimization first to see analysis.")
                return

            st.markdown("### Portfolio Analysis")
            
            # Performance metrics over time
            st.subheader("Historical Performance")
            cumret_constrained = compute_cumulative_returns(
                st.session_state.data,
                st.session_state.weights_constrained
            )
            cumret_unconstrained = compute_cumulative_returns(
                st.session_state.data,
                st.session_state.weights_unconstrained
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumret_constrained.index,
                y=cumret_constrained.values,
                name="Constrained Portfolio"
            ))
            fig.add_trace(go.Scatter(
                x=cumret_unconstrained.index,
                y=cumret_unconstrained.values,
                name="Unconstrained Portfolio"
            ))
            fig.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            add_backtest_section()
            
        with tab4:
            add_factor_analysis_section()

def plot_efficient_frontier(mu, S, weight_bounds, allow_short, optimization_results, risk_free_rate=0.0):
    """Helper function to plot the efficient frontier"""
    frontier_constrained = compute_efficient_frontier_points(
        mu, S, weight_bounds, risk_free_rate
    )
    frontier_unconstrained = compute_efficient_frontier_points(
        mu, S, 
        ((None, None) if allow_short else (0,1)),
        risk_free_rate
    )

    fig = go.Figure()
    
    # Plot frontiers
    if frontier_constrained:
        vol_c, ret_c = zip(*frontier_constrained)
        fig.add_trace(go.Scatter(
            x=vol_c, y=ret_c,
            mode='lines',
            name='Constrained Frontier'
        ))
    
    if frontier_unconstrained:
        vol_u, ret_u = zip(*frontier_unconstrained)
        fig.add_trace(go.Scatter(
            x=vol_u, y=ret_u,
            mode='lines',
            name='Unconstrained Frontier'
        ))

    # Add optimal points
    perf_c = optimization_results['performance_constrained']
    perf_u = optimization_results['performance_unconstrained']
    
    fig.add_trace(go.Scatter(
        x=[perf_c[1]], y=[perf_c[0]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Constrained Optimal'
    ))
    
    fig.add_trace(go.Scatter(
        x=[perf_u[1]], y=[perf_u[0]],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Unconstrained Optimal'
    ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Expected Annual Return",
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def add_risk_analysis_section():
    """Add risk analysis section to the portfolio tab"""
    st.subheader("Risk Analysis")
    
    if 'optimization_results' not in st.session_state:
        st.info("Please run optimization first to see risk analysis.")
        return
            
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Risk Metrics")
        # Calculate daily returns from price data
        asset_returns = st.session_state.data.pct_change().dropna()
        # Calculate risk metrics using the returns
        risk_metrics_df = calculate_risk_metrics(
            asset_returns,
            st.session_state.weights_constrained
        )
        # Display metrics as a formatted table
        st.dataframe(
            risk_metrics_df[['Metric', 'Formatted Value']]
            .rename(columns={'Formatted Value': 'Value'})
            .set_index('Metric')
        )
            
    with col2:
        st.markdown("#### Stress Testing")
        scenarios = {
            "Market Crash": -0.20,
            "Moderate Decline": -0.10,
            "Slight Decline": -0.05,
            "Base Case": 0.0,
            "Slight Recovery": 0.05
        }
        # Ensure portfolio_returns is initialized
        if 'portfolio_returns' in st.session_state:
            stress_results = run_stress_test_scenarios(
                st.session_state.portfolio_returns,
                scenarios
            )
            st.dataframe(stress_results.style.format({
                "Shock (%)": "{:.2%}", 
                "VaR": "{:.2%}", 
                "CVaR": "{:.2%}", 
                "Max Loss": "{:.2%}", 
                "Recovery Days": "{:.0f}"
            }))
        else:
            st.warning("Portfolio returns are not available for stress testing.")

def add_backtest_section():
    """Add backtesting analysis section"""
    st.subheader("Portfolio Backtesting")
    
    if 'weights_constrained' not in st.session_state:
        st.info("Please run optimization first to perform backtesting.")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
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
        
    with col2:
        transaction_costs = st.slider(
            "Transaction Costs (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        ) / 100.0
    
    # Run backtest
    backtest_results = run_backtest(
        st.session_state.data,
        st.session_state.weights_constrained,
        rebalance_frequency=freq_map[rebalance_freq],
        transaction_costs=transaction_costs
    )
    
    # Calculate and display metrics
    metrics = calculate_backtest_metrics(backtest_results)
    
    # Display results
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Backtest Metrics")
        st.dataframe(metrics[['Metric', 'Formatted Value']]
        .rename(columns={'Formatted Value': 'Value'})
        .set_index('Metric')
        )
        
    with col4:
        st.markdown("#### Portfolio Value")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['portfolio_value'],
            name="Portfolio Value"
        ))
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def add_factor_analysis_section():
    """Add factor analysis section to the portfolio tab"""
    st.subheader("Factor Analysis")
    
    if 'optimization_results' not in st.session_state:
        st.info("Please run optimization first to see factor analysis.")
        return
        
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
            st.session_state.weights_constrained
        )
        st.dataframe(factor_contrib.style.format({
            'Coefficient': '{:.4f}',
            'T-Stat': '{:.2f}',
            'P-Value': '{:.4f}'
        }))
