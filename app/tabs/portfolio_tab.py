from scipy import stats
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pypfopt import EfficientFrontier

from modules.config.settings import config
from modules.config.constants import OPTIONS_TICKERS, DEFAULT_TICKERS, TRADING_DAYS_PER_YEAR
from modules.data.data_manager import get_data
from modules.models.portfolio.optimization import (
    run_portfolio_optimization,
    compute_efficient_frontier_points,
    compute_cumulative_returns
)

from modules.risk.metrics.risk_metrics import calculate_risk_metrics, calculate_risk_contribution, calculate_rolling_metrics_extended

from modules.risk.analysis.stress_testing import run_stress_test_scenarios, monte_carlo_portfolio_simulation

from modules.risk.backtesting.portfolio_backtester import run_backtest, calculate_backtest_metrics

from modules.risk.analysis.factor_analysis import (
    perform_factor_analysis,
    calculate_factor_contribution
)

from modules.risk.analysis.tail_analysis import extreme_value_analysis, create_qq_plot

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
            # rolling_col1, rolling_col2 = st.columns([2, 1])
            
            with metrics_col3:
                st.markdown("#### Rolling Risk Plots")
                rolling_window = st.number_input(
                    "Rolling Window (days)", 
                    21, 500, 30, 1,
                    key=f"rolling_window_{is_constrained}"
                )
                display_rolling_metrics(is_constrained, rolling_window)
            with metrics_col4:
                # Tail Risk Analysis dentro de la misma columna
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

def plot_simulation_distribution(sim_returns, is_constrained):
    """Plot the distribution of simulated returns"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim_returns,
        nbinsx=50,
        name=f"{'Constrained' if is_constrained else 'Unconstrained'} Returns"
    ))
    fig.update_layout(
        title="Distribution of Simulated Returns",
        xaxis_title="Return",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_simulation_paths(mc_paths, is_constrained):
    """Plot Monte Carlo simulation paths"""
    fig = go.Figure()
    
    # Plot a sample of paths
    for i in range(min(20, mc_paths.shape[0])):
        fig.add_trace(go.Scatter(
            y=mc_paths[i],
            mode='lines',
            opacity=0.5,
            name=f"Path {i+1}",
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{'Constrained' if is_constrained else 'Unconstrained'} Monte Carlo Paths",
        xaxis_title="Time",
        yaxis_title="Portfolio Value"
    )
    st.plotly_chart(fig, use_container_width=True)

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
    sim_returns, mc_paths = monte_carlo_portfolio_simulation(
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
        st.metric("VaR", f"{-sim_var:.2%}")
    with col4:
        st.metric("CVaR", f"{-sim_cvar:.2%}")
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
        final_values = np.exp(np.log1p(mc_paths).cumsum(axis=1))
        median_path = np.median(final_values, axis=0)
        lower_band = np.percentile(final_values, (1 - confidence_level) * 100, axis=0)
        upper_band = np.percentile(final_values, confidence_level * 100, axis=0)
        
        fig_paths = go.Figure()
        
        # Add sample paths
        for i in range(min(10, mc_paths.shape[0])):
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
