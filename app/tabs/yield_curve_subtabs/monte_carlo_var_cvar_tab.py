import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats
from modules.models.interest_rate.vasicek import vasicek_model_simulation
from modules.models.options.interest_rate.interest_rate_options import black_cap_floor_pricing
from modules.models.interest_rate.bond_pv01 import calculate_bond_pv
from modules.risk.metrics.risk_metrics import calculate_cvar, calculate_var

def calculate_portfolio_value(bond_params, cap_params, weights, simulated_rate, yield_curve_data, valuation_date):
    """Calculate portfolio value under a simulated interest rate scenario."""
    try:
        # Ensure positive rates
        simulated_rate = max(0.0001, simulated_rate)  # Minimum rate of 0.01%
        
        # Create simulated yield curve (silently)
        simulated_yield_curve = pd.DataFrame(index=[valuation_date])
        for tenor in ['3month', '2year', '5year', '7year', '10year', '30year']:
            simulated_yield_curve[tenor] = simulated_rate * 100

        # Bond valuation
        bond_value = calculate_bond_pv(
            maturity_years=bond_params['maturity'],
            coupon_rate=bond_params['coupon'] * 100,
            notional=bond_params['notional'],
            yield_curve_data=simulated_yield_curve,
            valuation_date=valuation_date
        )
        
        # Cap/Floor valuation
        cap_floor_value = black_cap_floor_pricing(
            forward_rate=simulated_rate,
            strike_rate=cap_params['strike'],
            maturity=cap_params['maturity'],
            volatility=cap_params['volatility'],
            option_type=cap_params['type']
        ) * cap_params['notional']
        
        # Handle None or NaN values
        if bond_value is None or np.isnan(bond_value):
            bond_value = 0
        if cap_floor_value is None or np.isnan(cap_floor_value):
            cap_floor_value = 0
            
        portfolio_value = weights['bond'] * bond_value + weights['cap'] * cap_floor_value
        return portfolio_value
    
    except Exception as e:
        # Log error without printing to console
        return 0

def render_monte_carlo_var_cvar_tab():
    """Renders the Monte Carlo VaR/CVaR Analysis tab."""
    
    # Header with description
    st.header("Monte Carlo VaR/CVaR Analysis")

    # Create tabs for better organization
    setup_tab, simulation_tab, results_tab = st.tabs([
        "Portfolio Setup", "Simulation Parameters", "Results Analysis"
    ])

    with setup_tab:
        st.markdown("### Portfolio Configuration")
        
        # Bond Configuration
        with st.expander("Treasury Bond Parameters", expanded=True):
            col_bond1, col_bond2, col_bond3 = st.columns(3)
            with col_bond1:
                bond_maturity = st.number_input(
                    "Maturity (Years)",
                    min_value=1.0,
                    max_value=30.0,
                    value=10.0,
                    step=1.0,
                    help="Time to maturity of the Treasury bond"
                )
            with col_bond2:
                bond_coupon = st.number_input(
                    "Coupon (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=4.0,
                    step=0.5
                ) / 100.0
            with col_bond3:
                bond_notional = st.number_input(
                    "Notional",
                    min_value=10000,
                    max_value=1000000,
                    value=100000,
                    step=10000
                )

        # Cap/Floor Configuration
        with st.expander("Cap/Floor Parameters", expanded=True):
            col_cap1, col_cap2, col_cap3, col_cap4, col_cap5 = st.columns(5)
            with col_cap1:
                option_type = st.selectbox(
                    "Type",
                    ["cap", "floor"]
                )
            with col_cap2:
                cap_strike = st.number_input(
                    "Strike (%)",
                    min_value=0.01,
                    max_value=10.0,
                    value=6.0,
                    step=0.1
                ) / 100.0
            with col_cap3:
                cap_maturity = st.number_input(
                    "Maturity (Years)",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.5
                )
            with col_cap4:
                cap_volatility = st.number_input(
                    "Volatility (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=25.0,
                    step=1.0
                ) / 100.0
            with col_cap5:
                cap_notional = st.number_input(
                    "Notional Cap/Floor",
                    min_value=10000,
                    max_value=1000000,
                    value=50000,
                    step=10000
                )

        # Portfolio Weights with validation
        with st.expander("Portfolio Weights", expanded=True):
            col_weights1, col_weights2 = st.columns(2)
            with col_weights1:
                bond_weight = st.slider(
                    "Bond Weight (%)",
                    min_value=0,
                    max_value=100,
                    value=70,
                    step=1
                ) / 100.0
            with col_weights2:
                cap_weight = 1.0 - bond_weight
            st.metric("Cap/Floor Weight (%)", f"{cap_weight*100:.1f}%")

    with simulation_tab:
        st.markdown("### Simulation Settings")
        
        # Monte Carlo Parameters
        with st.expander("Monte Carlo Parameters", expanded=True):
            col_mc1, col_mc2, col_mc3 = st.columns(3)
            with col_mc1:
                n_sims = st.number_input(
                    "Number of Simulations",
                    min_value=1000,
                    max_value=20000,
                    value=5000,
                    step=1000
                )
            with col_mc2:
                time_horizon = st.number_input(
                    "Time Horizon (Years)",
                    min_value=1,
                    max_value=10,
                    value=1,
                    step=1
                )
            with col_mc3:
                confidence_level = st.slider(
                    "Confidence Level",
                    min_value=0.90,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    format="%.2f"
                )

        # Vasicek Model Parameters (optional)
        with st.expander("Advanced: Vasicek Model Parameters", expanded=False):
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            with col_v1:
                mean_reversion = st.number_input(
                    "Mean Reversion Speed",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    help="Speed at which rates return to the long-term mean"
                )
            with col_v2:
                long_term_mean = st.number_input(
                    "Long-term Mean Rate",
                    min_value=0.01,
                    max_value=0.1,
                    value=0.05,
                    help="Long-term average interest rate"
                )
            with col_v3:
                volatility = st.number_input(
                    "Volatility",
                    min_value=0.01,
                    max_value=0.1,
                    value=0.02,
                    help="Volatility of interest rate changes"
                )
            with col_v4:
                initial_rate = st.number_input(
                    "Initial Rate",
                    min_value=0.01,
                    max_value=0.1,
                    value=0.05,
                    help="Initial interest rate at the start of the simulation"
                )

    # Ejecutar simulación inmediatamente si hay datos
    simulation_results = None
    if 'yield_curve_data' in st.session_state and not st.session_state.yield_curve_data.empty:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # Package parameters
                bond_params = {
                    'maturity': bond_maturity,
                    'coupon': bond_coupon,
                    'notional': bond_notional
                }
                
                cap_params = {
                    'type': option_type,
                    'strike': cap_strike,
                    'maturity': cap_maturity,
                    'volatility': cap_volatility,
                    'notional': cap_notional
                }
                
                weights = {
                    'bond': bond_weight,
                    'cap': cap_weight
                }
                
                # Run Vasicek simulation
                simulated_rates = vasicek_model_simulation(
                    a=mean_reversion,
                    b=long_term_mean,
                    sigma=volatility,
                    r0=initial_rate,
                    n_simulations=n_sims,
                    time_horizon_years=time_horizon
                )
                
                valuation_date = pd.Timestamp('today')
                initial_yield_curve = st.session_state.yield_curve_data.copy()
                
                # Calculate initial portfolio value
                initial_portfolio_value = calculate_portfolio_value(
                    bond_params,
                    cap_params,
                    weights,
                    initial_rate,
                    initial_yield_curve,
                    valuation_date
                )
                
                if initial_portfolio_value > 0:
                    # Run simulation scenarios
                    portfolio_values = []
                    valid_scenarios = 0
                    
                    # Add progress bar
                    progress_bar = st.progress(0)
                    
                    for i in range(n_sims):
                        try:
                            simulated_rate = simulated_rates.iloc[-1, i]
                            
                            if not np.isnan(simulated_rate):
                                portfolio_value = calculate_portfolio_value(
                                    bond_params, cap_params, weights,
                                    simulated_rate, initial_yield_curve, valuation_date
                                )
                                
                                if portfolio_value > 0:
                                    portfolio_values.append(
                                        (portfolio_value - initial_portfolio_value) / initial_portfolio_value
                                    )
                                    valid_scenarios += 1
                            
                            progress_bar.progress((i + 1) / n_sims)
                        
                        except Exception:
                            continue

                    if len(portfolio_values) >= n_sims * 0.1:
                        portfolio_returns = np.array(portfolio_values)
                        var = calculate_var(portfolio_returns, confidence_level)
                        cvar = calculate_cvar(portfolio_returns, confidence_level)
                        
                        simulation_results = {
                            'initial_value': initial_portfolio_value,
                            'var': var,
                            'cvar': cvar,
                            'mean_return': np.mean(portfolio_values),
                            'volatility': np.std(portfolio_values),
                            'skewness': scipy.stats.skew(portfolio_values),
                            'kurtosis': scipy.stats.kurtosis(portfolio_values),
                            'worst_return': np.min(portfolio_values),
                            'best_return': np.max(portfolio_values),
                            'worst_rate': simulated_rates.iloc[-1].min(),
                            'best_rate': simulated_rates.iloc[-1].max(),
                            'portfolio_returns': portfolio_returns,
                            'confidence_level': confidence_level,
                            'valid_scenarios': valid_scenarios,
                            'total_scenarios': n_sims,
                            'average_rate': simulated_rates.iloc[-1].mean()
                        }
                
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
    else:
        st.error("Error: Yield curve not loaded. Go to the 'Yield Curve' tab and load the data first.")

    with results_tab:
        if simulation_results is not None:
            # Summary metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Value", f"${simulation_results['initial_value']:,.2f}")
            with col2:
                st.metric("VaR", f"{simulation_results['var']:.2%}")
            with col3:
                st.metric("CVaR", f"{simulation_results['cvar']:.2%}")

            # Modify tabs to include paths visualization
            dist_tab, paths_tab, stats_tab, scenarios_tab = st.tabs([
                "Distribution Analysis", "Monte Carlo Paths", "Risk Statistics", "Scenario Details"
            ])

            with dist_tab:
                # Distribution plot with improvements
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=simulation_results['portfolio_returns'],
                    nbinsx=50,
                    name="Returns"
                ))
                
                fig.add_vline(
                    x=simulation_results['var'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"VaR {simulation_results['confidence_level']:.0%}"
                )
                
                fig.add_vline(
                    x=simulation_results['cvar'],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"CVaR {simulation_results['confidence_level']:.0%}"
                )
                
                fig.update_layout(
                    template="plotly_white",
                    title={
                        'text': "Portfolio Returns Distribution",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with paths_tab:
                st.subheader("Monte Carlo Simulation Paths")
                
                # Calculate confidence bands
                path_values = []
                time_points = np.linspace(0, time_horizon, len(simulated_rates))
                
                for i in range(n_sims):
                    path = simulated_rates.iloc[:, i]
                    path_values.append(path.values)
                
                paths_array = np.array(path_values)
                mean_path = np.mean(paths_array, axis=0)
                std_path = np.std(paths_array, axis=0)
                
                # Calculate confidence bands
                upper_95 = mean_path + 1.96 * std_path
                lower_95 = mean_path - 1.96 * std_path
                upper_99 = mean_path + 2.576 * std_path
                lower_99 = mean_path - 2.576 * std_path
                
                # Create paths plot with confidence bands
                fig_paths = go.Figure()
                
                # Add sample of paths (e.g., first 100)
                for i in range(min(100, n_sims)):
                    fig_paths.add_trace(go.Scatter(
                        x=time_points,
                        y=paths_array[i],
                        mode='lines',
                        line=dict(color='rgba(100,100,100,0.1)'),
                        showlegend=False,
                        name=f'Path {i+1}'
                    ))
                
                # Add confidence bands
                fig_paths.add_trace(go.Scatter(
                    x=time_points,
                    y=upper_99,
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                    name='99% Confidence Band',
                    fill=None
                ))
                
                fig_paths.add_trace(go.Scatter(
                    x=time_points,
                    y=lower_99,
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                    fill='tonexty',
                    name='99% Confidence Band'
                ))
                
                fig_paths.add_trace(go.Scatter(
                    x=time_points,
                    y=upper_95,
                    mode='lines',
                    line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
                    name='95% Confidence Band',
                    fill=None
                ))
                
                fig_paths.add_trace(go.Scatter(
                    x=time_points,
                    y=lower_95,
                    mode='lines',
                    line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
                    fill='tonexty',
                    name='95% Confidence Band'
                ))
                
                # Add mean path
                fig_paths.add_trace(go.Scatter(
                    x=time_points,
                    y=mean_path,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Mean Path'
                ))
                
                fig_paths.update_layout(
                    title="Monte Carlo Simulation Paths with Confidence Bands",
                    xaxis_title="Time (Years)",
                    yaxis_title="Interest Rate",
                    template="plotly_white",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    yaxis=dict(
                        tickformat='.2%'
                    )
                )
                
                st.plotly_chart(fig_paths, use_container_width=True)
                
                # Add path statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Final Rate", f"{mean_path[-1]:.2%}")
                with col2:
                    st.metric("95% CI Range", f"{(upper_95[-1] - lower_95[-1])*100:.2f}%")
                with col3:
                    st.metric("99% CI Range", f"{(upper_99[-1] - lower_99[-1])*100:.2f}%")

            with stats_tab:
                # Risk metrics table using simulation_results
                metrics_df = pd.DataFrame({
                    'Metric': ['Average Return', 'Volatility', 'Skewness', 'Kurtosis'],
                    'Value': [
                        f"{simulation_results['mean_return']:.2%}",
                        f"{simulation_results['volatility']:.2%}",
                        f"{simulation_results['skewness']:.2f}",
                        f"{simulation_results['kurtosis']:.2f}"
                    ]
                })
                st.dataframe(metrics_df.set_index('Metric'))

            with scenarios_tab:
                # Show worst and best scenarios using simulation_results
                scenarios_df = pd.DataFrame({
                    'Scenario': ['Worst', 'Best'],
                    'Return': [
                        f"{simulation_results['worst_return']:.2%}",
                        f"{simulation_results['best_return']:.2%}"
                    ],
                    'Rate Level': [
                        f"{simulation_results['worst_rate']:.2%}",
                        f"{simulation_results['best_rate']:.2%}"
                    ]
                })
                st.dataframe(scenarios_df.set_index('Scenario'))
        else:
            st.info("No simulation results available. Check if yield curve is loaded and parameters are valid.")
