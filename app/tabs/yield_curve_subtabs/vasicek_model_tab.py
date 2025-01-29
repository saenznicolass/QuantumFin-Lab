import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import sys
from scipy.stats import norm

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.models.interest_rate.vasicek import (
    analyze_mean_reversion,
    calculate_negative_rate_probability,
    calculate_term_structure,
    calculate_vasicek_statistics, 
    vasicek_model_simulation,
    calculate_confidence_bands
)

def render_model_description():
    """Renders the Vasicek model description section."""
    st.markdown("## Vasicek Short Rate Model")
    
    with st.expander("Model Description", expanded=False):
        st.markdown("""
        The Vasicek model describes interest rate evolution with mean-reversion properties.
        
        ### Key Features:
        - Mean reversion behavior
        - Allows for negative rates
        - Analytically tractable
        
        ### Mathematical Formulation:
        ```
        dr(t) = a(b - r(t))dt + σdW(t)
        ```
        
        ### Parameters:
        - **a**: Speed of reversion (how quickly rates return to mean)
        - **b**: Long-term mean level (where rates tend to go)
        - **σ**: Volatility (amount of random fluctuation)
        - **W(t)**: Wiener process (random component)
        """)

def get_model_parameters():
    """Collects model parameters from user input."""
    st.markdown("### Model Configuration")
    
    tabs = st.tabs(["Model Parameters", "Simulation Settings", "Advanced Options"])
    
    with tabs[0]:
        st.markdown("#### Core Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a_vasicek = st.slider(
                "Speed of Reversion (a)",
                min_value=0.01,
                max_value=2.0,
                value=0.5,
                step=0.05,
                format="%.2f",
                help="Higher values mean faster mean reversion"
            )
            st.markdown("##### Impact of 'a'")
            st.markdown("- Low: Slow mean reversion")
            st.markdown("- High: Fast mean reversion")
        
        with col2:
            b_vasicek = st.slider(
                "Long-term Mean (b)",
                min_value=0.01,
                max_value=0.15,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="The rate level where values tend to converge"
            )
            st.markdown("##### Impact of 'b'")
            st.markdown("- Centers the mean reversion")
            st.markdown("- Historical average rate")
        
        with col3:
            sigma_vasicek = st.slider(
                "Volatility (σ)",
                min_value=0.005,
                max_value=0.05,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Higher values mean more rate fluctuation"
            )
            st.markdown("##### Impact of 'σ'")
            st.markdown("- Low: Stable rates")
            st.markdown("- High: Volatile rates")
    
    with tabs[1]:
        col4, col5 = st.columns(2)
        
        with col4:
            n_simulations = st.number_input(
                "Number of Simulations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="More simulations = better accuracy but slower"
            )
            st.markdown("##### Simulation Count Impact")
            st.markdown("- < 500: Quick preview")
            st.markdown("- > 1000: Stable results")
        
        with col5:
            time_horizon = st.number_input(
                "Time Horizon (Years)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Longer horizon shows full mean reversion effect"
            )
            st.markdown("##### Time Horizon Impact")
            st.markdown("- Short: Near-term forecasting")
            st.markdown("- Long: Mean reversion study")
    
    with tabs[2]:
        st.markdown("#### Advanced Settings")
        col6, col7 = st.columns(2)
        
        with col6:
            confidence_level = st.slider(
                "Confidence Band Level",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.2f",
                help="Confidence level for the simulation bands"
            )
        
        with col7:
            paths_to_display = st.slider(
                "Paths to Display",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Number of individual paths to show in plot"
            )
    
    return a_vasicek, b_vasicek, sigma_vasicek, n_simulations, time_horizon, confidence_level, paths_to_display

def display_simulation_results(simulated_paths, stats):
    """Displays the simulation statistics."""
    st.markdown("### Simulation Results")
    
    metrics_container = st.container()
    with metrics_container:
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric(
                "Mean Final Rate", 
                f"{stats['mean']:.2%}",
                delta=f"{stats['mean'] - stats['initial']:.2%}",
                delta_color="normal"
            )
            st.metric(
                "Median Final Rate", 
                f"{stats['median']:.2%}"
            )
        
        with col_stats2:
            st.metric(
                "Minimum Rate", 
                f"{stats['min']:.2%}",
                help="Lowest simulated rate"
            )
            st.metric(
                "Maximum Rate", 
                f"{stats['max']:.2%}",
                help="Highest simulated rate"
            )
        
        with col_stats3:
            st.metric(
                "Rate Volatility", 
                f"{stats['std']:.2%}",
                help="Standard deviation of rates"
            )
            st.metric(
                "Range", 
                f"{stats['max'] - stats['min']:.2%}",
                help="Difference between max and min rates"
            )

def plot_simulation_paths(simulated_paths, paths_to_display, time_horizon):
    """Creates and displays the simulation paths plot."""
    fig = go.Figure()
    
    # Calculate and add confidence bands
    lower_band, upper_band = calculate_confidence_bands(simulated_paths, 0.95)
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=simulated_paths.index,
        y=upper_band,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Upper 95% CI'
    ))
    fig.add_trace(go.Scatter(
        x=simulated_paths.index,
        y=lower_band,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 100, 0, 0.2)',
        line=dict(width=0),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    # Plot individual paths
    paths_to_plot = min(paths_to_display, simulated_paths.shape[1])
    for i in range(paths_to_plot):
        fig.add_trace(go.Scatter(
            x=simulated_paths.index,
            y=simulated_paths.iloc[:, i],
            mode='lines',
            name=f'Path {i+1}' if paths_to_display <= 10 else None,
            opacity=0.3,
            line=dict(width=1, color='blue'),
            showlegend=paths_to_display <= 10
        ))
    
    # Add mean path
    mean_path = simulated_paths.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=simulated_paths.index,
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(width=2, color='red'),
        showlegend=True
    ))

    fig.update_layout(
        title=f'Vasicek Model Simulation ({paths_to_display} paths, {time_horizon} years)',
        xaxis_title='Date',
        yaxis_title='Interest Rate',
        yaxis_tickformat='.2%',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_analysis_sections(a_vasicek, b_vasicek, sigma_vasicek, r0_initial, simulated_paths):
    """Displays the term structure, mean reversion, and negative rates analysis sections."""
    # Term Structure Analysis
    st.markdown("### Term Structure Analysis")
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    term_structure = calculate_term_structure(
        a_vasicek, b_vasicek, sigma_vasicek, r0_initial, maturities
    )
    
    fig_term = go.Figure()
    fig_term.add_trace(go.Scatter(
        x=maturities,
        y=term_structure,
        mode='lines+markers',
        name='Model-Implied Term Structure'
    ))
    fig_term.update_layout(
        title='Model-Implied Term Structure',
        xaxis_title='Maturity (years)',
        yaxis_title='Rate',
        yaxis_tickformat='.2%'
    )
    st.plotly_chart(fig_term, use_container_width=True)
    
    # Mean Reversion Analysis
    st.markdown("### Mean Reversion Analysis")
    reversion_metrics = analyze_mean_reversion(simulated_paths, a_vasicek, b_vasicek)
    
    col_rev1, col_rev2 = st.columns(2)
    with col_rev1:
        st.metric("Half-Life of Mean Reversion", f"{reversion_metrics['half_life']:.2f} years")
        st.metric("Adjustment Speed", f"{reversion_metrics['adjustment_speed']:.2f}")
    with col_rev2:
        st.metric("Theoretical Long-term Mean", f"{reversion_metrics['theoretical_mean']:.2%}")
        st.metric("Simulated Mean", f"{reversion_metrics['simulated_mean']:.2%}")

def display_historical_data():
    """Displays historical Federal Funds Rate data."""
    try:
        ffr_data = pd.read_csv(
            "data/IR/federal_funds_rate.csv",
            index_col=0,
            parse_dates=True
        )
        
        st.markdown("### Historical Federal Funds Rate Context")
        fig_ffr = go.Figure()
        fig_ffr.add_trace(go.Scatter(
            x=ffr_data.index,
            y=ffr_data['value']/100,
            mode='lines',
            name='Historical FFR'
        ))
        fig_ffr.update_layout(
            title='Historical Federal Funds Rate',
            xaxis_title='Date',
            yaxis_title='Rate',
            yaxis_tickformat='.2%'
        )
        st.plotly_chart(fig_ffr, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load historical Federal Funds Rate data: {e}")

def render_vasicek_model_tab():
    """Main function to render the Vasicek Model tab."""
    render_model_description()
    
    # Get parameters with new interface
    a_vasicek, b_vasicek, sigma_vasicek, n_simulations, time_horizon, confidence_level, paths_to_display = get_model_parameters()
    
    # Add a clear visual separation
    st.markdown("---")
    
    # Run simulation button with better styling and unique key
    if st.button("Run Vasicek Model Simulation", key="vasicek_sim", use_container_width=True):
        with st.spinner(" Simulating interest rate paths..."):
            # Initial rate set to long-term mean
            r0_initial = b_vasicek
            
            # Create tabs for different views
            result_tabs = st.tabs([
                "Simulation Paths", 
                "Term Structure", 
                "Mean Reversion Analysis",
                "Historical Comparison"
            ])
            
            # Run simulation
            simulated_paths = vasicek_model_simulation(
                a=a_vasicek,
                b=b_vasicek,
                sigma=sigma_vasicek,
                r0=r0_initial,
                n_simulations=n_simulations,
                time_horizon_years=time_horizon
            )
            
            # Calculate statistics
            stats = calculate_vasicek_statistics(simulated_paths)
            stats['initial'] = r0_initial
            
            # Display results in tabs
            with result_tabs[0]:
                display_simulation_results(simulated_paths, stats)
                plot_simulation_paths(simulated_paths, paths_to_display, time_horizon)
            
            with result_tabs[1]:
                display_analysis_sections(a_vasicek, b_vasicek, sigma_vasicek, r0_initial, simulated_paths)
            
            with result_tabs[2]:
                st.markdown("### Mean Reversion Analysis")
                reversion_metrics = analyze_mean_reversion(simulated_paths, a_vasicek, b_vasicek)
                
                col_rev1, col_rev2 = st.columns(2)
                with col_rev1:
                    st.metric(
                        "Half-Life of Mean Reversion",
                        f"{reversion_metrics['half_life']:.2f} years",
                        help="Time to cover half the distance to mean"
                    )
                    st.metric(
                        "Adjustment Speed",
                        f"{reversion_metrics['adjustment_speed']:.2f}",
                        help="Rate of convergence to mean"
                    )
                with col_rev2:
                    st.metric(
                        "Theoretical Mean",
                        f"{reversion_metrics['theoretical_mean']:.2%}",
                        help="Expected long-term average"
                    )
                    st.metric(
                        "Simulated Mean",
                        f"{reversion_metrics['simulated_mean']:.2%}",
                        help="Actual simulated average"
                    )
            
            with result_tabs[3]:
                display_historical_data()
