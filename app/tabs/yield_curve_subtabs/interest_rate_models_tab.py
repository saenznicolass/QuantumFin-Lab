import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from modules.models.interest_rate.hull_white import HullWhite, HWCalibrator
from modules.models.interest_rate.cir import CIR, CIRCalibrator
from modules.models.interest_rate.g2pp import G2PP, G2PPCalibrator
from typing import Tuple, List, Dict
import warnings

def render_model_description():
    """Renders the model description section."""
    st.markdown("## Interest Rate Models Comparison")
    
    with st.expander("Model Descriptions", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Hull-White Model")
            st.markdown("""
            Single-factor Gaussian model:
            ```
            dr(t) = [θ(t) - ar(t)]dt + σdW(t)
            ```
            - Mean reversion to time-dependent level
            - Allows negative rates
            - Analytically tractable
            """)
        
        with col2:
            st.markdown("### CIR Model")
            st.markdown("""
            Square-root diffusion:
            ```
            dr(t) = a(b - r(t))dt + σ√r(t)dW(t)
            ```
            - Non-negative rates
            - Mean reversion
            - State-dependent volatility
            """)
        
        with col3:
            st.markdown("### G2++ Model")
            st.markdown("""
            Two-factor Gaussian model:
            ```
            r(t) = x(t) + y(t) + φ(t)
            dx(t) = -ax(t)dt + σdW₁(t)
            dy(t) = -by(t)dt + ηdW₂(t)
            ```
            - Richer dynamics
            - Correlation between factors
            - Better fit to term structure
            """)

def get_model_parameters():
    """Collects model parameters from user input."""
    st.markdown("### Model Configuration")
    
    model_tabs = st.tabs(["Hull-White", "CIR", "G2++"])
    
    with model_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            hw_a = st.slider("Mean Reversion (a) - HW", 0.01, 1.0, 0.15, 0.01)  # Default from txt: 0.15
            hw_sigma = st.slider("Volatility (σ) - HW", 0.001, 0.1, 0.02, 0.001)  # Default from txt: 0.02
        with col2:
            hw_r0 = st.slider("Initial Rate (r₀) - HW", 0.01, 0.10, 0.03, 0.01)
    
    with model_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            cir_a = st.slider("Mean Reversion (a) - CIR", 0.01, 1.0, 0.3, 0.01)  # Default from txt: 0.3
            cir_sigma = st.slider("Volatility (σ) - CIR", 0.01, 0.5, 0.1, 0.01)  # Default from txt: 0.1
        with col2:
            cir_theta = st.slider("Long-term Mean (θ) - CIR", 0.01, 0.10, 0.05, 0.01)  # Default from txt: 0.05
            cir_r0 = st.slider("Initial Rate (r₀) - CIR", 0.01, 0.10, 0.03, 0.01)
    
    with model_tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            g2_a = st.slider("Mean Reversion 1 (a) - G2++", 0.01, 1.0, 0.1, 0.01)  # Default from txt: 0.1
            g2_sigma = st.slider("Volatility 1 (σ) - G2++", 0.001, 0.1, 0.01, 0.001)  # Default from txt: 0.01
            g2_b = st.slider("Mean Reversion 2 (b) - G2++", 0.01, 1.0, 0.2, 0.01)  # Default from txt: 0.2
        with col2:
            g2_eta = st.slider("Volatility 2 (η) - G2++", 0.001, 0.1, 0.015, 0.001)  # Default from txt: 0.015
            g2_rho = st.slider("Correlation (ρ) - G2++", -0.99, 0.99, -0.7, 0.01)  # Default from txt: -0.7
            g2_r0 = st.slider("Initial Rate (r₀) - G2++", 0.01, 0.10, 0.03, 0.01)
    
    sim_params = st.columns(3)
    with sim_params[0]:
        num_paths = st.number_input("Number of Paths", 100, 10000, 1000, 100)
    with sim_params[1]:
        time_horizon = st.number_input("Time Horizon (Years)", 1, 10, 5, 1)
    with sim_params[2]:
        num_steps = st.number_input("Time Steps", 50, 500, 100, 50)
    
    return {
        'hw': (hw_a, hw_sigma, hw_r0),
        'cir': (cir_a, cir_sigma, cir_theta, cir_r0),
        'g2': (g2_a, g2_sigma, g2_b, g2_eta, g2_rho, g2_r0),
        'sim': (num_paths, time_horizon, num_steps)
    }

def generate_market_data():
    """Generate synthetic market data for calibration."""
    # Generate data from true models with parameters from document
    true_hw = HullWhite(a=0.15, sigma=0.02, r0=0.03)
    true_cir = CIR(a=0.3, sigma=0.1, theta=0.05, r0=0.03)
    true_g2pp = G2PP(a=0.1, sigma=0.01, b=0.2, eta=0.015, rho=-0.7, r0=0.03)
    
    instruments = [{'T': t} for t in [1.0, 2.0, 3.0]]  # Bond maturities
    
    # Generate market prices with noise
    np.random.seed(42)
    market_prices = []
    for inst in instruments:
        hw_price = np.exp(-true_hw.r0 * inst['T'])
        cir_price = true_cir.bond_price(inst['T'])
        g2pp_price = np.mean(np.exp(-np.sum(true_g2pp.monte_carlo(1000, 10, inst['T']), axis=1)*inst['T']))
        # Add noise
        avg_price = (hw_price + cir_price + g2pp_price)/3
        market_prices.append(avg_price * np.random.uniform(0.98, 1.02))
    
    return instruments, market_prices

def plot_simulation_comparison(hw_paths, cir_paths, g2_paths, time_horizon):
    """Plots comparative simulation results."""
    fig = go.Figure()
    
    # Add paths for each model
    times = np.linspace(0, time_horizon, hw_paths.shape[1])
    
    # Hull-White paths
    for i in range(min(5, hw_paths.shape[0])):
        fig.add_trace(go.Scatter(
            x=times, y=hw_paths[i], mode='lines',
            line=dict(width=1, color='blue', dash='dash'),
            name=f'HW Path {i+1}' if i == 0 else None,
            showlegend=i == 0,
            opacity=0.3
        ))
    
    # CIR paths
    for i in range(min(5, cir_paths.shape[0])):
        fig.add_trace(go.Scatter(
            x=times, y=cir_paths[i], mode='lines',
            line=dict(width=1, color='red', dash='dot'),
            name=f'CIR Path {i+1}' if i == 0 else None,
            showlegend=i == 0,
            opacity=0.3
        ))
    
    # G2++ paths
    for i in range(min(5, g2_paths.shape[0])):
        fig.add_trace(go.Scatter(
            x=times, y=g2_paths[i], mode='lines',
            line=dict(width=1, color='green'),
            name=f'G2++ Path {i+1}' if i == 0 else None,
            showlegend=i == 0,
            opacity=0.3
        ))
    
    # Add mean paths
    fig.add_trace(go.Scatter(
        x=times, y=hw_paths.mean(axis=0),
        mode='lines', line=dict(width=2, color='blue'),
        name='HW Mean'
    ))
    fig.add_trace(go.Scatter(
        x=times, y=cir_paths.mean(axis=0),
        mode='lines', line=dict(width=2, color='red'),
        name='CIR Mean'
    ))
    fig.add_trace(go.Scatter(
        x=times, y=g2_paths.mean(axis=0),
        mode='lines', line=dict(width=2, color='green'),
        name='G2++ Mean'
    ))
    
    fig.update_layout(
        title='Interest Rate Model Comparison',
        xaxis_title='Time (years)',
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

def plot_separate_simulations(hw_paths, cir_paths, g2_paths, time_horizon, confidence_level=0.95):
    """Creates three separate plots for each model with confidence bands."""
    times = np.linspace(0, time_horizon, hw_paths.shape[1])
    
    # Create tabs for different visualizations
    plot_tabs = st.tabs(["Hull-White", "CIR", "G2++"])
    
    # Color schemes for each model - Updated colors and opacities
    hw_colors = {
        'main': 'rgb(31, 119, 180)', 
        'band': 'rgba(31, 119, 180, 0.15)',  # Lighter band
        'paths': 'rgba(0, 0, 0, 0.2)'  # Black paths with 0.2 opacity
    }
    cir_colors = {
        'main': 'rgb(255, 127, 14)', 
        'band': 'rgba(255, 127, 14, 0.15)', 
        'paths': 'rgba(0, 0, 0, 0.2)'
    }
    g2_colors = {
        'main': 'rgb(44, 160, 44)', 
        'band': 'rgba(44, 160, 44, 0.15)', 
        'paths': 'rgba(0, 0, 0, 0.2)'
    }
    
    def add_confidence_bands(fig, paths, colors):
        alpha = (1 - confidence_level) / 2
        mean = np.mean(paths, axis=0)
        lower = np.percentile(paths, alpha * 100, axis=0)
        upper = np.percentile(paths, (1 - alpha) * 100, axis=0)
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=times, y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=times, y=lower,
            mode='lines',
            fill='tonexty',
            fillcolor=colors['band'],
            line=dict(width=0),
            name=f'{confidence_level:.0%} Confidence Interval'
        ))
        
        # Add mean path - Made thicker and more visible
        fig.add_trace(go.Scatter(
            x=times, y=mean,
            mode='lines',
            line=dict(width=4, color=colors['main']),  # Increased width
            name='Mean Path'
        ))
        
        return fig
    
    # Plot each model with confidence bands
    with plot_tabs[0]:
        fig_hw = go.Figure()
        # Increased number of visible paths to 25
        for i in range(min(25, hw_paths.shape[0])):
            fig_hw.add_trace(go.Scatter(
                x=times, y=hw_paths[i],
                mode='lines',
                line=dict(width=1, color=hw_colors['paths']),  # Black paths
                name=f'Path {i+1}',
                showlegend=False
            ))
        fig_hw = add_confidence_bands(fig_hw, hw_paths, hw_colors)
        fig_hw.update_layout(
            title='Hull-White Model Simulations',
            xaxis_title='Time (years)',
            yaxis_title='Interest Rate',
            yaxis_tickformat='.2%',
            template='plotly_white',
            hovermode='x unified',
            plot_bgcolor='white'  # Ensure white background
        )
        st.plotly_chart(fig_hw, use_container_width=True)
    
    with plot_tabs[1]:
        fig_cir = go.Figure()
        for i in range(min(25, cir_paths.shape[0])):  # Also increased to 25 paths
            fig_cir.add_trace(go.Scatter(
                x=times, y=cir_paths[i],
                mode='lines',
                line=dict(width=1, color=cir_colors['paths']),
                name=f'Path {i+1}',
                showlegend=False
            ))
        fig_cir = add_confidence_bands(fig_cir, cir_paths, cir_colors)
        fig_cir.update_layout(
            title='CIR Model Simulations',
            xaxis_title='Time (years)',
            yaxis_title='Interest Rate',
            yaxis_tickformat='.2%',
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_cir, use_container_width=True)
    
    with plot_tabs[2]:
        fig_g2 = go.Figure()
        for i in range(min(25, g2_paths.shape[0])):  # Also increased to 25 paths
            fig_g2.add_trace(go.Scatter(
                x=times, y=g2_paths[i],
                mode='lines',
                line=dict(width=1, color=g2_colors['paths']),
                name=f'Path {i+1}',
                showlegend=False
            ))
        fig_g2 = add_confidence_bands(fig_g2, g2_paths, g2_colors)
        fig_g2.update_layout(
            title='G2++ Model Simulations',
            xaxis_title='Time (years)',
            yaxis_title='Interest Rate',
            yaxis_tickformat='.2%',
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_g2, use_container_width=True)

def calibrate_with_progress(instruments: List[Dict], market_prices: List[float]) -> Tuple:
    """Calibrate models with progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Hull-White calibration
        status_text.text("Calibrating Hull-White model...")
        progress_bar.progress(10)
        hw_calibrator = HWCalibrator(instruments, market_prices)
        hw_model, hw_rmse = hw_calibrator.calibrate()
        progress_bar.progress(40)
        
        # CIR calibration
        status_text.text("Calibrating CIR model...")
        cir_calibrator = CIRCalibrator(instruments, market_prices)
        cir_model, cir_rmse = cir_calibrator.calibrate()
        progress_bar.progress(70)
        
        # G2++ calibration
        status_text.text("Calibrating G2++ model...")
        g2_calibrator = G2PPCalibrator(instruments, market_prices)
        g2_model, g2_rmse = g2_calibrator.calibrate()
        progress_bar.progress(100)
        status_text.text("Calibration complete!")
        
        return hw_model, hw_rmse, cir_model, cir_rmse, g2_model, g2_rmse
    except Exception as e:
        status_text.error(f"Calibration error: {str(e)}")
        return None
    finally:
        # Clean up progress elements after 2 seconds
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

def load_real_market_data():
    """Load real market data from files."""
    try:
        # Load Treasury yields data
        yields_data = pd.read_csv("data/IR/treasury_yields.csv", index_col=0, parse_dates=True)
        # Load Federal Funds Rate data
        ffr_data = pd.read_csv("data/IR/federal_funds_rate.csv", index_col=0, parse_dates=True)
        
        # Get most recent data point
        latest_yields = yields_data.iloc[-1]
        r0 = ffr_data['value'].iloc[-1] / 100  # Convert from percentage
        
        # Create mapping between years and column names
        maturity_map = {
            0.25: '3month',
            2: '2year',
            5: '5year',
            7: '7year',
            10: '10year',
            30: '30year'
        }
        
        # Create instruments using available maturities
        instruments = [{'T': t} for t in maturity_map.keys()]
        
        # Convert yields to prices using P = exp(-rT)
        market_prices = [
            np.exp(-latest_yields[maturity_map[t]]/100 * t) 
            for t in maturity_map.keys()
        ]
        
        st.info(f"Loaded market data - Federal Funds Rate: {r0:.2%}")
        st.info(f"Latest yields: {', '.join([f'{k}: {latest_yields[v]:.2f}%' for k,v in maturity_map.items()])}")
        
        return instruments, market_prices, r0
        
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return None, None, None

def render_interest_rate_models_tab():
    """Main function to render the Interest Rate Models tab."""
    render_model_description()
    
    # Add data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Synthetic Data", "Real Market Data"],
        help="Choose between synthetic data or real market data from US Treasury yields"
    )
    
    params = get_model_parameters()
    
    if st.button("Run Interest Rate Models Simulation", key="ir_models_sim", use_container_width=True):
        try:
            with st.spinner("Preparing simulation..."):
                # Get data based on selection
                if data_source == "Synthetic Data":
                    instruments, market_prices = generate_market_data()
                    st.info("Using synthetic market data for calibration")
                else:
                    instruments, market_prices, r0 = load_real_market_data()
                    if instruments is None:
                        st.error("Failed to load real market data")
                        return
                    st.info("Using real US Treasury market data for calibration")
                    # Update initial rates for models with real FFR
                    params['hw'] = (params['hw'][0], params['hw'][1], r0)
                    params['cir'] = (params['cir'][0], params['cir'][1], params['cir'][2], r0)
                    params['g2'] = (params['g2'][0], params['g2'][1], params['g2'][2], 
                                  params['g2'][3], params['g2'][4], r0)
                
                # Display market data
                st.markdown("### Market Data")
                market_df = pd.DataFrame({
                    'Maturity (Years)': [inst['T'] for inst in instruments],
                    'Price': market_prices,
                    'Implied Yield': [-np.log(p)/t for p, t in zip(market_prices, [inst['T'] for inst in instruments])]
                })
                st.dataframe(market_df.style.format({
                    'Price': '{:.4f}',
                    'Implied Yield': '{:.2%}'
                }))
                
                # Continue with existing calibration and simulation code
                calibration_results = calibrate_with_progress(instruments, market_prices)
                
                if calibration_results is not None:
                    hw_model, hw_rmse, cir_model, cir_rmse, g2_model, g2_rmse = calibration_results
                    
                    # Display calibration results
                    cal_results = pd.DataFrame({
                        'Model': ['Hull-White', 'CIR', 'G2++'],
                        'RMSE': [hw_rmse, cir_rmse, g2_rmse],
                        'Parameters': [
                            f"a={hw_model.a:.4f}, σ={hw_model.sigma:.4f}",
                            f"a={cir_model.a:.4f}, σ={cir_model.sigma:.4f}, θ={cir_model.theta:.4f}",
                            f"a={g2_model.a:.4f}, σ={g2_model.sigma:.4f}, b={g2_model.b:.4f}, η={g2_model.eta:.4f}, ρ={g2_model.rho:.4f}"
                        ]
                    })
                    st.dataframe(cal_results, use_container_width=True)
                    
                    # Run simulations with progress
                    st.markdown("### Simulation Progress")
                    sim_progress = st.progress(0)
                    sim_status = st.empty()
                    
                    num_paths, time_horizon, num_steps = params['sim']
                    
                    sim_status.text("Simulating Hull-White paths...")
                    hw_paths = hw_model.monte_carlo(num_paths, num_steps, time_horizon)
                    sim_progress.progress(33)
                    
                    sim_status.text("Simulating CIR paths...")
                    cir_paths = cir_model.monte_carlo(num_paths, num_steps, time_horizon)
                    sim_progress.progress(66)
                    
                    sim_status.text("Simulating G2++ paths...")
                    g2_paths = g2_model.monte_carlo(num_paths, num_steps, time_horizon)
                    sim_progress.progress(100)
                    sim_status.text("Simulations complete!")
                    
                    # Clean up progress elements after 2 seconds
                    time.sleep(2)
                    sim_progress.empty()
                    sim_status.empty()
                    
                    # Display plots and statistics
                    st.markdown("### Model Simulations")
                    plot_separate_simulations(hw_paths, cir_paths, g2_paths, time_horizon, confidence_level=0.95)
                    
                    # Additional statistics
                    st.markdown("### Model Statistics")
                    stats_df = pd.DataFrame({
                        'Statistic': ['Mean', 'Std Dev', 'Min', 'Max'],
                        'Hull-White': [
                            f"{hw_paths.mean():.2%}",
                            f"{hw_paths.std():.2%}",
                            f"{hw_paths.min():.2%}",
                            f"{hw_paths.max():.2%}"
                        ],
                        'CIR': [
                            f"{cir_paths.mean():.2%}",
                            f"{cir_paths.std():.2%}",
                            f"{cir_paths.min():.2%}",
                            f"{cir_paths.max():.2%}"
                        ],
                        'G2++': [
                            f"{g2_paths.mean():.2%}",
                            f"{g2_paths.std():.2%}",
                            f"{g2_paths.min():.2%}",
                            f"{g2_paths.max():.2%}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
