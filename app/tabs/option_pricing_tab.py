# app/tabs/option_pricing_tab.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.models.options.black_scholes import black_scholes
from modules.models.options.binomial import binomial_option_pricing
from modules.models.options.merton import merton_option_pricing
from modules.models.options.heston import (
    heston_option_pricing,
    heston_option_pricing_lambda
)
from modules.models.options.sabr import sabr_option_pricing
from modules.models.options.exotic.barrier import barrier_option_pricing
from modules.models.options.exotic.asian import (
    asian_option_pricing,
    geometric_asian_option_pricing
)
from modules.models.options.exotic.digital import digital_option_pricing
from modules.risk.greeks.greeks_first_sec import calculate_greeks

from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf

# Streamlit caching decorator for the basic Black-Scholes
@st.cache_data
def cached_black_scholes(S, K, T, r, sigma, option_type='call'):
    return black_scholes(S, K, T, r, sigma, option_type)


def generate_option_price_heatmap(K, T, r, spot_range, vol_range, option_type='call'):
    """
    Generate data for an option price heatmap.
    """
    spot_prices = np.linspace(spot_range[0], spot_range[1], 50)
    volatilities = np.linspace(vol_range[0], vol_range[1], 50)
    spot_grid, vol_grid = np.meshgrid(spot_prices, volatilities)
    heatmap_data = np.zeros_like(spot_grid)

    for i in range(spot_grid.shape[0]):
        for j in range(spot_grid.shape[1]):
            S = spot_grid[i, j]
            sigma = vol_grid[i, j]
            heatmap_data[i, j] = black_scholes(S, K, T, r, sigma, option_type)

    return spot_prices, volatilities, heatmap_data


def calculate_implied_volatility(option_price, S, K, T, r, option_type='call', tolerance=1e-6, max_iterations=100):
    """
    Calculate implied volatility using the bisection method.
    """
    low = 0.01
    high = 2.0  # A reasonable upper bound for volatility
    for _ in range(max_iterations):
        mid = (low + high) / 2
        price_diff = black_scholes(S, K, T, r, mid, option_type) - option_price
        if abs(price_diff) < tolerance:
            return mid
        elif price_diff > 0:
            high = mid
        else:
            low = mid
    return np.nan  # Return NaN if no convergence


def binomial_model_convergence_data(S, K, T, r, sigma, max_steps=100, step_size=10):
    steps_range = np.arange(step_size, max_steps + 1, step_size)
    binomial_prices = [
        binomial_option_pricing(S, K, T, r, sigma, n, 'call')
        for n in steps_range
    ]
    bs_price = black_scholes(S, K, T, r, sigma, 'call')
    return steps_range, binomial_prices, bs_price


def render_option_pricing_tab():
    """
    Renders the entire Option Pricing tab in Streamlit.
    Incorporates Black-Scholes, Greeks, heatmaps, 3D surfaces, implied vol, etc.
    """
    st.header("Black-Scholes Option Pricing and Greeks")

    # --- Global Input Parameters from Sidebar (example) ---
    S_global = st.session_state.get('S_global', 100.0)
    K_global = st.session_state.get('K_global', 100.0)
    T_global = st.session_state.get('T_global', 1.0)
    sigma_global = st.session_state.get('sigma_global', 0.2)
    r_global = st.session_state.get('r_global', 0.05)

    # Option Prices
    call_price = cached_black_scholes(S_global, K_global, T_global, r_global, sigma_global, 'call')
    put_price = cached_black_scholes(S_global, K_global, T_global, r_global, sigma_global, 'put')

    st.write("### Option Prices")
    st.write(f"**Call Option Price**: ${call_price:.2f}")
    st.write(f"**Put Option Price**: ${put_price:.2f}")

    # Greeks
    greeks = calculate_greeks(S_global, K_global, T_global, r_global, sigma_global)
    st.write("### Option Greeks")
    greeks_df = pd.DataFrame(list(greeks.items()), columns=['Greek', 'Value'])
    st.dataframe(greeks_df.style.format({'Value': '{:.6f}'}))

    # --- Option Price Heatmaps ---
    st.write("### Option Price Heatmaps with Interactive Ranges")
    spot_min, spot_max = st.slider(
        "Spot Price Range",
        min_value=0.0,
        max_value=2 * K_global,
        value=(K_global * 0.5, K_global * 1.5)
    )
    vol_min, vol_max = st.slider(
        "Volatility Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.01, 1.0)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Call Option Heatmap")
        spot_prices_call, volatilities_call, heatmap_data_call = generate_option_price_heatmap(
            K_global, T_global, r_global, (spot_min, spot_max), (vol_min, vol_max), option_type='call'
        )
        text_data_call = np.round(heatmap_data_call, 2).astype(str)
        fig_call = go.Figure(data=go.Heatmap(
            x=spot_prices_call,
            y=volatilities_call,
            z=heatmap_data_call,
            text=text_data_call,
            texttemplate="%{text}",
            colorscale='YlGnBu',
            colorbar=dict(title='Option Price')
        ))
        fig_call.update_layout(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            title='Call Option Price Heatmap'
        )
        st.plotly_chart(fig_call, use_container_width=True)

    with col2:
        st.write("#### Put Option Heatmap")
        spot_prices_put, volatilities_put, heatmap_data_put = generate_option_price_heatmap(
            K_global, T_global, r_global, (spot_min, spot_max), (vol_min, vol_max), option_type='put'
        )
        text_data_put = np.round(heatmap_data_put, 2).astype(str)
        fig_put = go.Figure(data=go.Heatmap(
            x=spot_prices_put,
            y=volatilities_put,
            z=heatmap_data_put,
            text=text_data_put,
            texttemplate="%{text}",
            colorscale='YlGnBu',
            colorbar=dict(title='Option Price')
        ))
        fig_put.update_layout(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            title='Put Option Price Heatmap'
        )
        st.plotly_chart(fig_put, use_container_width=True)

    # --- 3D Surface Plots ---
    st.write("### 3D Surface Plot of Option Price")
    col3, col4 = st.columns(2)

    with col3:
        st.write("#### Call Option 3D Surface")
        spot_prices_3d = np.linspace(spot_min, spot_max, 50)
        volatilities_3d = np.linspace(vol_min, vol_max, 50)
        spot_grid_3d, vol_grid_3d = np.meshgrid(spot_prices_3d, volatilities_3d)
        option_prices_3d_call = np.vectorize(black_scholes)(
            spot_grid_3d, K_global, T_global, r_global, vol_grid_3d, 'call'
        )
        fig_3d_call = go.Figure(data=[go.Surface(x=spot_grid_3d, y=vol_grid_3d, z=option_prices_3d_call)])
        fig_3d_call.update_layout(
            title='Call Option Price Surface',
            scene=dict(
                xaxis_title='Spot Price',
                yaxis_title='Volatility',
                zaxis_title='Option Price'
            )
        )
        st.plotly_chart(fig_3d_call, use_container_width=True)

    with col4:
        st.write("#### Put Option 3D Surface")
        option_prices_3d_put = np.vectorize(black_scholes)(
            spot_grid_3d, K_global, T_global, r_global, vol_grid_3d, 'put'
        )
        fig_3d_put = go.Figure(data=[go.Surface(x=spot_grid_3d, y=vol_grid_3d, z=option_prices_3d_put)])
        fig_3d_put.update_layout(
            title='Put Option Price Surface',
            scene=dict(
                xaxis_title='Spot Price',
                yaxis_title='Volatility',
                zaxis_title='Option Price'
            )
        )
        st.plotly_chart(fig_3d_put, use_container_width=True)

    # --- Greeks vs. Spot Price ---
    st.write("### Greeks vs Spot Price")
    spot_range_values = np.linspace(K_global * 0.5, K_global * 1.5, 100)

    delta_calls = []
    delta_puts = []
    gamma_values = []
    vega_values = []
    theta_calls = []
    theta_puts = []
    rho_calls = []
    rho_puts = []
    vomma_values = []
    vanna_values = []
    speed_values = []
    zomma_values = []

    for S_spot in spot_range_values:
        g = calculate_greeks(S_spot, K_global, T_global, r_global, sigma_global)
        delta_calls.append(g['Delta Call'])
        delta_puts.append(g['Delta Put'])
        gamma_values.append(g['Gamma'])
        vega_values.append(g['Vega'])
        theta_calls.append(g['Theta Call'])
        theta_puts.append(g['Theta Put'])
        rho_calls.append(g['Rho Call'])
        rho_puts.append(g['Rho Put'])
        vomma_values.append(g['Vomma'])
        vanna_values.append(g['Vanna'])
        speed_values.append(g['Speed'])
        zomma_values.append(g['Zomma'])

    greek_plot_tabs = st.tabs(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vomma', 'Vanna', 'Speed', 'Zomma'])

    with greek_plot_tabs[0]:
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(x=spot_range_values, y=delta_calls, mode='lines', name='Delta Call'))
        fig_delta.add_trace(go.Scatter(x=spot_range_values, y=delta_puts, mode='lines', name='Delta Put'))
        fig_delta.update_layout(title='Delta vs Spot Price', xaxis_title='Spot Price', yaxis_title='Delta')
        st.plotly_chart(fig_delta, use_container_width=True)

    with greek_plot_tabs[1]:
        fig_gamma = go.Figure()
        fig_gamma.add_trace(go.Scatter(x=spot_range_values, y=gamma_values, mode='lines', name='Gamma'))
        fig_gamma.update_layout(title='Gamma vs Spot Price', xaxis_title='Spot Price', yaxis_title='Gamma')
        st.plotly_chart(fig_gamma, use_container_width=True)

    with greek_plot_tabs[2]:
        fig_vega = go.Figure()
        fig_vega.add_trace(go.Scatter(x=spot_range_values, y=vega_values, mode='lines', name='Vega'))
        fig_vega.update_layout(title='Vega vs Spot Price', xaxis_title='Spot Price', yaxis_title='Vega')
        st.plotly_chart(fig_vega, use_container_width=True)

    with greek_plot_tabs[3]:
        fig_theta = go.Figure()
        fig_theta.add_trace(go.Scatter(x=spot_range_values, y=theta_calls, mode='lines', name='Theta Call'))
        fig_theta.add_trace(go.Scatter(x=spot_range_values, y=theta_puts, mode='lines', name='Theta Put'))
        fig_theta.update_layout(title='Theta vs Spot Price', xaxis_title='Spot Price', yaxis_title='Theta')
        st.plotly_chart(fig_theta, use_container_width=True)

    with greek_plot_tabs[4]:
        fig_rho = go.Figure()
        fig_rho.add_trace(go.Scatter(x=spot_range_values, y=rho_calls, mode='lines', name='Rho Call'))
        fig_rho.add_trace(go.Scatter(x=spot_range_values, y=rho_puts, mode='lines', name='Rho Put'))
        fig_rho.update_layout(title='Rho vs Spot Price', xaxis_title='Spot Price', yaxis_title='Rho')
        st.plotly_chart(fig_rho, use_container_width=True)

    with greek_plot_tabs[5]:
        fig_vomma = go.Figure()
        fig_vomma.add_trace(go.Scatter(x=spot_range_values, y=vomma_values, mode='lines', name='Vomma'))
        fig_vomma.update_layout(title='Vomma vs Spot Price', xaxis_title='Spot Price', yaxis_title='Vomma')
        st.plotly_chart(fig_vomma, use_container_width=True)

    with greek_plot_tabs[6]:
        fig_vanna = go.Figure()
        fig_vanna.add_trace(go.Scatter(x=spot_range_values, y=vanna_values, mode='lines', name='Vanna'))
        fig_vanna.update_layout(title='Vanna vs Spot Price', xaxis_title='Spot Price', yaxis_title='Vanna')
        st.plotly_chart(fig_vanna, use_container_width=True)

    with greek_plot_tabs[7]:
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(x=spot_range_values, y=speed_values, mode='lines', name='Speed'))
        fig_speed.update_layout(title='Speed vs Spot Price', xaxis_title='Spot Price', yaxis_title='Speed')
        st.plotly_chart(fig_speed, use_container_width=True)

    with greek_plot_tabs[8]:
        fig_zomma = go.Figure()
        fig_zomma.add_trace(go.Scatter(x=spot_range_values, y=zomma_values, mode='lines', name='Zomma'))
        fig_zomma.update_layout(title='Zomma vs Spot Price', xaxis_title='Spot Price', yaxis_title='Zomma')
        st.plotly_chart(fig_zomma, use_container_width=True)

    # --- Implied Volatility ---
    st.write("### Implied Volatility Calculation")
    option_type_iv = st.selectbox("Option Type for Implied Volatility", options=['call', 'put'])
    market_price = st.number_input("Market Price of Option", min_value=0.01, value=10.0, step=0.01)

    if st.button("Calculate Implied Volatility"):
        implied_vol = calculate_implied_volatility(
            market_price,
            S_global,
            K_global,
            T_global,
            r_global,
            option_type_iv
        )
        if implied_vol is not None and not np.isnan(implied_vol):
            st.success(f"Implied Volatility: {implied_vol:.4f}")
        else:
            st.error("Could not calculate implied volatility. Check inputs or convergence.")

    # --- Binomial Model Convergence ---
    st.write("### Binomial Model Convergence to Black-Scholes")
    max_steps_input = st.slider("Maximum Steps for Binomial Model", min_value=10, max_value=500, value=100, step=10)
    steps_range, binomial_prices, bs_price = binomial_model_convergence_data(
        S_global, K_global, T_global, r_global, sigma_global, max_steps=max_steps_input
    )
    fig_convergence = go.Figure()
    fig_convergence.add_trace(go.Scatter(
        x=steps_range,
        y=binomial_prices,
        mode='lines+markers',
        name='Binomial Price'
    ))
    fig_convergence.add_hline(
        y=bs_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f'Black-Scholes Price: {bs_price:.2f}'
    )
    fig_convergence.update_layout(
        title='Binomial Model Convergence',
        xaxis_title='Number of Steps',
        yaxis_title='Option Price'
    )
    st.plotly_chart(fig_convergence, use_container_width=True)

    st.info("Note: Heston and SABR model calibration is a placeholder and requires further implementation. Exotic options (barrier, Asian, digital) are included only as simplified examples.")
