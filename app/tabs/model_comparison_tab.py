# app/tabs/model_comparison_tab.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Import from modules
from modules.models.options.black_scholes import black_scholes
from modules.models.options.binomial import binomial_option_pricing
from modules.models.options.merton import merton_option_pricing
from modules.models.options.heston import (
    heston_option_pricing,
    heston_option_pricing_lambda
)
from modules.models.options.sabr import sabr_option_pricing
from modules.models.options.exotic.barrier import barrier_option_pricing
from modules.models.options.exotic.asian import asian_option_pricing
from modules.models.options.exotic.digital import digital_option_pricing

def render_model_comparison_tab():
    """
    Renders the Option Pricing Model Comparison tab in Streamlit.
    Compares various models: Black-Scholes, Binomial, Merton, Heston, SABR, etc.
    """
    st.header("Option Pricing Model Comparison")

    # Example: Additional sidebar parameters (in practice, you'd unify sidebar usage)
    with st.sidebar:
        st.subheader("Input Parameters for Model Comparison")
        N = st.number_input(
            "Number of Steps (Binomial Model)",
            min_value=1,
            value=100,
            step=1,
            help="Number of steps in the Binomial model."
        )
        jump_intensity_comp = st.number_input(
            "Jump Intensity (Merton Model)",
            min_value=0.0,
            value=0.1,
            step=0.01
        )
        jump_mean_comp = st.number_input(
            "Jump Size Mean (Merton Model)",
            min_value=-1.0,
            value=0.0,
            step=0.01
        )
        jump_std_comp = st.number_input(
            "Jump Size Std Dev (Merton Model)",
            min_value=0.0,
            value=0.2,
            step=0.01
        )
        kappa = st.number_input("Kappa (Heston Models)", min_value=0.0, value=2.0, step=0.1)
        theta = st.number_input("Theta (Heston Models)", min_value=0.0, value=0.2, step=0.01)
        rho = st.number_input("Rho (Heston Models)", min_value=-1.0, max_value=1.0, value=0.5, step=0.01)
        v0 = st.number_input("V0 (Heston Models)", min_value=0.0, value=0.2, step=0.01)
        sigma_v = st.number_input("Vol of Vol σ_v (Heston)", min_value=0.0, value=0.3, step=0.01)
        lambd = st.number_input("Lambda (Heston 2 Model)", min_value=0.0, value=0.1, step=0.01)
        alpha_sabr = st.number_input("Alpha (SABR Model)", min_value=0.0, value=0.2, step=0.01)
        beta_sabr = st.number_input("Beta (SABR Model)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        rho_sabr = st.number_input("Rho (SABR Model)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        nu_sabr = st.number_input("Nu (SABR Model)", min_value=0.0, value=0.3, step=0.01)

    # Retrieve global parameters (could come from session_state or any config)
    S_global = st.session_state.get('S_global', 100.0)
    K_global = st.session_state.get('K_global', 100.0)
    T_global = st.session_state.get('T_global', 1.0)
    sigma_global = st.session_state.get('sigma_global', 0.2)
    r_global = st.session_state.get('r_global', 0.05)

    models = ['Black-Scholes', 'Binomial', 'Merton', 'Heston', 'Heston 2', 'SABR', 'Barrier', 'Asian']
    option_prices_dict = {
        'Model': models,
        'Call Price': [],
        'Put Price': []
    }

    # Compare each model
    for model in models:
        if model == 'Black-Scholes':
            call_p = black_scholes(S_global, K_global, T_global, r_global, sigma_global, 'call')
            put_p  = black_scholes(S_global, K_global, T_global, r_global, sigma_global, 'put')
        elif model == 'Binomial':
            call_p = binomial_option_pricing(S_global, K_global, T_global, r_global, sigma_global, N, 'call')
            put_p  = binomial_option_pricing(S_global, K_global, T_global, r_global, sigma_global, N, 'put')
        elif model == 'Merton':
            call_p = merton_option_pricing(
                S_global, K_global, T_global, r_global, sigma_global,
                jump_intensity_comp, jump_mean_comp, jump_std_comp, 'call'
            )
            put_p  = merton_option_pricing(
                S_global, K_global, T_global, r_global, sigma_global,
                jump_intensity_comp, jump_mean_comp, jump_std_comp, 'put'
            )
        elif model == 'Heston':
            call_p = heston_option_pricing(S_global, K_global, T_global, r_global, kappa, theta, sigma_v, rho, v0, 'call')
            put_p  = heston_option_pricing(S_global, K_global, T_global, r_global, kappa, theta, sigma_v, rho, v0, 'put')
        elif model == 'Heston 2':
            call_p = heston_option_pricing_lambda(S_global, K_global, T_global, r_global, kappa, theta, sigma_v, rho, v0, lambd, 'call')
            put_p  = heston_option_pricing_lambda(S_global, K_global, T_global, r_global, kappa, theta, sigma_v, rho, v0, lambd, 'put')
        elif model == 'SABR':
            call_p = sabr_option_pricing(S_global, K_global, T_global, alpha_sabr, beta_sabr, rho_sabr, nu_sabr, r_global, 'call')
            put_p  = sabr_option_pricing(S_global, K_global, T_global, alpha_sabr, beta_sabr, rho_sabr, nu_sabr, r_global, 'put')
        elif model == 'Barrier':
            # Example: up-and-out barrier at 1.1 * S
            H = S_global * 1.1
            call_p = barrier_option_pricing(S_global, K_global, H, T_global, r_global, sigma_global, 'call', barrier_type='up-and-out')
            put_p  = barrier_option_pricing(S_global, K_global, H, T_global, r_global, sigma_global, 'put',  barrier_type='up-and-out')
        elif model == 'Asian':
            call_p = asian_option_pricing(S_global, K_global, T_global, r_global, sigma_global, 'call')
            put_p  = asian_option_pricing(S_global, K_global, T_global, r_global, sigma_global, 'put')
        else:
            call_p = put_p = np.nan

        option_prices_dict['Call Price'].append(call_p)
        option_prices_dict['Put Price'].append(put_p)

    # Display table of model results
    comp_data = pd.DataFrame(option_prices_dict)
    st.write("### Option Prices by Model")
    st.dataframe(
        comp_data.style.format({'Call Price': '${:.2f}', 'Put Price': '${:.2f}'}),
        use_container_width=True
    )

    # Comparative Visualization
    st.write("### Comparative Visualization of Option Pricing Models")
    option_type_comp = st.selectbox("Option Type for Comparison", options=['call', 'put'])
    selected_models = st.multiselect("Select Models to Compare", options=models, default=models)

    spot_prices_comp = np.linspace(K_global * 0.5, K_global * 1.5, 50)
    option_prices_models = {}

    # Helper to get price from each model, for each spot S
    def get_model_price(model, S_spot, option_type):
        if model == 'Black-Scholes':
            return black_scholes(S_spot, K_global, T_global, r_global, sigma_global, option_type)
        elif model == 'Binomial':
            return binomial_option_pricing(S_spot, K_global, T_global, r_global, sigma_global, N, option_type)
        elif model == 'Merton':
            return merton_option_pricing(
                S_spot, K_global, T_global, r_global, sigma_global,
                jump_intensity_comp, jump_mean_comp, jump_std_comp, option_type
            )
        elif model == 'Heston':
            return heston_option_pricing(S_spot, K_global, T_global, r_global, kappa, theta, sigma_v, rho, v0, option_type)
        elif model == 'Heston 2':
            return heston_option_pricing_lambda(S_spot, K_global, T_global, r_global, kappa, theta, sigma_v, rho, v0, lambd, option_type)
        elif model == 'SABR':
            return sabr_option_pricing(S_spot, K_global, T_global, alpha_sabr, beta_sabr, rho_sabr, nu_sabr, r_global, option_type)
        elif model == 'Barrier':
            return barrier_option_pricing(S_spot, K_global, S_spot * 1.1, T_global, r_global, sigma_global, option_type, barrier_type='up-and-out')
        elif model == 'Asian':
            return asian_option_pricing(S_spot, K_global, T_global, r_global, sigma_global, option_type)
        else:
            return np.nan

    # Build dict of prices for selected models
    for model in selected_models:
        prices = []
        for S_spot in spot_prices_comp:
            price = get_model_price(model, S_spot, option_type_comp)
            prices.append(price)
        option_prices_models[model] = prices

    col5, col6 = st.columns(2)

    with col5:
        st.write(f"#### {option_type_comp.capitalize()} Option Prices by Model")
        fig_comp_models = go.Figure()
        for model in selected_models:
            fig_comp_models.add_trace(
                go.Scatter(
                    x=spot_prices_comp,
                    y=option_prices_models[model],
                    mode='lines',
                    name=model
                )
            )
        fig_comp_models.update_layout(
            title=f'{option_type_comp.capitalize()} Option Prices by Model',
            xaxis_title='Spot Price',
            yaxis_title='Option Price'
        )
        st.plotly_chart(fig_comp_models, use_container_width=True)

    # For convenience, also show the "other" option type
    option_type_comp_other = 'put' if option_type_comp == 'call' else 'call'
    option_prices_models_other = {}
    for model in selected_models:
        prices = []
        for S_spot in spot_prices_comp:
            price = get_model_price(model, S_spot, option_type_comp_other)
            prices.append(price)
        option_prices_models_other[model] = prices

    with col6:
        st.write(f"#### {option_type_comp_other.capitalize()} Option Prices by Model")
        fig_comp_models_other = go.Figure()
        for model in selected_models:
            fig_comp_models_other.add_trace(
                go.Scatter(
                    x=spot_prices_comp,
                    y=option_prices_models_other[model],
                    mode='lines',
                    name=model
                )
            )
        fig_comp_models_other.update_layout(
            title=f'{option_type_comp_other.capitalize()} Option Prices by Model',
            xaxis_title='Spot Price',
            yaxis_title='Option Price'
        )
        st.plotly_chart(fig_comp_models_other, use_container_width=True)

