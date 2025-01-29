from app.tabs.yield_curve_subtabs.treasury_analysis import render_treasury_analysis
from app.tabs.yield_curve_subtabs.vasicek_model_tab import render_vasicek_model_tab
from app.tabs.yield_curve_subtabs.cap_floor_pricing_tab import render_cap_floor_pricing_tab
from app.tabs.yield_curve_subtabs.pv01_bond_tab import render_pv01_bond_tab
from app.tabs.yield_curve_subtabs.monte_carlo_var_cvar_tab import render_monte_carlo_var_cvar_tab
from app.tabs.yield_curve_subtabs.interest_rate_models_tab import render_interest_rate_models_tab

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Future imports for additional subtabs will go here

def render_yield_curve_tab():
    """Main function to render the Yield Curve tab"""
    st.header("U.S. Treasury Yield Curve & Interest Rate Models")
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "Treasury Analysis",
        "Interest Rate Models",
        "Vasicek Model",
        "Caps/Floors Pricing",
        "Bond PV01",
        "Monte Carlo"
    ])
    
    with tabs[0]:
        render_treasury_analysis()
    with tabs[1]:
        render_interest_rate_models_tab()
    with tabs[2]:
        render_vasicek_model_tab()
    with tabs[3]:
        render_pv01_bond_tab()
    with tabs[4]:
        render_cap_floor_pricing_tab()
    with tabs[5]:
        render_monte_carlo_var_cvar_tab()
