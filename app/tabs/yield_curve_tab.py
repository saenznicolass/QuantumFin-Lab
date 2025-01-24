from app.tabs.yield_curve_subtabs.treasury_analysis import render_treasury_analysis
from app.tabs.yield_curve_subtabs.vasicek_model_tab import render_vasicek_model_tab
from app.tabs.yield_curve_subtabs.cap_floor_pricing_tab import render_cap_floor_pricing_tab
from app.tabs.yield_curve_subtabs.pv01_bond_tab import render_pv01_bond_tab
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Future imports for additional subtabs will go here

def render_yield_curve_tab():
    """Main function to render the Yield Curve tab"""
    st.header("U.S. Treasury Yield Curve & Interest Rate Models")
    
    # Create tabs for different analyses
    treasury_tab, vasicek_tab, cap_floor_tab, pv01_tab = st.tabs([
        "Treasury Analysis",
        "Vasicek Model",
        "Caps/Floors Pricing",
        "Bond PV01"
    ])
    
    with treasury_tab:
        render_treasury_analysis()
        
    with vasicek_tab:
        render_vasicek_model_tab()

    with cap_floor_tab:
        render_cap_floor_pricing_tab()
        
    with pv01_tab:
        render_pv01_bond_tab()
