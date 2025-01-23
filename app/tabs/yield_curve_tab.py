from app.tabs.yield_curve_subtabs.treasury_analysis import render_treasury_analysis
from app.tabs.yield_curve_subtabs.vasicek_model_tab import render_vasicek_model_tab
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Future imports for additional subtabs will go here

def render_yield_curve_tab():
    """Main function to render the Yield Curve tab"""
    st.header("U.S. Treasury Yield Curve & Interest Rate Models")
    
    # Create tabs for different analyses
    treasury_tab, vasicek_tab = st.tabs([
        "Treasury Analysis",
        "Vasicek Model"
    ])
    
    with treasury_tab:
        render_treasury_analysis()
        
    with vasicek_tab:
        render_vasicek_model_tab()
