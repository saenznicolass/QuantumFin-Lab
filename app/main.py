import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.tabs.trading_strategies_tab import render_trading_strategies_tab
from app.tabs.portfolio_tab import render_portfolio_tab
from app.tabs.option_pricing_tab import render_option_pricing_tab
from app.tabs.yield_curve_tab import render_yield_curve_tab
import streamlit as st

from tabs.model_comparison_tab import render_model_comparison_tab


def main():
    st.set_page_config(page_title="QuantumFin-Lab", layout="wide")
    st.sidebar.title("QuantumFin-Lab Navigation")
    selected_tab = st.sidebar.radio("Select a tab:", ["Yield Curve Analysis", "Trading Strategies", "Portfolio Optimization and Risk", "Option Pricing", "Model Comparison"])

    if selected_tab == "Trading Strategies":
        render_trading_strategies_tab()
    elif selected_tab == "Portfolio Optimization and Risk":
        render_portfolio_tab()
    elif selected_tab == "Option Pricing":
        render_option_pricing_tab()
    elif selected_tab == "Model Comparison":
        render_model_comparison_tab()
    elif selected_tab == "Yield Curve Analysis":
        render_yield_curve_tab()

if __name__ == "__main__":
    main()
