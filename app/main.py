from tabs.trading_strategies_tab import render_trading_strategies_tab
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from tabs.option_pricing_tab import render_option_pricing_tab
from tabs.model_comparison_tab import render_model_comparison_tab
from tabs.portfolio_tab import render_portfolio_tab


def main():
    st.set_page_config(page_title="QuantumFin-Lab", layout="wide")
    st.sidebar.title("QuantumFin-Lab Navigation")
    selected_tab = st.sidebar.radio("Select a tab:", ["Trading Strategies", "Portfolio Optimization and Risk", "Option Pricing", "Model Comparison"])

    if selected_tab == "Trading Strategies":
        render_trading_strategies_tab()
    elif selected_tab == "Portfolio Optimization and Risk":
        render_portfolio_tab()
    elif selected_tab == "Option Pricing":
        render_option_pricing_tab()
    elif selected_tab == "Model Comparison":
        render_model_comparison_tab()

if __name__ == "__main__":
    main()
