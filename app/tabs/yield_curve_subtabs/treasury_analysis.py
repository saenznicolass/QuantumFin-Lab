import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

def load_yield_curve_data():
    """Load and validate yield curve data"""
    try:
        DATA_PATH = os.path.join("data", "IR", "treasury_yields.csv")
        data_yc = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        if not data_yc.empty:
            # print("Columns in yield_curve_data:", data_yc.columns.tolist())
            # Store in session state for other tabs to use
            st.session_state.yield_curve_data = data_yc
            return data_yc
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def plot_yield_evolution(data_yc):
    """Create time evolution plot of yields"""
    fig = go.Figure()
    
    # Add slider for date range selection
    start_date = st.date_input("Start Date", data_yc.index.min())
    end_date = st.date_input("End Date", data_yc.index.max())
    
    # Filter data based on selected date range
    mask = (data_yc.index >= pd.Timestamp(start_date)) & (data_yc.index <= pd.Timestamp(end_date))
    filtered_data = data_yc[mask]
    
    # Allow users to select specific maturities
    selected_maturities = st.multiselect(
        "Select Maturities",
        options=data_yc.columns.tolist(),
        default=data_yc.columns.tolist()
    )
    
    for maturity in selected_maturities:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data[maturity],
            mode='lines',
            name=maturity
        ))

    fig.update_layout(
        title='Yield Evolution by Maturity',
        xaxis_title='Date',
        yaxis_title='Yield (%)',
        hovermode='x unified',
        legend_title='Maturity'
    )
    return fig

def plot_current_curve(data_yc):
    """Create current yield curve plot with historical comparison"""
    fig = go.Figure()
    
    # Allow selection of comparison dates
    dates = data_yc.index.date
    selected_date = st.selectbox("Select Date for Yield Curve", dates)
    compare_date = st.selectbox("Compare with Date (optional)", ["None"] + [d.strftime("%Y-%m-%d") for d in dates])
    
    # Plot selected date
    selected_yields = data_yc.loc[str(selected_date)]
    fig.add_trace(go.Scatter(
        x=list(range(len(data_yc.columns))),
        y=selected_yields,
        mode='lines+markers',
        name=f'Curve {selected_date}'
    ))
    
    # Plot comparison date if selected
    if compare_date != "None":
        compare_yields = data_yc.loc[compare_date]
        fig.add_trace(go.Scatter(
            x=list(range(len(data_yc.columns))),
            y=compare_yields,
            mode='lines+markers',
            name=f'Curve {compare_date}'
        ))
    
    fig.update_layout(
        title='U.S. Treasury Yield Curve Comparison',
        xaxis=dict(
            ticktext=list(data_yc.columns),
            tickvals=list(range(len(data_yc.columns)))
        ),
        yaxis_title='Yield (%)',
        showlegend=True
    )
    return fig

def plot_3d_yield_curves(data_yc):
    """Create a 3D plot of yield curves over time"""
    fig = go.Figure()

    # Create a mesh grid for dates and maturities
    dates = data_yc.index
    maturities = data_yc.columns
    X, Y = np.meshgrid(dates, maturities)
    Z = data_yc.T.values

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(
            title='Yield (%)',
            titleside='right'
        )
    ))

    fig.update_layout(
        title='3D Yield Curves Over Time',
        scene=dict(
            xaxis_title='Date',
            yaxis_title='Maturity',
            zaxis_title='Yield (%)'
        )
    )
    return fig

def render_treasury_analysis():
    """Main function to render the Yield Curve tab"""
    st.header("U.S. Treasury Yield Curve Analysis")
    
    # Add tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Yield Evolution", 
        "Current Curve", 
        "Statistical Analysis",
        "Documentation"
    ])
    
    # Load data and store in session state if not already loaded
    if 'yield_curve_data' not in st.session_state:
        data_yc = load_yield_curve_data()
        if data_yc is None:
            st.error("Failed to load yield curve data. Please check the data file.")
            return
    else:
        data_yc = st.session_state.yield_curve_data
        
    # Add a refresh button
    if st.button("Refresh Yield Curve Data"):
        data_yc = load_yield_curve_data()
        if data_yc is None:
            st.error("Failed to refresh yield curve data.")
            return
            
    with tab1:
        st.plotly_chart(plot_yield_evolution(data_yc), use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_current_curve(data_yc), use_container_width=True)
        with col2:
            st.plotly_chart(plot_3d_yield_curves(data_yc), use_container_width=False)
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Descriptive Statistics")
            st.dataframe(data_yc.describe())
        
        with col2:
            st.subheader("Correlation Matrix")
            fig = go.Figure(data=go.Heatmap(
                z=data_yc.corr(),
                x=data_yc.columns,
                y=data_yc.columns,
                colorscale='RdBu',
                zmin=-1, zmax=1
            ))
            st.plotly_chart(fig, use_container_width=True)
            
    with tab4:
        st.markdown("""
        ### About the Yield Curve Analysis
        
        This tool provides simple analysis of U.S. Treasury Yield Curves:
        
        - **Yield Evolution**: Track how yields change over time for different maturities
        - **Curve Comparison**: Compare yield curves across different dates
        - **Statistical Analysis**: View correlations and key statistics
        
        #### Key Concepts:
        - The yield curve shows interest rates across different loan terms
        - A normal curve slopes upward (longer terms = higher yields)
        - An inverted curve (slopes down) often signals recession risks
        
        #### Data Sources:
        - U.S. Treasury data via Alpha Vantage API
        - Daily updates for accurate analysis
        """)