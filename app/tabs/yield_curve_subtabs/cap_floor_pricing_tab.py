import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
from typing import Dict, Tuple, Optional

from modules.models.options.interest_rate.interest_rate_options import (
    black_cap_floor_pricing,
    calculate_forward_rate
)

def load_treasury_data() -> Optional[pd.DataFrame]:
    """Load treasury yield data."""
    try:
        # Leer el archivo JSON directamente
        with open("data/IR/all_market_data.json", "r") as f:
            data = json.load(f)
        
        # Convertir la lista de diccionarios a DataFrame
        df = pd.DataFrame(data['treasury_yields'])
        
        # Convertir la columna de fecha a datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Establecer la fecha como índice
        df.set_index('date', inplace=True)
        
        # Ordenar por fecha descendente
        df.sort_index(ascending=False, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading treasury data: {str(e)}")
        return None

def get_latest_yields(data: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Get latest yields from treasury data."""
    if data is None or data.empty:
        # Retornar tasas de ejemplo si no hay datos
        return {
            "3month": 0.04,
            "2year": 0.045,
            "5year": 0.05,
            "7year": 0.052,
            "10year": 0.055,
            "30year": 0.06,
        }
    
    try:
        latest = data.iloc[0]
        return {
            "3month": latest["3month"] / 100,
            "2year": latest["2year"] / 100,
            "5year": latest["5year"] / 100,
            "7year": latest["7year"] / 100,
            "10year": latest["10year"] / 100,
            "30year": latest["30year"] / 100,
        }
    except Exception as e:
        st.warning(f"Error getting latest yields, using example rates: {str(e)}")
        return {
            "3month": 0.04,
            "2year": 0.045,
            "5year": 0.05,
            "7year": 0.052,
            "10year": 0.055,
            "30year": 0.06,
        }

def generate_sensitivity_data(
    base_forward_rate: float,
    base_strike: float,
    base_maturity: float,
    base_volatility: float,
    option_type: str
) -> Dict[str, Dict[str, np.ndarray]]:
    """Genera datos para gráficos de sensibilidad."""
    
    # Rangos para cada variable
    forward_rates = np.linspace(base_forward_rate * 0.5, base_forward_rate * 1.5, 50)
    strikes = np.linspace(base_strike * 0.5, base_strike * 1.5, 50)
    maturities = np.linspace(0.1, base_maturity * 2, 50)
    volatilities = np.linspace(base_volatility * 0.5, base_volatility * 1.5, 50)
    
    # Calcular precios variando cada parámetro
    prices = {
        'forward_rate': np.array([black_cap_floor_pricing(f, base_strike, base_maturity, base_volatility, option_type) 
                                for f in forward_rates]),
        'strike': np.array([black_cap_floor_pricing(base_forward_rate, k, base_maturity, base_volatility, option_type) 
                           for k in strikes]),
        'maturity': np.array([black_cap_floor_pricing(base_forward_rate, base_strike, t, base_volatility, option_type) 
                             for t in maturities]),
        'volatility': np.array([black_cap_floor_pricing(base_forward_rate, base_strike, base_maturity, v, option_type) 
                               for v in volatilities])
    }
    
    return {
        'ranges': {
            'forward_rate': forward_rates,
            'strike': strikes,
            'maturity': maturities,
            'volatility': volatilities
        },
        'prices': prices
    }

def plot_sensitivity_charts(sensitivity_data: Dict[str, Dict[str, np.ndarray]], notional: float = 1.0) -> Dict[str, go.Figure]:
    """Crea gráficos de sensibilidad para cada parámetro."""
    figs = {}
    
    # Labels y títulos para cada parámetro
    param_info = {
        'forward_rate': ('Tasa Forward', 'Sensibilidad a Tasa Forward'),
        'strike': ('Strike Rate', 'Sensibilidad a Strike Rate'),
        'maturity': ('Vencimiento (Años)', 'Sensibilidad a Vencimiento'),
        'volatility': ('Volatilidad', 'Sensibilidad a Volatilidad')
    }
    
    for param, (x_label, title) in param_info.items():
        fig = go.Figure()
        
        x_vals = sensitivity_data['ranges'][param]
        y_vals = sensitivity_data['prices'][param] * notional
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name=f'Precio vs {x_label}'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title='Precio',
            showlegend=True,
            hovermode='x unified'
        )
        
        figs[param] = fig
    
    return figs

def plot_price_surface(
    forward_rates: np.ndarray,
    volatilities: np.ndarray,
    strike: float,
    maturity: float,
    option_type: str,
    notional: float = 1.0
) -> go.Figure:
    """Crea superficie 3D de precios variando tasa forward y volatilidad."""
    
    # Crear grid de valores
    forward_grid, vol_grid = np.meshgrid(forward_rates, volatilities)
    prices = np.zeros_like(forward_grid)
    
    # Calcular precios para cada combinación
    for i in range(forward_grid.shape[0]):
        for j in range(forward_grid.shape[1]):
            prices[i,j] = black_cap_floor_pricing(
                forward_grid[i,j],
                strike,
                maturity,
                vol_grid[i,j],
                option_type
            ) * notional
    
    # Crear figura 3D
    fig = go.Figure(data=[go.Surface(
        x=forward_grid,
        y=vol_grid,
        z=prices,
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title=f'Superficie de Precios - {option_type.title()}',
        scene=dict(
            xaxis_title='Tasa Forward',
            yaxis_title='Volatilidad',
            zaxis_title='Precio'
        ),
        width=800,
        height=800
    )
    
    return fig

def render_cap_floor_pricing_tab():
    """Renders the Cap and Floor Pricing tab within the Yield Curve section."""
    st.subheader("Cap and Floor Pricing with Black Model")

    # Load treasury data with better error handling
    treasury_data = load_treasury_data()
    latest_yields = get_latest_yields(treasury_data)

    # Add data source info
    if treasury_data is not None and not treasury_data.empty:
        st.info("Using updated U.S. Treasury data")
    else:
        st.warning("Using example rates due to data issues")

    # Documentation/explanation
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        ### Black Model for Caps/Floors
        This implementation uses a simplified version of the Black model to calculate prices for interest rate Caps and Floors.
        
        **Main Simplifications:**
        - Flat volatility (constant across all strikes and maturities)
        - Forward rate derived using simplified yield curve
        - No convexity adjustments
        
        **Model Formula:**
        - Cap: `DF * [F * N(d1) - K * N(d2)]`
        - Floor: `DF * [K * N(-d2) - F * N(-d1)]`
        
        where:
        - F: Forward rate
        - K: Strike rate
        - DF: Discount factor
        - N(): Normal distribution function
        """)

    # Input parameters
    st.markdown("### Cap/Floor Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strike_rate = st.number_input(
            "Strike Rate (%)",
            min_value=0.01,
            max_value=15.0,
            value=5.0,
            step=0.1,
            format="%.2f"
        ) / 100.0

        maturity = st.number_input(
            "Maturity (Years)",
            min_value=0.1,
            max_value=30.0,
            value=1.0,
            step=0.1,
            format="%.1f"
        )

    with col2:
        volatility = st.number_input(
            "Volatility (%)",
            min_value=0.01,
            max_value=100.0,
            value=20.0,
            step=0.5,
            format="%.1f"
        ) / 100.0

        option_type = st.selectbox(
            "Option Type",
            ["Cap", "Floor"]
        ).lower()

    with col3:
        notional = st.number_input(
            "Notional Principal (USD)",
            min_value=0,
            value=1_000_000,  # Default 1M USD
            step=100_000,
            format="%d",
            help="Contract notional amount (optional)"
        )

        show_details = st.checkbox("Show Advanced Details", value=False)

    # Calculate forward rate
    if latest_yields:
        forward_rate = calculate_forward_rate(latest_yields, maturity)
        st.info(f"Calculated Forward Rate: {forward_rate:.2%}")
    else:
        forward_rate = 0.05
        st.warning("Using example forward rate (5%) due to missing data")

    # Calculate price
    if st.button("Calculate Cap/Floor Price"):
        with st.spinner("Calculating price..."):
            price = black_cap_floor_pricing(
                forward_rate=forward_rate,
                strike_rate=strike_rate,
                maturity=maturity,
                volatility=volatility,
                option_type=option_type
            )

            if not np.isnan(price):
                results_container = st.container()
                
                with results_container:
                    st.markdown("### Pricing Results")
                    
                    price_cols = st.columns(2)
                    
                    with price_cols[0]:
                        st.metric(
                            label="Price (% of Notional)",
                            value=f"{price:.4%}"
                        )
                    
                    with price_cols[1]:
                        if notional > 0:
                            dollar_price = price * notional
                            st.metric(
                                label="Price (USD)",
                                value=f"${dollar_price:,.2f}"
                            )

                if show_details:
                    st.markdown("### Calculation Details")
                    detail_cols = st.columns(2)
                    
                    with detail_cols[0]:
                        st.markdown("**Market Parameters:**")
                        st.write(f"- Forward Rate: {forward_rate:.2%}")
                        st.write(f"- Volatility: {volatility:.1%}")
                        st.write(f"- Maturity: {maturity:.1f} years")
                    
                    with detail_cols[1]:
                        st.markdown("**Contract Parameters:**")
                        st.write(f"- Type: {option_type.title()}")
                        st.write(f"- Strike Rate: {strike_rate:.2%}")
                        if notional > 0:
                            st.write(f"- Notional: ${notional:,}")
                        
                    st.info("""
                    **Note:** This price is calculated using a simplified model with:
                    - Flat volatility
                    - Constant forward rate
                    - No convexity adjustments
                    """)

                # Visualizations
                st.markdown("### Visualizations")
                
                sensitivity_data = generate_sensitivity_data(
                    forward_rate,
                    strike_rate,
                    maturity,
                    volatility,
                    option_type
                )
                
                charts = plot_sensitivity_charts(sensitivity_data, notional)
                
                sens_tabs = st.tabs([
                    "Forward Rate",
                    "Strike Rate",
                    "Maturity",
                    "Volatility",
                    "3D Surface"
                ])
                
                with sens_tabs[0]:
                    st.plotly_chart(charts['forward_rate'], use_container_width=True)
                
                with sens_tabs[1]:
                    st.plotly_chart(charts['strike'], use_container_width=True)
                
                with sens_tabs[2]:
                    st.plotly_chart(charts['maturity'], use_container_width=True)
                
                with sens_tabs[3]:
                    st.plotly_chart(charts['volatility'], use_container_width=True)
                
                with sens_tabs[4]:
                    # Create and display 3D surface plot
                    forward_range = np.linspace(forward_rate * 0.5, forward_rate * 1.5, 30)
                    vol_range = np.linspace(volatility * 0.5, volatility * 1.5, 30)
                    surface_fig = plot_price_surface(
                        forward_range,
                        vol_range,
                        strike_rate,
                        maturity,
                        option_type,
                        notional
                    )
                    st.plotly_chart(surface_fig, use_container_width=True)

            else:
                st.error("Calculation error. Please verify parameters.")
