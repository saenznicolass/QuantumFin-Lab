import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.models.interest_rate.bond_pv01 import calculate_bond_pv01, calculate_bond_pv

def create_duration_estimate(price_up, price_down, current_price, yield_change):
    """Estimate modified duration from price changes."""
    return abs(price_up - price_down) / (2 * current_price * yield_change)

def render_pv01_bond_tab():
    """Render the Bond PV01 Calculation tab."""
    st.header("Bond PV01 & Risk Analysis")
      
        
    # Main layout with tabs
    tab1, tab2, tab3 = st.tabs(["Calculator", "Risk Analysis", "Documentation"])
    
    with tab1:
                # Sidebar para configuración avanzada
        with st.sidebar:
            st.subheader("Advanced Settings")
            shock_size = st.number_input(
                "Basis Point Shock Size",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Size of yield shock in basis points for PV01 calculation"
            )

        # Principal inputs row
        col1, col2, col3 = st.columns(3)
        with col1:
            maturity = st.number_input(
                "Maturity (Years)",
                min_value=0.25,
                max_value=30.0,
                value=10.0,
                step=0.25
            )
        with col2:
            coupon = st.number_input(
                "Coupon Rate (%)",
                min_value=0.0,
                max_value=15.0,
                value=4.0,
                step=0.125
            )
        with col3:
            notional = st.number_input(
                "Notional",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000,
                format="%d"
            )

        # Yield curve data check
        if 'yield_curve_data' not in st.session_state or st.session_state.yield_curve_data.empty:
            st.error("⚠️ Please load yield curve data in the Treasury Analysis tab first!")
            return

        yield_curve = st.session_state.yield_curve_data
        
        # Date selection and calculation
        col1, col2 = st.columns([2,1])
        with col1:
            dates = yield_curve.index.date
            selected_date = st.selectbox(
                "Valuation Date",
                options=dates,
                index=len(dates)-1
            )

        valuation_date = pd.Timestamp(selected_date)
        
        if valuation_date not in yield_curve.index:
            st.error(f"No yield curve data for {selected_date}")
            return
            
        try:
            with st.spinner("Analyzing bond metrics..."):
                # Current Yield Curve
                current_yields = yield_curve.loc[valuation_date]
                
                # Calculations
                pv01 = calculate_bond_pv01(
                    maturity_years=maturity,
                    coupon_rate=coupon,
                    notional=notional,
                    yield_curve_data=yield_curve,
                    valuation_date=valuation_date,
                    basis_point_shock=shock_size/100.0
                )
                
                current_price = calculate_bond_pv(
                    maturity_years=maturity,
                    coupon_rate=coupon,
                    notional=notional,
                    yield_curve_data=yield_curve,
                    valuation_date=valuation_date
                )

                if np.isnan(pv01) or np.isnan(current_price):
                    st.error("Calculation error. Please check parameters.")
                    return

                # Métricas principales
                st.markdown("### Key Risk Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "PV01",
                        f"${pv01:,.2f}",
                        delta=f"{(pv01/current_price)*100:.4f}% of price"
                    )
                with col2:
                    st.metric(
                        "Bond Price",
                        f"${current_price:,.2f}",
                        delta=f"{(current_price/notional-1)*100:.2f}% of par"
                    )
                with col3:
                    # Estimate modified duration
                    yield_up = yield_curve.copy() + 0.0001  # 1bp up
                    yield_down = yield_curve.copy() - 0.0001  # 1bp down
                    
                    price_up = calculate_bond_pv(maturity, coupon, notional, yield_up, valuation_date)
                    price_down = calculate_bond_pv(maturity, coupon, notional, yield_down, valuation_date)
                    
                    mod_duration = create_duration_estimate(price_up, price_down, current_price, 0.0001)
                    st.metric(
                        "Mod. Duration",
                        f"{mod_duration:.2f} years",
                        delta=f"{mod_duration*100:.1f}% price chg/1%yield"
                    )
                with col4:
                    closest_rate = current_yields[yield_curve.columns[0]]
                    ytm_estimate = f"{closest_rate:.2f}%"
                    st.metric(
                        "Est. YTM",
                        ytm_estimate,
                        delta="Using closest treasury rate"
                    )

                # Visualizaciones
                col1, col2 = st.columns(2)
                
                with col1:
                    # Curva de rendimiento actual
                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(
                        x=[col for col in yield_curve.columns],
                        y=current_yields.values,
                        mode='lines+markers',
                        name='Current Yield Curve'
                    ))
                    fig_curve.update_layout(
                        title="Current Yield Curve",
                        xaxis_title="Maturity",
                        yaxis_title="Yield (%)",
                        height=300
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)
                
                with col2:
                    # Análisis de sensibilidad
                    bp_changes = [-50, -25, -10, -5, 0, 5, 10, 25, 50]
                    prices = []
                    
                    for bp in bp_changes:
                        shocked_yield = yield_curve.copy()
                        shocked_yield.loc[valuation_date] += (bp / 100)
                        price = calculate_bond_pv(
                            maturity_years=maturity,
                            coupon_rate=coupon,
                            notional=notional,
                            yield_curve_data=shocked_yield,
                            valuation_date=valuation_date
                        )
                        prices.append(price)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=bp_changes,
                        y=prices,
                        mode='lines+markers',
                        name='Bond Price'
                    ))
                    
                    fig.update_layout(
                        title="Price Sensitivity to Yield Changes",
                        xaxis_title="Yield Change (bp)",
                        yaxis_title="Bond Price ($)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                # Tabla detallada
                if st.checkbox("Show Detailed Analysis Table", value=True):
                    # Calcular PV01 estimates de manera separada para evitar división por cero
                    pv01_estimates = []
                    for bp in bp_changes:
                        if bp != 0:  # Evitar división por cero
                            price_diff = prices[bp_changes.index(bp)] - current_price
                            pv01_est = abs(price_diff / bp)
                            pv01_estimates.append(pv01_est)
                        else:
                            pv01_estimates.append(np.nan)  # Usar NaN para bp = 0
                    
                    sensitivity_df = pd.DataFrame({
                        'Yield Change (bp)': bp_changes,
                        'Bond Price ($)': prices,
                        'Abs. Change ($)': [p - current_price for p in prices],
                        'Rel. Change (%)': [(p - current_price)/current_price * 100 for p in prices],
                        'PV01 Estimate ($)': pv01_estimates
                    })
                    
                    st.dataframe(
                        sensitivity_df.style.format({
                            'Bond Price ($)': '${:,.2f}',
                            'Abs. Change ($)': '${:,.2f}',
                            'Rel. Change (%)': '{:,.3f}%',
                            'PV01 Estimate ($)': '${:,.2f}'
                        }),
                        hide_index=True
                    )
                
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
            st.error("Please verify your inputs and try again.")

    with tab2:
        st.markdown("""
        ### Risk Analysis Tools
        
        #### Key Risk Measures:
        
        1. **PV01 (Price Value of 01)**
           - Measures price sensitivity to 1bp yield change
           - Higher PV01 = Higher interest rate risk
           - Useful for hedge ratio calculations
        
        2. **Modified Duration**
           - Measures % price change for 1% yield change
           - Approximately equal to PV01 * 100 / price
           - Key measure for portfolio immunization
        
        3. **Price Sensitivity**
           - Non-linear relationship between price and yield
           - Demonstrates convexity effect
           - Important for large yield movements
        """)
        
    with tab3:
        st.markdown("""
        ### Bond PV01 Documentation
        
        #### Overview
        The Bond PV01 calculator provides tools for analyzing interest rate risk of fixed-rate bonds.
        
        #### Calculations
        - **Present Value**: Discounted cash flow using treasury curve rates
        - **PV01**: Price change for 1bp parallel shift in yields
        - **Modified Duration**: Estimated using small yield changes
        
        #### Assumptions & Limitations
        1. Uses treasury curve for discounting
        2. Assumes parallel yield curve shifts
        3. Semi-annual coupon payments
        4. No credit spread considerations
        
        #### Usage Tips
        - Compare PV01 across different maturities
        - Use sensitivity analysis for larger moves
        - Consider duration for quick risk estimates
        - Monitor convexity for long-duration bonds
        """)
