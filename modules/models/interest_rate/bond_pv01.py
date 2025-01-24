import numpy as np
import pandas as pd

def convert_treasury_period_to_years(period: str) -> float:
    """Convert treasury period string to years float."""
    if 'MO' in period:
        return float(period.replace('MO', '')) / 12
    elif 'Y' in period:
        return float(period.replace('Y', ''))
    return 0

def find_closest_maturity_column(yield_curve_data: pd.DataFrame, time_to_maturity_years: float) -> str:
    """Find closest maturity column in yield curve data."""
    try:
        # Imprime las columnas disponibles para depuración
        # print("Available columns:", yield_curve_data.columns.tolist())
        
        # Mapeo más flexible para diferentes formatos posibles
        maturity_map = {}
        for col in yield_curve_data.columns:
            col_lower = col.lower()
            if 'month' in col_lower or 'mo' in col_lower:
                # Extrae el número antes de "month" o "mo"
                months = float(''.join(filter(str.isdigit, col_lower)))
                maturity_map[col] = months / 12.0
            elif 'year' in col_lower or 'y' in col_lower:
                # Extrae el número antes de "year" o "y"
                years = float(''.join(filter(str.isdigit, col_lower)))
                maturity_map[col] = years
        
        if not maturity_map:
            raise ValueError(f"No valid maturity columns found. Available columns: {yield_curve_data.columns.tolist()}")
            
        # Encontrar el vencimiento más cercano
        closest_maturity = min(maturity_map.items(), 
                             key=lambda x: abs(x[1] - time_to_maturity_years))
        return closest_maturity[0]
    
    except Exception as e:
        print(f"Error in find_closest_maturity_column: {str(e)}")
        return None

def calculate_bond_pv(maturity_years: float,
                     coupon_rate: float,
                     notional: float,
                     yield_curve_data: pd.DataFrame,
                     valuation_date: pd.Timestamp) -> float:
    """Calculate present value of a bond using the yield curve."""
    try:
        coupon_rate_decimal = coupon_rate / 100.0
        maturity_date = valuation_date + pd.DateOffset(years=maturity_years)
        payment_dates = pd.date_range(
            start=valuation_date, 
            end=maturity_date, 
            freq='6ME'  # Semi-annual payments
        )

        pv = 0.0
        for payment_date in payment_dates:
            if payment_date > valuation_date:
                time_to_payment = (payment_date - valuation_date).days / 365.25
                closest_maturity = find_closest_maturity_column(yield_curve_data, time_to_payment)
                
                if closest_maturity:
                    discount_rate = yield_curve_data.loc[valuation_date, closest_maturity] / 100.0
                    
                    if payment_date == maturity_date:
                        cash_flow = notional * (1 + coupon_rate_decimal/2)
                    else:
                        cash_flow = notional * (coupon_rate_decimal/2)
                    
                    pv += cash_flow / (1 + discount_rate)**time_to_payment
        
        return pv
    except Exception as e:
        print(f"Error calculating bond PV: {str(e)}")
        return np.nan

def calculate_bond_pv01(maturity_years: float,
                       coupon_rate: float,
                       notional: float,
                       yield_curve_data: pd.DataFrame,
                       valuation_date: pd.Timestamp,
                       basis_point_shock: float = 0.01) -> float:
    """Calculate PV01 (Price Value of a Basis Point) of a bond."""
    try:
        pv_original = calculate_bond_pv(
            maturity_years, coupon_rate, notional, 
            yield_curve_data, valuation_date
        )

        yield_curve_shocked = yield_curve_data.copy() + basis_point_shock
        
        pv_shocked = calculate_bond_pv(
            maturity_years, coupon_rate, notional,
            yield_curve_shocked, valuation_date
        )

        pv01 = abs(pv_original - pv_shocked)
        return pv01
    
    except Exception as e:
        print(f"Error calculating PV01: {str(e)}")
        return np.nan
