import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple

def black_cap_floor_pricing(
    forward_rate: float,
    strike_rate: float,
    maturity: float,
    volatility: float,
    option_type: str,
    discount_factor: float = 1.0
) -> float:
    """
    Calculates the price of a Cap or Floor using the Black model.

    Parameters
    ----------
    forward_rate : float
        Forward rate of the underlying interest rate.
    strike_rate : float
        Strike rate (exercise rate).
    maturity : float
        Time to maturity in years.
    volatility : float
        Black (normal) volatility.
    option_type : str
        Option type: 'cap' or 'floor'.
    discount_factor : float, optional
        Discount factor for expected payoff (default: 1.0).

    Returns
    -------
    float
        Price of the Cap or Floor.

    Notes
    -----
    This is a simplified implementation of the Black model for Caps/Floors that:
    - Assumes flat volatility
    - Uses constant forward rate
    - Does not consider convexity adjustments
    """
    try:
        # Validate inputs
        if forward_rate <= 0 or strike_rate <= 0 or maturity <= 0 or volatility <= 0:
            return 0.0

        # Handle degenerate cases
        if abs(forward_rate - strike_rate) < 1e-10:
            # ATM case - use simple approximation
            if option_type.lower() == 'cap':
                return discount_factor * forward_rate * volatility * np.sqrt(maturity / 2 * np.pi)
            else:
                return discount_factor * strike_rate * volatility * np.sqrt(maturity / 2 * np.pi)

        # Regular Black formula calculation
        try:
            d1 = (np.log(forward_rate / strike_rate) + 0.5 * volatility**2 * maturity) / (volatility * np.sqrt(maturity))
            d2 = d1 - volatility * np.sqrt(maturity)

            if option_type.lower() == 'cap':
                price = discount_factor * (forward_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2))
            elif option_type.lower() == 'floor':
                price = discount_factor * (strike_rate * norm.cdf(-d2) - forward_rate * norm.cdf(-d1))
            else:
                return 0.0
            
            return max(0.0, price)
        except (ValueError, RuntimeWarning):
            # If log calculation fails, return intrinsic value
            if option_type.lower() == 'cap':
                return max(0.0, discount_factor * (forward_rate - strike_rate))
            else:
                return max(0.0, discount_factor * (strike_rate - forward_rate))
            
    except Exception as e:
        # print(f"Error in black_cap_floor_pricing: {str(e)}")
        return 0.0

def calculate_forward_rate(yields: Dict[str, float], maturity: float) -> float:
    """
    Calculates a simplified forward rate based on a yield curve.

    Parameters
    ----------
    yields : Dict[str, float]
        Dictionary with yields by tenor.
    maturity : float
        Maturity for which to calculate the forward rate.

    Returns
    -------
    float
        Calculated forward rate.
    """
    # Simple implementation using linear interpolation
    maturities = np.array([float(k.replace('month', '')) / 12 if 'month' in k else float(k.replace('year', '')) 
                          for k in yields.keys()])
    rates = np.array(list(yields.values()))
    
    return np.interp(maturity, maturities, rates)
