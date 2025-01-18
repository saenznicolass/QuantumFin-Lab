# modules/models/options/asian.py

import numpy as np
from modules.models.options.black_scholes import black_scholes
from scipy.stats import norm

def asian_option_pricing(S, K, T, r, sigma, option_type='call'):
    """
    Calculate an approximate price for an arithmetic Asian option using
    a simplification (average price ~ current price).
    """
    try:
        avg_S = S  # Simplified assumption that the average equals current price
        sigma_avg = sigma / np.sqrt(3)
        option_price = black_scholes(avg_S, K, T, r, sigma_avg, option_type)
        return option_price
    except Exception:
        return np.nan

def geometric_asian_option_pricing(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the price of a geometric Asian option (closed-form approximation).
    """
    try:
        sigma_g = sigma / np.sqrt(3)
        mu_g = 0.5 * (r - 0.5 * sigma**2 + (r + 0.5 * sigma**2)) * T
        d1 = (np.log(S / K) + mu_g + 0.5 * sigma_g**2 * T) / (sigma_g * np.sqrt(T))
        d2 = d1 - sigma_g * np.sqrt(T)
        if option_type == 'call':
            return (
                np.exp(-r * T) * S * np.exp((mu_g - r) * T) * norm.cdf(d1)
                - np.exp(-r * T) * K * norm.cdf(d2)
            )
        else:
            return (
                np.exp(-r * T) * K * norm.cdf(-d2)
                - np.exp(-r * T) * S * np.exp((mu_g - r) * T) * norm.cdf(-d1)
            )
    except Exception:
        return np.nan
