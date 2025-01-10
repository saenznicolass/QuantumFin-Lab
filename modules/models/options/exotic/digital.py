# modules/models/options/digital.py

import numpy as np
from scipy.stats import norm

def digital_option_pricing(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the price of a digital (cash-or-nothing) option.
    """
    try:
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return np.exp(-r * T) * norm.cdf(d2)
        else:
            return np.exp(-r * T) * norm.cdf(-d2)
    except Exception:
        return np.nan
