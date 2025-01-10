# modules/models/options/black_scholes.py

import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.
    
    :param S: Current underlying asset price
    :param K: Strike price
    :param T: Time to maturity (in years)
    :param r: Risk-free interest rate
    :param sigma: Volatility (annualized)
    :param option_type: 'call' or 'put'
    :return: Option price
    """
    try:
        S = float(S)
        K = float(K)
        T = float(T)
        r = float(r)
        sigma = float(sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return option_price
    except Exception:
        return np.nan
