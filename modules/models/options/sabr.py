# modules/models/options/sabr.py

import numpy as np
from scipy.stats import norm

def sabr_option_pricing(F, K, T, alpha, beta, rho, nu, r, option_type='call'):
    """
    Calculate option price using the SABR model approximation formula.
    """
    try:
        if F <= 0 or K <= 0 or T <= 0 or alpha <= 0 or nu <= 0:
            return np.nan

        epsilon = 1e-7
        if abs(F - K) < epsilon:
            # ATM approximation
            V = (alpha / (F ** (1 - beta))) * (
                1 + ((1 - beta) ** 2 * alpha ** 2) / (24 * F ** (2 - 2 * beta))
                + (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
                + ((2 - 3 * rho ** 2) * nu ** 2) / 24
            ) * T
        else:
            # Regular SABR approximation
            z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
            V = (alpha * z / x_z) * (F * K) ** ((beta - 1) / 2) * (
                1 + ((1 - beta) ** 2 * alpha ** 2) / (24 * (F * K) ** (1 - beta))
                + (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2))
                + ((2 - 3 * rho ** 2) * nu ** 2) / 24
            ) * T

        d1 = (np.log(F / K) + 0.5 * V ** 2 * T) / (V * np.sqrt(T))
        d2 = d1 - V * np.sqrt(T)
        
        if option_type == 'call':
            option_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            option_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        return option_price
    except Exception:
        return np.nan
