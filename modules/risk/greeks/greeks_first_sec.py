# modules/risk/greeks/greeks.py

import numpy as np
from scipy.stats import norm

def calculate_greeks(S, K, T, r, sigma):
    """
    Calculate the Greeks for the Black-Scholes model, including 
    first-order (Delta, Gamma, Theta, Vega, Rho) and extended Greeks 
    (Vomma, Vanna, Speed, Zomma).
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        # Delta
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1

        # Gamma
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))

        # Theta
        theta_call = (
            - (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        theta_put = (
            - (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

        # Vega
        vega = S * np.sqrt(T) * pdf_d1

        # Rho
        rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)
        rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        # Extended Greeks
        vomma = vega * d1 * d2 / sigma if sigma != 0 else 0
        vanna = vega * (1 - d1 / (sigma * np.sqrt(T))) if sigma != 0 else 0
        speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1) if sigma != 0 else 0
        zomma = gamma * ((d1 * d2 - 1) / sigma) if sigma != 0 else 0

        return {
            'Delta Call': delta_call,
            'Delta Put': delta_put,
            'Gamma': gamma,
            'Vega': vega,
            'Theta Call': theta_call,
            'Theta Put': theta_put,
            'Rho Call': rho_call,
            'Rho Put': rho_put,
            'Vomma': vomma,
            'Vanna': vanna,
            'Speed': speed,
            'Zomma': zomma
        }
    except Exception:
        return {}
