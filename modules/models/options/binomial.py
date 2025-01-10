# modules/models/options/binomial.py

import numpy as np

def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call'):
    """
    Calculate option price using the Binomial model.
    
    :param S: Current underlying asset price
    :param K: Strike price
    :param T: Time to maturity (in years)
    :param r: Risk-free interest rate
    :param sigma: Volatility (annualized)
    :param N: Number of binomial steps
    :param option_type: 'call' or 'put'
    :return: Option price
    """
    try:
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)

        # Initialize asset prices at maturity
        ST = np.array([S * u ** j * d ** (N - j) for j in range(N + 1)])
        
        # Initialize option values at maturity
        if option_type == 'call':
            C = np.maximum(ST - K, 0)
        else:
            C = np.maximum(K - ST, 0)

        # Backward induction
        for _ in range(N):
            C = np.exp(-r * dt) * (p * C[1:] + (1 - p) * C[:-1])
        return C[0]
    except Exception:
        return np.nan
