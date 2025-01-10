# modules/models/options/merton.py

import numpy as np
from modules.models.options.black_scholes import black_scholes

def merton_option_pricing(S, K, T, r, sigma, jump_intensity, jump_mean, jump_std, option_type='call'):
    """
    Calculate option price using Merton's Jump-Diffusion model.
    
    :param S: Current underlying asset price
    :param K: Strike price
    :param T: Time to maturity (in years)
    :param r: Risk-free interest rate
    :param sigma: Volatility (annualized)
    :param jump_intensity: Lambda parameter (jump intensity)
    :param jump_mean: Mean of the jump size
    :param jump_std: Std dev of the jump size
    :param option_type: 'call' or 'put'
    :return: Option price
    """
    try:
        lambda_p = jump_intensity
        # kappa = E[e^{jump}] - 1
        kappa = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1
        r_adj = r - lambda_p * kappa
        sigma_adj = np.sqrt(sigma ** 2 + (jump_std ** 2) * lambda_p / T)
        return black_scholes(S, K, T, r_adj, sigma_adj, option_type)
    except Exception:
        return np.nan
