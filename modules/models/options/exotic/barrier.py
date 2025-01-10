# modules/models/options/barrier.py

import numpy as np
from scipy.stats import norm
from modules.models.options.black_scholes import black_scholes

def barrier_option_pricing(S, K, H, T, r, sigma, option_type='call', barrier_type='up-and-out'):
    """
    Calculate option price for a barrier option using an analytic approach (up-and-out, down-and-out).
    """
    try:
        if barrier_type == 'up-and-out':
            if S >= H:
                return 0
            else:
                option_price = black_scholes(S, K, T, r, sigma, option_type)
                lambda_val = (r + (sigma ** 2) / 2) / (sigma ** 2)
                x1 = np.log(S / H) / (sigma * np.sqrt(T)) + lambda_val * sigma * np.sqrt(T)
                y1 = np.log((H ** 2) / (S * K)) / (sigma * np.sqrt(T)) + lambda_val * sigma * np.sqrt(T)
                price_adjustment = S * (H / S) ** (2 * lambda_val) * norm.cdf(y1)
                if option_type == 'call':
                    return option_price - price_adjustment
                else:
                    return option_price - price_adjustment
        
        elif barrier_type == 'down-and-out':
            if S <= H:
                return 0
            else:
                option_price = black_scholes(S, K, T, r, sigma, option_type)
                lambda_val = (r + (sigma ** 2) / 2) / (sigma ** 2)
                x1 = np.log(S / H) / (sigma * np.sqrt(T)) + lambda_val * sigma * np.sqrt(T)
                y1 = np.log((H ** 2) / (S * K)) / (sigma * np.sqrt(T)) + lambda_val * sigma * np.sqrt(T)
                price_adjustment = S * (H / S) ** (2 * lambda_val) * norm.cdf(y1)
                if option_type == 'call':
                    return option_price - price_adjustment
                else:
                    return option_price - price_adjustment
        else:
            return np.nan
    except Exception:
        return np.nan
