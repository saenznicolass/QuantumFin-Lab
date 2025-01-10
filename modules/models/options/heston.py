# modules/models/options/heston.py

import numpy as np
from scipy.integrate import quad
from modules.models.options.black_scholes import black_scholes

def heston_option_pricing(S, K, T, r, kappa, theta, sigma_v, rho, v0, option_type='call'):
    """
    Calculate option price using the Heston model without risk premium lambda.
    """
    try:
        def integrand(phi):
            u = 0.5
            b = kappa
            a = kappa * theta
            d = np.sqrt((rho * sigma_v * phi * 1j - b) ** 2 - sigma_v ** 2 * (2 * u * phi * 1j - phi ** 2))
            g = (b - rho * sigma_v * phi * 1j + d) / (b - rho * sigma_v * phi * 1j - d)
            C = r * phi * 1j * T + a / sigma_v ** 2 * (
                (b - rho * sigma_v * phi * 1j + d) * T
                - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
            )
            D = (b - rho * sigma_v * phi * 1j + d) / sigma_v ** 2 * (
                (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
            )
            return np.real(
                np.exp(-1j * phi * np.log(K))
                * np.exp(C + D * v0 + 1j * phi * np.log(S))
                / (phi * 1j)
            )

        integral = quad(integrand, 0, 100, limit=100)[0]
        call_price = S * (0.5 + integral / np.pi) - K * np.exp(-r * T) * 0.5
        if option_type == 'call':
            return call_price
        else:
            put_price = call_price - S + K * np.exp(-r * T)
            return put_price
    except Exception:
        return np.nan


def heston_option_pricing_lambda(S, K, T, r, kappa, theta, sigma_v, rho, v0, lambd, option_type='call'):
    """
    Calculate option price using the Heston model with risk premium lambda.
    """
    try:
        def integrand(phi):
            u = 0.5
            b = kappa + lambd
            a = kappa * theta
            d = np.sqrt((rho * sigma_v * phi * 1j - b) ** 2 - sigma_v ** 2 * (2 * u * phi * 1j - phi ** 2))
            g = (b - rho * sigma_v * phi * 1j + d) / (b - rho * sigma_v * phi * 1j - d)
            C = r * phi * 1j * T + a / sigma_v ** 2 * (
                (b - rho * sigma_v * phi * 1j + d) * T
                - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
            )
            D = (b - rho * sigma_v * phi * 1j + d) / sigma_v ** 2 * (
                (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
            )
            return np.real(
                np.exp(-1j * phi * np.log(K))
                * np.exp(C + D * v0 + 1j * phi * np.log(S))
                / (phi * 1j)
            )

        integral = quad(integrand, 0, 100, limit=100)[0]
        call_price = S * (0.5 + integral / np.pi) - K * np.exp(-r * T) * 0.5
        if option_type == 'call':
            return call_price
        else:
            put_price = call_price - S + K * np.exp(-r * T)
            return put_price
    except Exception:
        return np.nan
