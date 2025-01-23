import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.stats import norm

def vasicek_model_simulation(
    a: float,
    b: float,
    sigma: float,
    r0: float,
    n_simulations: int,
    time_horizon_years: int,
    time_steps_per_year: int = 252
) -> pd.DataFrame:
    """
    Simulates short-term interest rate paths using the Vasicek model.

    Parameters
    ----------
    a : float
        Mean reversion speed (kappa).
    b : float
        Long-term mean level (theta).
    sigma : float
        Interest rate process volatility.
    r0 : float
        Initial interest rate.
    n_simulations : int
        Number of paths to simulate.
    time_horizon_years : int
        Simulation time horizon in years.
    time_steps_per_year : int, optional
        Number of time steps per year (default: 252 - trading days).

    Returns
    -------
    pd.DataFrame
        DataFrame with simulated paths. Each column represents a simulation,
        and the index represents time (in days if time_steps_per_year=252).
    """
    dt = 1 / time_steps_per_year
    time_horizon_steps = int(time_horizon_years * time_steps_per_year)
    time_index = pd.date_range(
        start=pd.Timestamp.now(), 
        periods=time_horizon_steps, 
        freq='B'
    )

    rates = np.zeros((n_simulations, time_horizon_steps))
    rates[:, 0] = r0

    for i in range(1, time_horizon_steps):
        dWt = np.random.normal(0, np.sqrt(dt), size=n_simulations)
        rates[:, i] = rates[:, i-1] + a * (b - rates[:, i-1]) * dt + sigma * dWt

    return pd.DataFrame(rates.T, index=time_index)

def calculate_vasicek_statistics(simulated_paths: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics for the simulated Vasicek paths.
    """
    final_rates = simulated_paths.iloc[-1]
    return {
        'mean': final_rates.mean(),
        'std': final_rates.std(),
        'min': final_rates.min(),
        'max': final_rates.max(),
        'median': final_rates.median()
    }

def calculate_confidence_bands(simulated_paths: pd.DataFrame, confidence_level: float = 0.95) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate confidence bands for the simulated paths.
    
    Parameters
    ----------
    simulated_paths : pd.DataFrame
        DataFrame containing simulated paths
    confidence_level : float
        Confidence level (default: 0.95 for 95% confidence bands)
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Lower and upper confidence bands
    """
    alpha = (1 - confidence_level) / 2
    lower_band = simulated_paths.quantile(alpha, axis=1)
    upper_band = simulated_paths.quantile(1 - alpha, axis=1)
    
    return lower_band, upper_band

def calculate_term_structure(a: float, b: float, sigma: float, r0: float, maturities: np.ndarray) -> pd.Series:
    """
    Calculate the term structure implied by Vasicek model parameters.
    
    Parameters
    ----------
    a : float
        Mean reversion speed
    b : float
        Long-term mean level
    sigma : float
        Volatility
    r0 : float
        Initial short rate
    maturities : np.ndarray
        Array of maturities in years
        
    Returns
    -------
    pd.Series
        Term structure of interest rates
    """
    B = (1 - np.exp(-a * maturities)) / a
    A = (b - (sigma**2)/(2*a**2)) * (B - maturities) - (sigma**2 * B**2)/(4*a)
    rates = -1/maturities * (A - B * r0)
    return pd.Series(rates, index=maturities)

def calculate_negative_rate_probability(a: float, b: float, sigma: float, r0: float, horizon: float) -> float:
    """
    Calculate the probability of negative rates at a given horizon.
    
    Parameters
    ----------
    a : float
        Mean reversion speed
    b : float
        Long-term mean level
    sigma : float
        Volatility
    r0 : float
        Initial short rate
    horizon : float
        Time horizon in years
        
    Returns
    -------
    float
        Probability of negative rates
    """
    mean = r0 * np.exp(-a * horizon) + b * (1 - np.exp(-a * horizon))
    var = (sigma**2/(2*a)) * (1 - np.exp(-2*a*horizon))
    std = np.sqrt(var)
    
    return norm.cdf(0, mean, std)

def analyze_mean_reversion(simulated_paths: pd.DataFrame, a: float, b: float) -> Dict[str, float]:
    """
    Analyze the mean reversion characteristics of the simulated paths.
    
    Parameters
    ----------
    simulated_paths : pd.DataFrame
        Simulated interest rate paths
    a : float
        Mean reversion speed
    b : float
        Long-term mean level
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing mean reversion metrics
    """
    final_rates = simulated_paths.iloc[-1]
    mean_final = final_rates.mean()
    
    return {
        'theoretical_mean': b,
        'simulated_mean': mean_final,
        'mean_difference': abs(b - mean_final),
        'half_life': np.log(2)/a,
        'adjustment_speed': a
    }
