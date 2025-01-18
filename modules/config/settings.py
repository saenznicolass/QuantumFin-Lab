"""
modules/config/settings.py

Global configurations for the QuantumFin-Lab project.
Centralizes all configuration settings used throughout the application.
"""

import os
from .constants import (
    DEFAULT_TICKERS,
    TRADING_DAYS_PER_YEAR,
    CONFIDENCE_LEVELS,
    PORTFOLIO_OBJECTIVES,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_MIN_WEIGHT,
    DEFAULT_MAX_WEIGHT
)

class Config:
    """
    Configuration class that holds all settings for the application.
    Integrates with constants.py and allows for environment variable overrides.
    """
    # Data fetching settings
    DATA_PERIOD = os.getenv("DATA_PERIOD", "2y")
    DATA_INTERVAL = os.getenv("DATA_INTERVAL", "1d")

    # Risk and portfolio settings
    RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", DEFAULT_RISK_FREE_RATE))
    MIN_WEIGHT = float(os.getenv("MIN_WEIGHT", DEFAULT_MIN_WEIGHT))
    MAX_WEIGHT = float(os.getenv("MAX_WEIGHT", DEFAULT_MAX_WEIGHT))
    ALLOW_SHORT_SELLING = os.getenv("ALLOW_SHORT_SELLING", "True").lower() in ("true", "1", "yes")

    # Market data settings
    DEFAULT_TICKERS = DEFAULT_TICKERS
    TRADING_DAYS = TRADING_DAYS_PER_YEAR

    # Risk management settings
    CONFIDENCE_LEVELS = CONFIDENCE_LEVELS
    DEFAULT_CONFIDENCE_LEVEL = 0.95

    # Portfolio optimization settings
    PORTFOLIO_OBJECTIVES = PORTFOLIO_OBJECTIVES
    DEFAULT_PORTFOLIO_OBJECTIVE = "Max Sharpe Ratio"

    # Feature toggles
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "True").lower() in ("true", "1", "yes")

    # Advanced optimization settings
    MAX_OPTIMIZATION_ITERATIONS = int(os.getenv("MAX_OPTIMIZATION_ITERATIONS", "1000"))
    OPTIMIZATION_TOLERANCE = float(os.getenv("OPTIMIZATION_TOLERANCE", "1e-6"))

    # Cache settings
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Time to live in seconds

    # API configuration
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

    # Monte Carlo simulation settings
    MC_SIMULATIONS = int(os.getenv("MC_SIMULATIONS", "10000"))
    MC_TIME_HORIZON = int(os.getenv("MC_TIME_HORIZON", "252"))

    # Rolling window analysis settings
    DEFAULT_ROLLING_WINDOW = int(os.getenv("DEFAULT_ROLLING_WINDOW", "252"))

    @classmethod
    def get_portfolio_constraints(cls, allow_short=None):
        """
        Returns portfolio constraints based on configuration and parameters.
        """
        if allow_short is None:
            allow_short = cls.ALLOW_SHORT_SELLING

        return {
            'min_weight': -1.0 if allow_short else 0.0,
            'max_weight': cls.MAX_WEIGHT,
            'allow_short': allow_short
        }

    @classmethod
    def get_optimization_params(cls):
        """
        Returns optimization parameters as a dictionary.
        """
        return {
            'max_iterations': cls.MAX_OPTIMIZATION_ITERATIONS,
            'tolerance': cls.OPTIMIZATION_TOLERANCE,
            'risk_free_rate': cls.RISK_FREE_RATE
        }

# Create a global instance of the configuration
config = Config()

# Validate configuration on import
def validate_config(config):
    """Validates critical configuration parameters."""
    assert 0 <= config.RISK_FREE_RATE <= 1, "Risk-free rate must be between 0 and 1"
    assert config.MIN_WEIGHT <= config.MAX_WEIGHT, "MIN_WEIGHT must be less than or equal to MAX_WEIGHT"
    assert config.MC_SIMULATIONS > 0, "Number of Monte Carlo simulations must be positive"

try:
    validate_config(config)
except AssertionError as e:
    import warnings
    warnings.warn(f"Configuration validation failed: {str(e)}")
