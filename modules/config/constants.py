"""
modules/config/constants.py

Constant values that are used throughout the QuantumFin-Lab project.
These values are generally not expected to change during runtime.
"""

# Ticker options for the dropdown menu
OPTIONS_TICKERS = [
    'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'AAPL', 'NFLX', 'META', 'NVDA', 'PYPL',
    'INTC', 'AMD', 'CRM', 'ADBE', 'CSCO', 'QCOM', 'UBER', 'ZM', 'SQ', 'SHOP',
    'BARC.L', 'BBVA.MC', 'BNP.PA', 'BP.L', 'CAD=X', 'CBK.DE', 'CHF=X', 'CNY=X',
    'DBK.DE', 'EEM', 'ENI.MI', 'EQNR.OL', 'EURUSD=X', 'EWZ', 'FSLR', 'FXI',
    'GBPUSD=X', 'GLE.PA', 'HSBA.L', 'JPY=X', 'MXN=X', 'NEE',
    'TCS', 'CAMS', 'ITC', 'INFY'
]
# Default tickers for the application
DEFAULT_TICKERS = [
    'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BARC.L', 'BBVA.MC', 'NFLX', 
    'BNP.PA', 'BP.L', 'CAD=X', 'CBK.DE', 'CHF=X', 'CNY=X',
    'DBK.DE', 'EEM', 'ENI.MI', 'EQNR.OL', 'EURUSD=X', 'GBPUSD=X'
]

# Days in a trading year (approx)
TRADING_DAYS_PER_YEAR: int = 252  # Typical number of trading days in a year

# Typical default confidence levels for VaR, CVaR, etc.
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Common portfolio optimization objectives
PORTFOLIO_OBJECTIVES = [
    "Max Sharpe Ratio",
    "Min Volatility",
    "Target Return"
]

# Default maximum steps for binomial model
BINOMIAL_MAX_STEPS: int = 500

# Common used text for disclaimers
DISCLAIMER_TEXT = (
    "The calculations and models provided in this application "
    "are for educational purposes only and should not be used "
    "for actual trading or investment decisions."
)

# Example currency or ticker references
DEFAULT_CURRENCY = "USD"
DEFAULT_BENCHMARK_TICKER = "SPY"

# Example base directory or subfolder for data storage
DATA_STORAGE_PATH = "data/downloads"

# Additional constants relevant to the portfolio environment
MINIMUM_PORTFOLIO_SIZE = 2

# Portfolio optimization parameters
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_MAX_WEIGHT = 1.0
