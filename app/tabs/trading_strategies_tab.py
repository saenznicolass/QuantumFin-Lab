import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from modules.trading.strategies.technical.moving_averages import (
#     TripleMAStrategy, AdaptiveEMAStrategy
# )
from modules.trading.strategies.technical.rsi import (
    RSIMACDStrategy, DynamicMomentumStrategy
)
from modules.trading.strategies.technical.bollinger_bands import (
    BollingerBreakoutStrategy, AdaptiveVolatilityStrategy
)
from modules.trading.strategies.technical.moving_averages import ( TripleMAStrategy, AdaptiveEMAStrategy )

from modules.trading.backtest.engine import BacktestEngine
from modules.visualization.streamlit.strategy_display import display_strategy_dashboard
from modules.market_data.providers.yahoo import fetch_market_data




def render_trading_strategies_tab():
    """Render the Trading Strategies Analysis Tab"""
    st.title("Technical Trading Strategies Analysis")
    
    # Sidebar configuration
    st.sidebar.header("Strategy Configuration")
    
    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Select Strategy Type",
        [
            "Triple Moving Average",
            "Adaptive EMA",
            "RSI-MACD Confluence",
            "Dynamic Momentum",
            "Bollinger Breakout",
            "Adaptive Volatility"
        ]
    )
    
    # Data parameters
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    lookback_period = st.sidebar.slider(
        "Lookback Period (Years)",
        min_value=1,
        max_value=10,
        value=2
    )
    
    # Strategy specific parameters
    params = get_strategy_parameters(strategy_type)
    
    # Risk management parameters
    st.sidebar.subheader("Risk Management")
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.0, 20.0, 2.0) / 100
    take_profit = st.sidebar.slider("Take Profit (%)", 0.0, 40.0, 6.0) / 100
    
    # Execution parameters
    st.sidebar.subheader("Execution Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Capital",
        min_value=1000,
        value=100000
    )
    commission = st.sidebar.slider(
        "Commission (%)",
        0.0,
        1.0,
        0.1
    ) / 100
    
    # Fetch market data
    try:
        # Fetch and prepare market data
        data = fetch_market_data(ticker, lookback_years=lookback_period)
        
        # Debug information
        with st.sidebar.expander("Debug Info"):
            st.write("Data Shape:", data.shape)
            st.write("Columns:", data.columns.tolist())
            st.write("Index Type:", type(data.index))
            st.write("Column Types:", data.dtypes.to_dict())
            st.write("First few rows:", data.head())
        
        # Additional validations
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Ensure all required columns exist and are properly named
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                st.error(f"Missing required column: {col}")
                return
        
        # Create strategy instance
        strategy_class = get_strategy_class(strategy_type)
        strategy = strategy_class(params, data)
        
        # Create and run backtest
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=initial_capital,
            commission=commission
        )
        
        results = engine.run_backtest(
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Get and prepare benchmark data
        try:
            benchmark_data = fetch_market_data("SPY", lookback_years=lookback_period)
            benchmark_returns = benchmark_data['Close'].pct_change()
            benchmark_cum = (1 + benchmark_returns).cumprod() * initial_capital
        except Exception as e:
            st.warning(f"Could not fetch benchmark data: {str(e)}")
            benchmark_cum = None
        
        # Display results
        display_strategy_dashboard(results, benchmark_cum)
        
        # Add parameter optimization section
        if st.checkbox("Run Parameter Optimization"):
            st.subheader("Parameter Optimization")
            
            # Select parameter to optimize
            param_to_optimize = st.selectbox(
                "Select Parameter to Optimize",
                list(params.keys())
            )
            
            # Get optimization range
            min_val, max_val = get_parameter_range(param_to_optimize)
            step = (max_val - min_val) / 20
            
            param_range = np.arange(min_val, max_val, step)
            optimization_results = []
            
            with st.spinner("Running optimization..."):
                for value in param_range:
                    test_params = params.copy()
                    test_params[param_to_optimize] = value
                    
                    strategy = strategy_class(test_params, data)
                    engine = BacktestEngine(
                        data=data,
                        strategy=strategy,
                        initial_capital=initial_capital,
                        commission=commission
                    )
                    
                    result = engine.run_backtest(
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    optimization_results.append({
                        'parameter_value': value,
                        'sharpe_ratio': result['performance']['Risk-Adjusted']['sharpe_ratio'],
                        'total_return': result['performance']['Returns']['total_return']
                    })
            
            # Plot optimization results
            opt_df = pd.DataFrame(optimization_results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(
                    opt_df.set_index('parameter_value')['sharpe_ratio']
                )
                st.caption(f"Sharpe Ratio vs {param_to_optimize}")
            
            with col2:
                st.line_chart(
                    opt_df.set_index('parameter_value')['total_return']
                )
                st.caption(f"Total Return vs {param_to_optimize}")
    
    except Exception as e:
        st.error(f"Error running strategy: {str(e)}")

def get_strategy_class(strategy_type: str):
    """Get the appropriate strategy class based on selection"""
    strategy_map = {
        "Triple Moving Average": TripleMAStrategy,
        "Adaptive EMA": AdaptiveEMAStrategy,
        "RSI-MACD Confluence": RSIMACDStrategy,
        "Dynamic Momentum": DynamicMomentumStrategy,
        "Bollinger Breakout": BollingerBreakoutStrategy,
        "Adaptive Volatility": AdaptiveVolatilityStrategy
    }
    return strategy_map[strategy_type]

def get_strategy_parameters(strategy_type: str) -> Dict:
    """Get default parameters for selected strategy"""
    if strategy_type == "Triple Moving Average":
        return {
            'short_window': st.sidebar.slider("Short Window", 1, 20, 5),
            'medium_window': st.sidebar.slider("Medium Window", 10, 50, 21),
            'long_window': st.sidebar.slider("Long Window", 30, 200, 63),
            'min_trend_strength': st.sidebar.slider("Min Trend Strength", 0.0, 0.1, 0.02),
            'position_size': st.sidebar.slider("Position Size", 0.0, 1.0, 0.1)
        }
    elif strategy_type == "Adaptive EMA":
        return {
            'base_period': st.sidebar.slider("Base Period", 5, 50, 21),
            'atr_period': st.sidebar.slider("ATR Period", 5, 30, 14),
            'volatility_factor': st.sidebar.slider("Volatility Factor", 0.5, 5.0, 2.0),
            'adaptation_rate': st.sidebar.slider("Adaptation Rate", 0.01, 0.5, 0.1)
        }
    elif strategy_type == "RSI-MACD Confluence":
        return {
            'rsi_period': st.sidebar.slider("RSI Period", 5, 30, 14),
            'macd_fast': st.sidebar.slider("MACD Fast", 5, 20, 12),
            'macd_slow': st.sidebar.slider("MACD Slow", 15, 40, 26),
            'macd_signal': st.sidebar.slider("MACD Signal", 5, 20, 9),
            'rsi_overbought': st.sidebar.slider("RSI Overbought", 60, 90, 70),
            'rsi_oversold': st.sidebar.slider("RSI Oversold", 10, 40, 30)
        }
    elif strategy_type == "Dynamic Momentum":
        return {
            'rsi_period': st.sidebar.slider("RSI Period", 5, 30, 14),
            'roc_period': st.sidebar.slider("ROC Period", 5, 20, 10),
            'adx_period': st.sidebar.slider("ADX Period", 5, 30, 14),
            'adx_threshold': st.sidebar.slider("ADX Threshold", 15, 40, 25),
            'band_factor': st.sidebar.slider("Band Factor", 0.5, 3.0, 1.5)
        }
    elif strategy_type == "Bollinger Breakout":
        return {
            'bb_period': st.sidebar.slider("BB Period", 5, 50, 20),
            'bb_std': st.sidebar.slider("BB Std Dev", 1.0, 4.0, 2.0),
            'volume_factor': st.sidebar.slider("Volume Factor", 0.5, 3.0, 1.5),
            'atr_period': st.sidebar.slider("ATR Period", 5, 30, 14)
        }
    else:  # Adaptive Volatility
        return {
            'keltner_period': st.sidebar.slider("Keltner Period", 5, 50, 20),
            'atr_factor': st.sidebar.slider("ATR Factor", 0.5, 4.0, 2.0),
            'momentum_period': st.sidebar.slider("Momentum Period", 5, 20, 10),
            'min_volatility': st.sidebar.slider("Min Volatility", 0.005, 0.05, 0.01)
        }

def get_parameter_range(param_name: str) -> tuple:
    """Get optimization range for a parameter"""
    ranges = {
        'short_window': (2, 20),
        'medium_window': (10, 50),
        'long_window': (30, 200),
        'min_trend_strength': (0.01, 0.1),
        'position_size': (0.05, 1.0),
        'base_period': (5, 50),
        'atr_period': (5, 30),
        'volatility_factor': (0.5, 5.0),
        'adaptation_rate': (0.01, 0.5),
        'rsi_period': (5, 30),
        'macd_fast': (5, 20),
        'macd_slow': (15, 40),
        'macd_signal': (5, 20),
        'rsi_overbought': (60, 90),
        'rsi_oversold': (10, 40),
        'roc_period': (5, 20),
        'adx_period': (5, 30),
        'adx_threshold': (15, 40),
        'band_factor': (0.5, 3.0),
        'bb_period': (5, 50),
        'bb_std': (1.0, 4.0),
        'volume_factor': (0.5, 3.0),
        'keltner_period': (5, 50),
        'atr_factor': (0.5, 4.0),
        'momentum_period': (5, 20),
        'min_volatility': (0.005, 0.05)
    }
    return ranges.get(param_name, (0, 1))
