# modules/data/data_manager.py

import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def get_data(tickers, start=None, end=None, interval="1d"):
    """
    Fetch historical adjusted close prices for the given tickers using yfinance.
    Cached for performance.

    :param tickers: List or string of ticker symbols
    :param start: Start date (datetime or string)
    :param end: End date (datetime or string)
    :param interval: Data interval (e.g., "1d" for daily)
    :return: DataFrame of adjusted close prices
    """
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval
        )["Adj Close"]
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {tickers}: {e}")
        return pd.DataFrame()
