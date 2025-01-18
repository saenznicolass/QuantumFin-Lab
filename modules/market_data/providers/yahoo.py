import yfinance as yf
import pandas as pd

def fetch_market_data(ticker: str, lookback_years: int = 2) -> pd.DataFrame:
    """Fetch historical market data from Yahoo Finance."""
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    
    try:
        # Fetch data with progress=False to avoid streamlit issues
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Convert MultiIndex columns to single level
        if isinstance(data.columns, pd.MultiIndex):
            # Ensure we take only the first level and maintain column order
            columns = []
            for col in data.columns:
                if isinstance(col, tuple):
                    columns.append(col[0])
                else:
                    columns.append(col)
            data.columns = columns

        # Ensure proper column names
        column_mapping = {
            'Adj Close': 'Close',  # Use Adjusted Close as Close
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Volume': 'Volume'
        }
        
        # Filter and rename columns
        valid_columns = [col for col in data.columns if col in column_mapping]
        data = data[valid_columns].rename(columns=column_mapping)
        
        # Final validation
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Found: {data.columns.tolist()}")
        
        # Ensure DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        return data[required_columns]  # Return only the required columns in specific order
        
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")