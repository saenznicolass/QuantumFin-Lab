import pandas as pd
from sklearn.decomposition import PCA
import yfinance as yf
import statsmodels.api as sm

def perform_factor_analysis(returns: pd.DataFrame, 
                          num_factors: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform PCA-based factor analysis on returns.
    
    :param returns: DataFrame of asset returns
    :param num_factors: number of factors to extract
    :return: (factor_loadings, factor_returns)
    """
    # Standardize returns
    std_returns = (returns - returns.mean()) / returns.std()
    
    # Perform PCA
    pca = PCA(n_components=num_factors)
    factor_returns = pd.DataFrame(
        pca.fit_transform(std_returns),
        index=returns.index,
        columns=[f'Factor_{i+1}' for i in range(num_factors)]
    )
    
    # Get factor loadings
    factor_loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'Factor_{i+1}' for i in range(num_factors)]
    )
    
    return factor_loadings, factor_returns

def calculate_factor_contribution(returns: pd.DataFrame,
                                weights: dict,
                                benchmark: str = '^GSPC') -> pd.DataFrame:
    """
    Calculate factor contributions to portfolio risk.
    
    Parameters and validation added for robustness.
    """
    try:
        # Calculate portfolio returns first
        portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
        
        # Get and align benchmark data
        benchmark_data = yf.download(
            benchmark,
            start=returns.index[0],
            end=returns.index[-1],
            progress=False
        )
        
        benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 30:
            raise ValueError("Insufficient overlapping data points")
            
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Regression
        X = sm.add_constant(benchmark_returns)
        model = sm.OLS(portfolio_returns, X).fit()
        
        # Validate results
        alpha, beta = model.params
        if not (-0.02 <= alpha <= 0.02) or not (0 <= beta <= 2):
            print(f"Warning: Unusual factor values - Alpha: {alpha:.4f}, Beta: {beta:.4f}")
        
        # Format results
        contributions = pd.DataFrame({
            'Factor': ['Alpha', 'Market Beta'],
            'Coefficient': [alpha, beta],
            'T-Stat': model.tvalues,
            'P-Value': model.pvalues
        })
        
        return contributions
        
    except Exception as e:
        print(f"Error in factor analysis: {str(e)}")
        return pd.DataFrame({
            'Factor': ['Error'],
            'Coefficient': [0],
            'T-Stat': [0],
            'P-Value': [1]
        })