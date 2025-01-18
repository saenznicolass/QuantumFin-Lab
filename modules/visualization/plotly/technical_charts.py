import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def plot_strategy_signals(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    indicators: Dict[str, pd.DataFrame],
    title: str = "Strategy Analysis"
) -> go.Figure:
    """Create an interactive plot with price, indicators and signals"""
    try:
        # Validate inputs
        if not isinstance(data, pd.DataFrame) or not isinstance(signals, pd.DataFrame):
            raise ValueError("Data and signals must be pandas DataFrames")

        # Ensure we have the signal column
        signal_col = 'signal'
        if signal_col not in signals.columns and 'position' in signals.columns:
            signal_col = 'position'
        
        if signal_col not in signals.columns:
            raise ValueError("No signal or position column found in signals DataFrame")

        # Create figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(title, 'Volume', 'Indicators')
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add signals with error handling
        long_signals = signals[signals[signal_col] > 0].index
        short_signals = signals[signals[signal_col] < 0].index

        if len(long_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=long_signals,
                    y=data.loc[long_signals, 'Low'] * 0.99,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2)
                    ),
                    name='Long Entry'
                ),
                row=1, col=1
            )

        if len(short_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=short_signals,
                    y=data.loc[short_signals, 'High'] * 1.01,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2)
                    ),
                    name='Short Entry'
                ),
                row=1, col=1
            )

        # Add volume bars
        colors = ['red' if close < open else 'green' 
                  for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )

        # Add indicators
        for name, indicator_data in indicators.items():
            if isinstance(indicator_data, pd.DataFrame):
                for col in indicator_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_data.index,
                            y=indicator_data[col],
                            name=f'{name} {col}',
                            line=dict(width=1)
                        ),
                        row=3, col=1
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data.index,
                        y=indicator_data,
                        name=name,
                        line=dict(width=1)
                    ),
                    row=3, col=1
                )

        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=900,
            showlegend=True,
            template='plotly_white'
        )

        return fig
    except Exception as e:
        raise ValueError(f"Error plotting strategy signals: {str(e)}")

def plot_equity_curve(
    performance_data: pd.DataFrame,
    benchmark_data: Optional[pd.Series] = None,
    title: str = "Strategy Performance"
) -> go.Figure:
    """
    Create an interactive equity curve plot with drawdown
    
    Parameters:
    -----------
    performance_data : pd.DataFrame
        DataFrame containing portfolio value and returns
    benchmark_data : pd.Series, optional
        Benchmark returns for comparison
    title : str
        Plot title
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Drawdown')
    )

    # Plot equity curve
    fig.add_trace(
        go.Scatter(
            x=performance_data.index,
            y=performance_data['portfolio_value'],
            name='Strategy',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    if benchmark_data is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data,
                name='Benchmark',
                line=dict(color='gray', width=2, dash='dash')
            ),
            row=1, col=1
        )

    # Calculate and plot drawdown
    portfolio_value = performance_data['portfolio_value']
    drawdown = (portfolio_value / portfolio_value.cummax() - 1) * 100

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red')
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        yaxis2_tickformat='%',
        hovermode='x unified'
    )

    return fig

def plot_monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """
    Create a monthly returns heatmap
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns series
    """
    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    ).to_frame()
    
    # Create monthly returns matrix
    monthly_returns.index = pd.MultiIndex.from_arrays([
        monthly_returns.index.year,
        monthly_returns.index.month
    ])
    returns_matrix = monthly_returns.unstack() * 100

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=returns_matrix.values,
        x=returns_matrix.columns,
        y=returns_matrix.index,
        colorscale='RdYlGn',
        text=np.round(returns_matrix.values, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(
            title='Return (%)',
            titleside='right'
        )
    ))

    # Update layout
    fig.update_layout(
        title='Monthly Returns (%)',
        xaxis_title='Month',
        yaxis_title='Year',
        height=400,
        template='plotly_white'
    )

    return fig
