import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional
import traceback

from ..plotly.technical_charts import (
    plot_strategy_signals,
    plot_equity_curve,
    plot_monthly_returns_heatmap
)

def display_strategy_dashboard(
    strategy_results: Dict,
    benchmark_data: Optional[pd.Series] = None
) -> None:
    """
    Display a comprehensive strategy dashboard in Streamlit
    
    Parameters:
    -----------
    strategy_results : Dict
        Dictionary containing strategy backtest results
    benchmark_data : pd.Series, optional
        Benchmark data for comparison
    """
    st.header("Strategy Analysis Dashboard")
    
    try:
        # Validate required data
        if not isinstance(strategy_results, dict):
            raise ValueError("Strategy results must be a dictionary")

        required_keys = ['data', 'signals', 'performance']
        missing_keys = [key for key in required_keys if key not in strategy_results]
        if missing_keys:
            raise ValueError(f"Missing required keys in strategy results: {missing_keys}")

        # Extract signals from either 'signals' or 'strategy_results'
        signals = None
        if 'strategy_results' in strategy_results and 'signals' in strategy_results['strategy_results']:
            signals = strategy_results['strategy_results']['signals']
        else:
            signals = strategy_results['signals']

        # Ensure we have valid signals DataFrame
        if not isinstance(signals, pd.DataFrame):
            raise ValueError("Signals must be a pandas DataFrame")

        # Extract signal column from available columns
        signal_columns = ['signal', 'position']
        signal_col = next((col for col in signal_columns if col in signals.columns), None)
        
        if signal_col is None:
            raise ValueError("No signal or position column found in signals DataFrame")

        # Create display signals DataFrame
        display_signals = pd.DataFrame(index=signals.index)
        display_signals['signal'] = signals[signal_col]

        # Create tabs for different analysis sections
        signals_tab, performance_tab, metrics_tab = st.tabs([
            "Trading Signals", "Performance Analysis", "Risk Metrics"
        ])
        
        with signals_tab:
            st.subheader("Strategy Signals and Indicators")
            
            # Plot strategy signals with error handling
            try:
                fig_signals = plot_strategy_signals(
                    data=strategy_results['data'],
                    signals=display_signals,
                    indicators=strategy_results.get('indicators', {})
                )
                st.plotly_chart(fig_signals, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting signals: {str(e)}")

            # Display trade summary if available
            if 'trades' in strategy_results and len(strategy_results['trades']) > 0:
                st.subheader("Trade Summary")
                trades_df = strategy_results['trades']
                st.dataframe(trades_df)
        
        with performance_tab:
            st.subheader("Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display key metrics
                metrics = strategy_results['performance']
                st.metric("Total Return", f"{metrics['Returns']['total_return']:.2%}")
                st.metric("Sharpe Ratio", f"{metrics['Risk-Adjusted']['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown", f"{metrics['Returns']['max_drawdown']:.2%}")
            
            with col2:
                st.metric("Win Rate", f"{metrics['Trading Stats']['win_rate']:.2%}")
                st.metric("Profit Factor", f"{metrics['Trading Stats'].get('profit_factor', 0):.2f}")
                st.metric("Recovery Factor", f"{metrics['Risk-Adjusted'].get('recovery_factor', 0):.2f}")
            
            # Plot equity curve
            fig_equity = plot_equity_curve(
                performance_data=strategy_results['results'],
                benchmark_data=benchmark_data
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Plot monthly returns heatmap
            fig_heatmap = plot_monthly_returns_heatmap(
                strategy_results['results']['returns']
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with metrics_tab:
            st.subheader("Risk Metrics")
            
            # Create metrics columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Returns")
                for metric, value in metrics['Returns'].items():
                    st.metric(metric, f"{value:.2%}")
            
            with col2:
                st.markdown("### Risk Metrics")
                for metric, value in metrics['Risk Metrics'].items():
                    st.metric(metric, f"{value:.2%}")
            
            with col3:
                st.markdown("### Risk-Adjusted")
                for metric, value in metrics['Risk-Adjusted'].items():
                    st.metric(metric, f"{value:.2f}")
            
            # Display rolling metrics if available
            if 'rolling_metrics' in strategy_results:
                st.subheader("Rolling Analysis")
                rolling_metrics = strategy_results['rolling_metrics']
                
                fig = go.Figure()
                for metric, values in rolling_metrics.items():
                    fig.add_trace(go.Scatter(
                        x=values.index,
                        y=values,
                        name=metric
                    ))
                
                fig.update_layout(
                    title="Rolling Performance Metrics",
                    xaxis_title="Date",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying strategy dashboard: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
