import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional
import traceback
import numpy as np
from plotly.subplots import make_subplots

from ..plotly.technical_charts import (
    plot_strategy_signals,
    plot_equity_curve,
    plot_monthly_returns_heatmap
)
from ...risk.metrics.drawdown import analyze_drawdowns

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
                
                # Group trades by entry/exit pairs
                if 'trade_id' not in trades_df.columns:
                    trades_df['trade_id'] = np.nan
                    current_id = 0
                    for i in range(0, len(trades_df)-1, 2):  # Skip last row if unpaired
                        trades_df.iloc[i:i+2, trades_df.columns.get_loc('trade_id')] = current_id
                        current_id += 1

                # Calculate trade metrics only for complete trades (entry + exit pairs)
                complete_trades = trades_df.dropna(subset=['trade_id']).copy()
                
                if len(complete_trades) > 0:
                    trade_metrics = complete_trades.groupby('trade_id').agg({
                        'entry_date': 'first',
                        'entry_signal': 'first',
                        'entry_price': 'first',
                        'exit_date': 'last',
                        'exit_signal': 'last',
                        'exit_price': 'last',
                        'position_size': 'first',
                        'commission': 'sum',
                        'pnl': 'sum'
                    }).reset_index()
                    
                    trade_metrics['return_pct'] = ((trade_metrics['exit_price'] - trade_metrics['entry_price']) / 
                                                 trade_metrics['entry_price'] * 100 * 
                                                 np.where(trade_metrics['entry_signal'].str.contains('Short'), -1, 1))
                    
                    display_columns = [
                        'entry_date', 'entry_signal', 'entry_price',
                        'exit_date', 'exit_signal', 'exit_price',
                        'position_size', 'commission', 'pnl', 'return_pct'
                    ]

                    st.dataframe(
                        trade_metrics[display_columns]
                        .style.format({
                            'entry_price': '{:.2f}',
                            'exit_price': '{:.2f}',
                            'position_size': '{:.2f}',
                            'commission': '{:.2f}',
                            'pnl': '{:.2f}',
                            'return_pct': '{:.2f}%'
                        })
                    )

                    # Update metadata with corrected trade counts
                    strategy_results['performance']['Trading Stats'].update({
                        'total_trades': len(trade_metrics),
                        'winning_trades': len(trade_metrics[trade_metrics['pnl'] > 0]),
                        'losing_trades': len(trade_metrics[trade_metrics['pnl'] < 0]),
                        'win_rate': len(trade_metrics[trade_metrics['pnl'] > 0]) / len(trade_metrics),
                        'profit_factor': (trade_metrics[trade_metrics['pnl'] > 0]['pnl'].sum() /
                                        abs(trade_metrics[trade_metrics['pnl'] < 0]['pnl'].sum()))
                    })

        with performance_tab:
            st.subheader("Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display key metrics with safe access to nested dictionaries
                metrics = strategy_results.get('performance', {})
                
                # Safely access nested values with get()
                returns_metrics = metrics.get('Returns', {})
                risk_adj_metrics = metrics.get('Risk-Adjusted', {})
                trading_stats = metrics.get('Trading Stats', {})
                
                st.metric(
                    "Total Return", 
                    f"{returns_metrics.get('total_return', 0):.2%}"
                )
                st.metric(
                    "Sharpe Ratio", 
                    f"{risk_adj_metrics.get('sharpe_ratio', 0):.2f}"
                )
                st.metric(
                    "Max Drawdown", 
                    f"{returns_metrics.get('max_drawdown', 0):.2%}"
                )
            
            with col2:
                st.metric(
                    "Win Rate", 
                    f"{trading_stats.get('win_rate', 0):.2%}"
                )
                st.metric(
                    "Profit Factor", 
                    f"{trading_stats.get('profit_factor', 0):.2f}"
                )
                st.metric(
                    "Recovery Factor", 
                    f"{risk_adj_metrics.get('recovery_factor', 0):.2f}"
                )
            
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
            
            # Unified drawdown visualization
            if 'portfolio_value' in strategy_results['results']:
                st.subheader("Drawdown Analysis")
                
                drawdown_info = strategy_results['performance']['Drawdown Analysis']
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Maximum Drawdown", 
                             f"{drawdown_info['summary']['max_drawdown']:.2%}")
                    st.metric("Current Drawdown", 
                             f"{drawdown_info['summary']['current_drawdown']:.2%}")
                
                with col2:
                    st.metric("Total Underwater Periods", 
                             f"{drawdown_info['underwater_periods']['total_periods']}")
                    st.metric("Avg Recovery Time", 
                             f"{drawdown_info['recovery_stats']['avg_recovery_time']:.1f} days")
                
                with col3:
                    st.metric("Max Recovery Time", 
                             f"{drawdown_info['recovery_stats']['max_recovery_time']:.1f} days")
                    st.metric("Total Recoveries", 
                             str(drawdown_info['recovery_stats']['total_recoveries']))
                
                # Display worst drawdowns
                if drawdown_info['worst_drawdowns']:
                    st.subheader("Worst Drawdowns")
                    worst_dd_df = pd.DataFrame(drawdown_info['worst_drawdowns'])
                    st.dataframe(
                        worst_dd_df.style.format({
                            'depth': '{:.2%}',
                            'duration': '{:.0f} days'
                        })
                    )

                # Display drawdown chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
                # ...existing drawdown chart code...

        def format_metric_value(value: float, metric_type: str) -> str:
            """Format metric value based on its type"""
            if metric_type in ['total_return', 'annualized_return', 'max_drawdown', 
                              'annualized_volatility', 'downside_volatility', 'win_rate']:
                return f"{value:.2%}"
            elif metric_type in ['sharpe_ratio', 'sortino_ratio', 'profit_factor', 
                                'recovery_factor']:
                return f"{value:.2f}"
            else:
                return f"{value:.2f}"

        with metrics_tab:
            st.subheader("Risk Metrics")
            
            # Create metrics columns with error handling
            col1, col2, col3, col4 = st.columns(4)
            
            # Helper function to safely display metrics
            def display_section_metrics(container, section_name, metrics_dict):
                with container:
                    st.markdown(f"### {section_name}")
                    if section_name in metrics_dict:
                        for metric, value in metrics_dict[section_name].items():
                            formatted_value = format_metric_value(value, metric)
                            st.metric(metric.replace('_', ' ').title(), formatted_value)
                    else:
                        st.warning(f"No {section_name.lower()} metrics available")

            # Display each section with error handling
            display_section_metrics(col1, "Returns", metrics)
            display_section_metrics(col2, "Risk Metrics", metrics)
            display_section_metrics(col3, "Risk-Adjusted", metrics)
            display_section_metrics(col4, "Trading Stats", metrics)

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
