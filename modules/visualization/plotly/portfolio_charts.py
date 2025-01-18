import streamlit as st
import plotly.graph_objects as go

from modules.models.portfolio.optimization import compute_efficient_frontier_points

def plot_efficient_frontier(mu, S, weight_bounds, allow_short, optimization_results, risk_free_rate=0.0):
    """Enhanced efficient frontier plot"""
    # Generate frontier points
    frontier_constrained = compute_efficient_frontier_points(mu, S, weight_bounds, risk_free_rate)
    frontier_unconstrained = compute_efficient_frontier_points(
        mu, S,
        ((None, None) if allow_short else (0,1)),
        risk_free_rate
    )

    fig = go.Figure()

    # Plot frontiers with styling
    if frontier_constrained:
        vol_c, ret_c = zip(*frontier_constrained)
        fig.add_trace(go.Scatter(
            x=vol_c, y=ret_c,
            mode='lines',
            name='Constrained Frontier',
            line=dict(color='blue', width=2),
            hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}'
        ))

    if frontier_unconstrained:
        vol_u, ret_u = zip(*frontier_unconstrained)
        fig.add_trace(go.Scatter(
            x=vol_u, y=ret_u,
            mode='lines',
            name='Unconstrained Frontier',
            line=dict(color='black', width=2),
            hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}'
        ))

    # Add optimal points with improved visibility
    perf_c = optimization_results['performance_constrained']
    perf_u = optimization_results['performance_unconstrained']

    fig.add_trace(go.Scatter(
        x=[perf_c[1]], y=[perf_c[0]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='Constrained Optimal',
        hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}'
    ))

    fig.add_trace(go.Scatter(
        x=[perf_u[1]], y=[perf_u[0]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='star'),
        name='Unconstrained Optimal',
        hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}'
    ))

    # Add risk-free rate line
    x_range = [0, max(max(vol_c), max(vol_u)) * 1.1]
    y_values = [risk_free_rate, risk_free_rate]
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_values,
        mode='lines',
        name='Risk-free Rate',
        line=dict(color='gray', dash='dash'),
        hovertemplate='Risk-free rate: %{y:.2%}'
    ))

    fig.update_layout(
        title={
            'text': "Portfolio Efficient Frontier",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Annualized Volatility",
        yaxis_title="Expected Annual Return",
        xaxis=dict(tickformat='.0%'),
        yaxis=dict(tickformat='.0%'),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_simulation_distribution(sim_returns, is_constrained):
    """Plot the distribution of simulated returns"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim_returns,
        nbinsx=50,
        name=f"{'Constrained' if is_constrained else 'Unconstrained'} Returns"
    ))
    fig.update_layout(
        title="Distribution of Simulated Returns",
        xaxis_title="Return",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_simulation_paths(mc_paths, is_constrained):
    """Plot Monte Carlo simulation paths"""
    fig = go.Figure()

    # Plot a sample of paths
    for i in range(min(20, mc_paths.shape[0])):
        fig.add_trace(go.Scatter(
            y=mc_paths[i],
            mode='lines',
            opacity=0.5,
            name=f"Path {i+1}",
            showlegend=False
        ))

    fig.update_layout(
        title=f"{'Constrained' if is_constrained else 'Unconstrained'} Monte Carlo Paths",
        xaxis_title="Time",
        yaxis_title="Portfolio Value"
    )
    st.plotly_chart(fig, use_container_width=True)
