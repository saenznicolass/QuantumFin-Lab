# modules/models/portfolio/risk_parity.py

import numpy as np
from scipy.optimize import minimize

def calculate_risk_parity_weights(returns, initial_guess=None):
    """
    Calculate risk parity portfolio weights using a basic approach:
    - Minimizes the variance of the risk contributions.

    :param returns: DataFrame of historical returns
    :param initial_guess: optional initial guess for weights
    :return: array of weights
    """
    if returns.empty:
        return []

    cov_matrix = returns.cov().values
    n_assets = cov_matrix.shape[0]

    if initial_guess is None:
        initial_guess = np.ones(n_assets) / n_assets

    def portfolio_volatility(weights):
        return np.sqrt(weights @ cov_matrix @ weights)

    def risk_contribution(weights):
        # marginal_risk is partial derivative of portfolio vol w.r.t. each weight
        port_vol = portfolio_volatility(weights)
        return (weights * (cov_matrix @ weights)) / port_vol

    def risk_parity_objective(weights):
        # We want each asset's risk_contribution to be equal
        rc = risk_contribution(weights)
        return np.sum((rc - rc.mean())**2)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    ]
    # For no short selling, add w >= 0 constraints:
    # but let's allow negative if you want to do short
    # so we won't specify bounds here unless you'd like to.

    result = minimize(
        risk_parity_objective,
        initial_guess,
        constraints=constraints,
        method='SLSQP'
    )

    if not result.success:
        return []

    weights_opt = result.x
    # If result.x has negative values but you don't want short, you'd clamp or re-normalize
    return weights_opt
