import numpy as np
from multiprocessing import cpu_count
from scipy.optimize import differential_evolution

class G2PP:
    def __init__(self, a=0.1, sigma=0.01, b=0.2, eta=0.01, rho=-0.7, r0=0.03):
        self.a = a
        self.sigma = sigma
        self.b = b
        self.eta = eta
        self.rho = rho
        self.r0 = r0

    def monte_carlo(self, num_paths=10000, num_steps=100, T=5):
        """Generate Monte Carlo paths for G2++ rates."""
        dt = T/num_steps
        x = np.zeros((num_paths, num_steps+1))
        y = np.zeros((num_paths, num_steps+1))
        corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        L = np.linalg.cholesky(corr_matrix)
        
        for t in range(num_steps):
            dw = np.random.normal(size=(num_paths, 2)) @ L.T * np.sqrt(dt)
            x[:,t+1] = x[:,t] - self.a*x[:,t]*dt + self.sigma*dw[:,0]
            y[:,t+1] = y[:,t] - self.b*y[:,t]*dt + self.eta*dw[:,1]
        return self.r0 + x + y

class G2PPCalibrator:
    def __init__(self, instruments, market_prices, r0=0.03):
        self.instruments = instruments
        self.market_prices = market_prices
        self.r0 = r0

    def loss(self, params):
        a, sigma, b, eta, rho = params
        try:
            model = G2PP(a, sigma, b, eta, rho, self.r0)
            model_prices = [np.mean(np.exp(-np.sum(model.monte_carlo(1000, 10, inst['T']), axis=1)*inst['T']))
                           for inst in self.instruments]
            return np.sqrt(np.mean((np.array(model_prices) - self.market_prices)**2))
        except:
            return np.inf

    def calibrate(self):
        bounds = [(0.001, 1), (0.001, 0.1),
                 (0.001, 1), (0.001, 0.1),
                 (-0.99, 0.99)]
        result = differential_evolution(
            self.loss, bounds, strategy='best1bin',
            popsize=30, mutation=(0.5, 1.9),
            recombination=0.7, seed=42,
            workers=cpu_count()
        )
        return G2PP(*result.x, self.r0), result.fun
