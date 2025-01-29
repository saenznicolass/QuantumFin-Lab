import numpy as np
from multiprocessing import cpu_count
from scipy.optimize import differential_evolution

class CIR:
    def __init__(self, a=0.1, sigma=0.05, theta=0.05, r0=0.03):
        self.a = a
        self.sigma = sigma
        self.theta = theta
        self.r0 = r0

    def bond_price(self, T):
        """Calculate zero-coupon bond price using CIR model."""
        gamma = np.sqrt(self.a**2 + 2*self.sigma**2)
        B = (2*(np.exp(gamma*T) - 1)) / \
            ((gamma + self.a)*(np.exp(gamma*T) - 1) + 2*gamma)
        A = ((2*gamma*np.exp((self.a + gamma)*T/2)) / \
            ((gamma + self.a)*(np.exp(gamma*T) - 1) + 2*gamma))**(2*self.a*self.theta/self.sigma**2)
        return A * np.exp(-B*self.r0)

    def monte_carlo(self, num_paths=10000, num_steps=100, T=5):
        """Generate Monte Carlo paths for CIR rates."""
        dt = T/num_steps
        rates = np.zeros((num_paths, num_steps+1))
        rates[:,0] = self.r0
        for t in range(num_steps):
            sqrt_r = np.sqrt(np.abs(rates[:,t]))
            dw = np.random.normal(0, np.sqrt(dt), num_paths)
            rates[:,t+1] = rates[:,t] + self.a*(self.theta - rates[:,t])*dt + \
                          self.sigma*sqrt_r*dw
            rates[:,t+1] = np.maximum(rates[:,t+1], 0)
        return rates

class CIRCalibrator:
    def __init__(self, instruments, market_prices, r0=0.03):
        self.instruments = instruments
        self.market_prices = market_prices
        self.r0 = r0

    def loss(self, params):
        a, sigma, theta = params
        if 2*a*theta < sigma**2:  # Feller condition
            return np.inf
        try:
            model = CIR(a, sigma, theta, self.r0)
            model_prices = [model.bond_price(inst['T']) for inst in self.instruments]
            return np.sqrt(np.mean((np.array(model_prices) - self.market_prices)**2))
        except:
            return np.inf

    def calibrate(self):
        bounds = [(0.01, 1.0), (0.01, 0.5), (0.001, 0.1)]
        result = differential_evolution(
            self.loss, bounds, strategy='best1bin',
            popsize=20, mutation=(0.5, 1.9),
            recombination=0.7, seed=42,
            workers=cpu_count()
        )
        return CIR(*result.x, self.r0), result.fun
