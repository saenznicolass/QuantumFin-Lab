import numpy as np
from scipy.stats import norm
from multiprocessing import cpu_count
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class HullWhite:
    def __init__(self, a=0.1, sigma=0.01, r0=0.03):
        self.a = a
        self.sigma = sigma
        self.r0 = r0

    def bond_option_price(self, T, S, K):
        """Calculate bond option price using Hull-White model."""
        B = (1 - np.exp(-self.a*(S-T)))/self.a
        P_T = np.exp(-self.r0*T)
        P_S = np.exp(-self.r0*S)
        sigma_p = self.sigma * np.sqrt((1 - np.exp(-2*self.a*T))/(2*self.a)) * B
        d1 = (np.log(P_S/(K*P_T)) + 0.5*sigma_p**2) / sigma_p
        d2 = d1 - sigma_p
        return P_S * norm.cdf(d1) - K * P_T * norm.cdf(d2)

    def monte_carlo(self, num_paths=10000, num_steps=100, T=5):
        """Generate Monte Carlo paths for Hull-White rates."""
        dt = T/num_steps
        rates = np.zeros((num_paths, num_steps+1))
        rates[:,0] = self.r0
        for t in range(num_steps):
            dr = -self.a*rates[:,t]*dt + self.sigma*np.sqrt(dt)*np.random.normal(size=num_paths)
            rates[:,t+1] = rates[:,t] + dr
        return rates

class HWCalibrator:
    def __init__(self, instruments, market_prices, r0=0.03):
        self.instruments = instruments
        self.market_prices = market_prices
        self.r0 = r0

    def loss(self, params):
        a, sigma = params
        try:
            model = HullWhite(a, sigma, self.r0)
            model_prices = [model.bond_option_price(**inst) for inst in self.instruments]
            return np.sqrt(np.mean((np.array(model_prices) - self.market_prices)**2))
        except:
            return np.inf

    def calibrate(self):
        bounds = [(0.001, 1.0), (0.001, 0.1)]
        result = differential_evolution(
            self.loss, bounds, 
            strategy='best1bin',
            popsize=15, 
            mutation=(0.5, 1.9),
            recombination=0.7, 
            seed=42,
            workers=1,  # Changed from cpu_count() to avoid warning
            updating='immediate'
        )
        return HullWhite(*result.x, self.r0), result.fun
