import numpy as np
from scipy.stats import norm
from multiprocessing import cpu_count
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class HullWhite2F:
    """Two-Factor Hull-White Model"""
    def __init__(self, a1=0.1, sigma1=0.01, a2=0.2, sigma2=0.01, rho=-0.5, r0=0.03):
        self.a1 = a1          # Mean reversion speed for first factor
        self.sigma1 = sigma1  # Volatility for first factor
        self.a2 = a2          # Mean reversion speed for second factor
        self.sigma2 = sigma2  # Volatility for second factor
        self.rho = rho        # Correlation between factors
        self.r0 = r0          # Initial short rate

    def monte_carlo(self, num_paths=10000, num_steps=100, T=5):
        """Generate Monte Carlo paths for two-factor Hull-White rates."""
        dt = T/num_steps
        x = np.zeros((num_paths, num_steps+1))  # First factor
        y = np.zeros((num_paths, num_steps+1))  # Second factor
        rates = np.zeros((num_paths, num_steps+1))
        rates[:,0] = self.r0
        
        # Create correlated Brownian motions
        corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        L = np.linalg.cholesky(corr_matrix)
        
        for t in range(num_steps):
            # Generate correlated random numbers
            z = np.random.normal(size=(num_paths, 2)) @ L.T
            
            # Update factors
            dx = -self.a1 * x[:,t] * dt + self.sigma1 * np.sqrt(dt) * z[:,0]
            dy = -self.a2 * y[:,t] * dt + self.sigma2 * np.sqrt(dt) * z[:,1]
            x[:,t+1] = x[:,t] + dx
            y[:,t+1] = y[:,t] + dy
            
            # Update rates (sum of factors)
            rates[:,t+1] = self.r0 + x[:,t+1] + y[:,t+1]
            
        return rates

    def bond_price(self, T, S):
        """Calculate zero-coupon bond price."""
        B1 = (1 - np.exp(-self.a1*(S-T)))/self.a1
        B2 = (1 - np.exp(-self.a2*(S-T)))/self.a2
        
        var = (self.sigma1**2/(2*self.a1**2))*(S-T + 2/self.a1*np.exp(-self.a1*(S-T)) - 1/(2*self.a1)) + \
              (self.sigma2**2/(2*self.a2**2))*(S-T + 2/self.a2*np.exp(-self.a2*(S-T)) - 1/(2*self.a2)) + \
              2*self.rho*self.sigma1*self.sigma2/(self.a1*self.a2)*(S-T + \
              (np.exp(-self.a1*(S-T))-1)/self.a1 + (np.exp(-self.a2*(S-T))-1)/self.a2 - \
              (np.exp(-(self.a1+self.a2)*(S-T))-1)/(self.a1+self.a2))
        
        return np.exp(-self.r0*(S-T) - B1*x - B2*y - 0.5*var)

class HW2FCalibrator:
    def __init__(self, instruments, market_prices, r0=0.03):
        self.instruments = instruments
        self.market_prices = market_prices
        self.r0 = r0

    def loss(self, params):
        a1, sigma1, a2, sigma2, rho = params
        try:
            model = HullWhite2F(a1, sigma1, a2, sigma2, rho, self.r0)
            model_prices = [np.mean(np.exp(-np.sum(model.monte_carlo(1000, 10, inst['T']), axis=1)*inst['T']))
                           for inst in self.instruments]
            return np.sqrt(np.mean((np.array(model_prices) - self.market_prices)**2))
        except:
            return np.inf

    def calibrate(self):
        bounds = [
            (0.001, 1.0),  # a1
            (0.001, 0.1),  # sigma1
            (0.001, 1.0),  # a2
            (0.001, 0.1),  # sigma2
            (-0.99, 0.99)  # rho
        ]
        result = differential_evolution(
            self.loss, bounds, 
            strategy='best1bin',
            popsize=20, 
            mutation=(0.5, 1.9),
            recombination=0.7, 
            seed=42,
            workers=1,  # Single worker to avoid warnings
            updating='immediate'
        )
        return HullWhite2F(*result.x, self.r0), result.fun
