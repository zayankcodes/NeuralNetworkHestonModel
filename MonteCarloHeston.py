import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import scipy.optimize as optimize
from Options import EuropeanCall
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
from scipy.optimize import root_scalar
from scipy.interpolate import SmoothBivariateSpline
import pandas as pd

class MonteCarloHeston:

    def __init__(self, S0, v0, r, d, kappa, theta, sigma, rho, N = 100, M = 20000):

        self.S0 = S0 # Initial spot price 
        self.v0 = v0 # Initial variance (3)
        self.r = r # Risk-free rate
        self.d = d # Dividend yield
        self.kappa = kappa # Mean reversion rate of variance (1)
        self.theta = theta # Long-term mean of variance 
        self.sigma = sigma # Volatility of variance (Volatility of volatility) 
        self.rho = rho # Correlation of Wiener processes 

        self.N = N 
        self.M = M

    def path_gen(self, T):

        dt = T / self.N

        var_path = np.zeros(self.N + 1)
        var_path[0] = self.v0 

        path = np.zeros(self.N + 1)
        path[0] = self.S0

        for t in range(1, self.N + 1):

            Z1 = np.random.normal()
            Z = np.random.normal()
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z

            var_path[t] = max(var_path[t-1] + self.kappa * (self.theta - var_path[t-1]) * dt 
                              + self.sigma * np.sqrt(max(var_path[t-1], 0)) * np.sqrt(dt) * Z2, 0)
            
            path[t] = path[t-1] * np.exp((self.r - self.d) * dt - (var_path[t-1] / 2) * dt + np.sqrt(var_path[t-1]) * np.sqrt(dt) * Z1)
        
        return path


    def pricer(self, option, T):
        payoffs = np.zeros(self.M)

        for i in range(self.M):
            path = self.path_gen(T)
            payoffs[i] = option.payoff(path)

        discounted_payoff = np.exp(-self.r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.M)
 
        return discounted_payoff, std_error




def implied_volatility(target_price, S, K, T, r, q, tol=1e-8, max_iterations=100):
    def black_scholes_price(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def objective(sigma):
        return black_scholes_price(S, K, T, r, q, sigma) - target_price

  
    bs_low = black_scholes_price(S, K, T, r, q, tol)
    bs_high = black_scholes_price(S, K, T, r, q, 5)


    if target_price < bs_low:
        return 0
    elif target_price > bs_high:
        target_price = bs_high

    try:
        implied_vol = optimize.brentq(objective, 1e-12, 5, maxiter=max_iterations, xtol=tol)
        return implied_vol
    except (ValueError, RuntimeError) as e:
        logging.info(f"Failed to compute IV for price={target_price}, S={S}, K={K}, T={T}: {e}")
        return np.nan

#Calibrated Parameters: [ 2.35357629  0.12073413  0.12073413 -0.89999996  0.0646868 ]

if __name__ == '__main__':

    S0 = 1       # Spot price
    options_data = pd.read_csv('yfinance_dataset.csv')
    print(options_data)
     
    r = 0.0439    # Risk-free rate
    q = 0.0044     # Dividend yield
    kappa = 2.35357629  # Speed of mean reversion
    theta = 0.12073413 # Long-term variance
    sigma = 0.75386587   # Volatility of variance
    rho = -0.89999996  # Correlation
    v0 = 0.0646868 # Initial variance
    

    K = 0.9346  # Strike price
    T = 0.562283



    N = 100
    M = 60000

    european_call = EuropeanCall(K)

    mch = MonteCarloHeston(S0, v0, r, q, kappa, theta, sigma, rho, N, M)



    price, std_error = mch.pricer(european_call, T)

    print(f'Price: {price}')
    print(f'Standard Error: {std_error}')

    implied_vol_call = implied_volatility(price, S0, K, T, r, q)

    print(f"Implied Volatility (Call): {implied_vol_call:.4f}")


    #0.297965




            

'''


S0 = 412.869995    # Spot price
   
     
r = 0.0441   # Risk-free rate
q = 0.008     # Dividend yield
kappa = 1.0  # Speed of mean reversion
theta = 0.1  # Long-term variance
sigma = 0.5   # Volatility of variance
rho = -0.5  # Correlation
v0 = 0.1 # Initial variance




heston_model = MonteCarloHeston(S0, v0, r, q, kappa, theta, sigma, rho)


strikes = np.linspace(0.8*S0, 1.2*S0, 10)
maturities = np.linspace(30/365.25, 2.0, 10)


iv_surface = []
grid_points = []


for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        european_call = EuropeanCall(K)  
        price, std_error = heston_model.pricer(european_call, T)
        iv = implied_volatility(price, S0, K, T, r, q)
        iv_surface.append(iv)
        grid_points.append((K, T))


strikes, maturities = np.meshgrid(strikes, maturities)
iv_surface = np.array(iv_surface).reshape((len(maturities), len(strikes)))
spline = SmoothBivariateSpline(
    np.ravel(strikes), np.ravel(maturities), np.ravel(iv_surface), s=0.1
)


fine_strikes = np.linspace(0.8*S0, 1.2*S0, 50)
fine_maturities = np.linspace(30/365.25, 2.0, 50)
fine_strikes, fine_maturities = np.meshgrid(fine_strikes, fine_maturities)
smoothed_iv = spline.ev(fine_strikes, fine_maturities)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(fine_strikes, fine_maturities, smoothed_iv, cmap='viridis')
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Maturity (T)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Smoothed Implied Volatility Surface')
plt.show()

'''