import numpy as np
import matplotlib.pyplot as plt


class MonteCarloPricer:

    def __init__(self, T, r, d, S, sigma, num_paths):

        self.T = T # Time-to-maturity (Years)
        self.r = r # Risk-free rate
        self.d = d # Dividend yield
        self.S = S # Initial spot price
        self.sigma = sigma # Volatility

        self.num_paths = num_paths
    
    
    def ST_gen(self, W = None):

        W = W if W is not None else np.random.standard_normal(size=self.num_paths)

        return self.S * np.exp((self.r - self.d) * self.T - 0.5 * self.sigma ** 2 * self.T + self.sigma * np.sqrt(self.T) * W)
    
    
    def pricer(self, option, W = None):

        ST = self.ST_gen(W = W) if W is not None else self.gen_ST()

        payoffs = option.payoff(ST)
        discounted_payoff = np.exp(-self.r * self.T) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(self.num_paths)

        return discounted_payoff, standard_error
    
    

class Option:
    def __init__(self, strike):
        self.strike = strike

class EuropeanCall(Option):
    def payoff(self, ST):
        return np.maximum(ST - self.strike, 0)
    
class EuropeanPut(Option):
    def payoff(self, ST):
        return np.maximum(self.strike - ST, 0)
    


class DigitalOption(Option):
    def __init__(self, strike, H = 1):
        super().__init__(strike)
        self.H = H

class DigitalCall(DigitalOption):
    def payoff(self, ST):
        return self.H * (ST > self.strike).astype(float)
    
class DigitalPut(DigitalOption):
    def payoff(self, ST):
        return self.H * (ST < self.strike).astype(float)
    


if __name__ == '__main__':
    T = 1  # Time-to-maturity (Years)
    r = 0.05 # Risk-free rate
    d = 0.03 # Dividend yield
    K = 103 # Strike price
    S = 100 # Spot price
    sigma = 0.1 # Volatility       
    num_paths = 100000  # Number of Monte Carlo paths
    H = 20 # Payoff of digital options


    european_call = EuropeanCall(K)
    european_put = EuropeanPut(K)

    digital_call = DigitalCall(K)
    digital_put = DigitalPut(K)


    mc = MonteCarloPricer(T, r, d, S, sigma, num_paths)


    price, error = mc.pricer(european_call)


    print(f"Price: {price:.4f}")
    print(f"Standard Error: {error:.4f}")

    










