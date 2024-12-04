import numpy as np

from scipy.stats import norm
import scipy.optimize as optimize
from scipy.interpolate import SmoothBivariateSpline
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import optimize
import logging
from scipy.integrate import trapezoid
from numba import jit
import cmath

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
logging.basicConfig(level=logging.INFO)
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import numpy as np
from scipy import integrate, optimize
from math import log, sqrt, exp, pi
from scipy.stats import norm

import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import time
import datetime

np.random.seed(42) # set a seed for the random generator


# scipy
from scipy.integrate import quad_vec  # quad_vec allows to compute integrals accurately
from scipy.stats import norm
from scipy.stats import qmc # to perform Latin Hypercube Sampling (LHS) 

@jit
def beta_function(u, tau, sigma, rho, kappa):
    return kappa - 1j * u * sigma * rho

@jit
def alpha_hat_function(u):
    return -0.5 * u * (u + 1j)

@jit
def d_function(u, tau, sigma, rho, kappa):

    gamma = 0.5 * sigma**2
    beta = beta_function(u, tau, sigma, rho, kappa)
    alpha_hat = alpha_hat_function(u)

    return np.sqrt(beta**2 - 4 * alpha_hat * gamma)

@jit
def g_function(u, tau, sigma, rho, kappa):

    beta = beta_function(u, tau, sigma, rho, kappa)
    d = d_function(u, tau, sigma, rho, kappa)

    return (beta - d) / (beta + d)

@jit
def A_function(u, tau, theta, sigma, rho, kappa):

    beta = beta_function(u, tau, sigma, rho, kappa)
    d = d_function(u, tau, sigma, rho, kappa)
    g = g_function(u, tau, sigma, rho, kappa)
    common_term = np.exp(-d*tau)
    A_u = (kappa * theta / (sigma**2)) * ((beta-d)*tau - 2*np.log((g*common_term-1) / (g-1)))   

    return A_u

@jit
def B_function(u, tau, sigma, rho, kappa):

    beta = beta_function(u, tau, sigma, rho, kappa)
    d = d_function(u, tau, sigma, rho, kappa)
    g = g_function(u, tau, sigma, rho, kappa)
    common_term = np.exp(-d*tau)
    B_u = ((beta-d) / (sigma**2)) * ((1 - common_term) / (1 - g*common_term))

    return B_u

     
@jit
def heston_charact_funct(u, tau, theta, sigma, rho, kappa, v0):

    beta = beta_function(u, tau, sigma, rho, kappa)      
    d = d_function(u, tau, sigma, rho, kappa)
    g = g_function(u, tau, sigma, rho, kappa)
    common_term = np.exp(-d*tau)
    A = A_function(u, tau, theta, sigma, rho, kappa)
    B = B_function(u, tau, sigma, rho, kappa)

    return np.exp(A + B * v0)


def integral_price(m, tau, theta, sigma, rho, kappa, v0):

    integrand = (lambda u: 
        np.real(np.exp((1j*u + 0.5)*m)*heston_charact_funct(u - 0.5j, tau, theta, sigma, rho, kappa, v0))/(u**2 + 0.25))
    integ, err = quad_vec(integrand, 0, np.inf)

    return integ


def heston_price(S0, K, r, q, T, kappa, theta, sigma, rho, v0):

    m = np.log(S0/K) + r*T
    integ = integral_price(m, T, theta, sigma, rho, kappa, v0)  
    price = S0 - K * np.exp(-r*T) * integ  / np.pi
         
    return price

def black_scholes_call_price(S0, K, r, q, T, sigma):

    if sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call


def implied_volatility(target_price, S0, K, r, q, T, tol=1e-8, max_iterations=100):

    objective = lambda sigma: black_scholes_call_price(S0, K, r, q, T, sigma) - target_price

    vol_lower = 1e-6
    vol_upper = 5.0  

    try:
        implied_vol = optimize.brentq(objective, vol_lower, vol_upper, xtol=tol, maxiter=max_iterations)
    except ValueError:
        implied_vol = np.nan

    return implied_vol

#Calibrated Parameters: [ 2.19553583  0.11861383  0.72169349 -0.89999987  0.07151662]
if __name__ == "__main__":
   
    S0 = 1    
    options_data = pd.read_csv('yfinance_dataset.csv')
    middle_index = len(options_data) // 2
    rows_to_print = 30
    start_index = max(0, middle_index - rows_to_print // 2)
    end_index = min(len(options_data), middle_index + rows_to_print // 2 + 1)
    print(options_data.iloc[start_index:end_index])  
     
    r = 0.0441  
    q = 0.008     
    kappa = 2.19553583  
    theta = 0.11861383 
    sigma = 0.72169349 
    rho = -0.89999987  
    v0 = 0.07151662

    T = 0.573119


    K = 370.0  / 417

 
    price = heston_price(S0, K, r, q, T, kappa, theta, sigma, rho, v0)
    


    iv = implied_volatility(price, S0, K, r, q, T) #0.317054 

    print(iv)

'''
S0 = 412.869995       
   
     
r = 0.0441  
q = 0.008     
kappa = 1.0
theta = 0.2  
sigma = 0.3  
rho = -0.8  
v0 = 0.2


strikes = np.linspace(S0 * 0.8, S0 * 1.2, 10)
maturities = np.linspace(30 / 365, 2, 10)


iv_surface = []
grid_points = []



for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        price = heston_price(S0, K, r, q, T, kappa, theta, sigma, rho, v0)
        iv = implied_volatility(price, S0, K, r, q, T)
        iv_surface.append(iv)
        grid_points.append((K, T))

# Interpolate with SmoothBivariateSpline
strikes, maturities = np.meshgrid(strikes, maturities)
iv_surface = np.array(iv_surface).reshape((len(maturities), len(strikes)))
spline = SmoothBivariateSpline(
    np.ravel(strikes), np.ravel(maturities), np.ravel(iv_surface), s=0.2
)

# Generate smoother data
fine_strikes = np.linspace(S0 * 0.8, S0 * 1.2, 50)
fine_maturities = np.linspace(30/365, 2.0, 50)
fine_strikes, fine_maturities = np.meshgrid(fine_strikes, fine_maturities)
smoothed_iv = spline.ev(fine_strikes, fine_maturities)

# Plot the implied volatility surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(fine_strikes, fine_maturities, smoothed_iv, cmap='viridis')
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Maturity (T)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Smoothed Implied Volatility Surface')
plt.show()
'''