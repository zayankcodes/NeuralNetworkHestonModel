
from math import log, sqrt, exp, pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate
from scipy.stats import norm, qmc
from scipy.integrate import quad, trapezoid, quad_vec
from scipy.optimize import brentq
from scipy.interpolate import SmoothBivariateSpline
from numba import jit


np.random.seed(42)

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

def heston_price(S0, K, r, T, q, kappa, theta, sigma, rho, v0):

    m = np.log(S0/K) + (r - q) * T
    integ = integral_price(m, T, theta, sigma, rho, kappa, v0)  
    price = S0 * np.exp(-q * T) - K * np.exp(-r * T) * integ  / np.pi
         
    return price

def black_scholes_call_price(S0, K, r, T, q, sigma):

    if sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    
    d1 = (log(S0 / K) + ((r-q) + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call

def implied_volatility(target_price, S0, K, r, T, q, tol=1e-8, max_iterations=100):

    objective = lambda sigma: black_scholes_call_price(S0, K, r, T, q, sigma) - target_price

    vol_lower = 1e-6
    vol_upper = 5.0  

    try:
        implied_vol = optimize.brentq(objective, vol_lower, vol_upper, xtol=tol, maxiter=max_iterations)
    except ValueError:
        implied_vol = np.nan

    return implied_vol



print(implied_volatility(20, 200, 190, 0.04, 1, 0))









