import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from HestonAnalytical import heston_price, implied_volatility



r = 0.0443
q = 0.008

options_data = pd.read_csv('yfinance_dataset.csv')

S0 = 1

num_samples = 30000
kappa_vals = np.random.uniform(0.5, 5.0, num_samples * 2)
theta_vals = np.random.uniform(0.01, 1.0, num_samples * 2)
sigma_vals = np.random.uniform(0.01, 0.5, num_samples * 2)
rho_vals = np.random.uniform(-0.9, 0.0, num_samples * 2)
v0_vals = np.random.uniform(0.01, 0.5, num_samples * 2)

feller_condition = 2 * kappa_vals * theta_vals > sigma_vals ** 2
valid_indices = np.where(feller_condition)[0][:num_samples]
kappa_samples = kappa_vals[valid_indices]
theta_samples = theta_vals[valid_indices]
sigma_samples = sigma_vals[valid_indices]
rho_samples = rho_vals[valid_indices]
v0_samples = v0_vals[valid_indices]

strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 30)
maturity_range = np.linspace(30 / 365.25, 2, 25)


def compute_option_and_iv(idx):
    kappa, theta, sigma, rho, v0 = (
        kappa_samples[idx],
        theta_samples[idx],
        sigma_samples[idx],
        rho_samples[idx],
        v0_samples[idx],
    )
  
    data_points = []
    for T in maturity_range:
        for K in strike_range:
            price = heston_price(S0, K, r, q, T, kappa, theta, sigma, rho, v0)
            iv = implied_volatility(price, S0, K, r, q, T)
            if iv > 0 and iv < 3.0:
                data_points.append({
                    'kappa': kappa,
                    'theta': theta,
                    'sigma': sigma,
                    'rho': rho,
                    'v0': v0,
                    'strike': K,
                    'maturity': T,
                    'implied_volatility': iv
                })
    return data_points 

results = Parallel(n_jobs=-1)(delayed(compute_option_and_iv)(i) for i in tqdm(range(num_samples), desc="Computing Volatility Grids"))

dataset = [point for sublist in results for point in sublist]

dataset_df = pd.DataFrame(dataset)
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)
dataset_df.to_csv('heston_dataset.csv', index=False)






