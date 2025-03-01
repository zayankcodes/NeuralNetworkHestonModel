import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import differential_evolution, minimize
from tensorflow import keras
from joblib import load
import matplotlib.pyplot as plt
from pyDOE import lhs
from HestonAnalytical import heston_price, implied_volatility

options_data = pd.read_csv('yfinance_dataset.csv')
underlying_price = options_data['underlying_price'].iloc[0]
options_data['strike'] = options_data['strike'] / underlying_price


iv_spline = SmoothBivariateSpline(
    x=options_data['strike'],
    y=options_data['T'],
    z=options_data['implied_volatility'],
    kx=3, ky=3, s=5.0
)
strike_grid = np.linspace(options_data['strike'].min(), options_data['strike'].max(), 30)
maturity_grid = np.linspace(options_data['T'].min(), options_data['T'].max(), 30)
strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturity_grid)
target_iv_surface = iv_spline.ev(strike_mesh.ravel(), maturity_mesh.ravel()).reshape(len(maturity_grid), len(strike_grid))

vol_oi_data = options_data[['strike', 'T', 'volume', 'open_interest']].dropna().to_numpy()
def find_nearest_volume_oi(K_val, T_val, data_arr):
    dists = np.sqrt((data_arr[:, 0] - K_val)**2 + (data_arr[:, 1] - T_val)**2)
    idx = np.argmin(dists)
    return data_arr[idx, 2], data_arr[idx, 3]
weights = np.zeros_like(strike_mesh, dtype=float)
for i in range(strike_mesh.shape[0]):
    for j in range(strike_mesh.shape[1]):
        K_val = strike_mesh[i, j]
        T_val = maturity_mesh[i, j]
        vol, oi = find_nearest_volume_oi(K_val, T_val, vol_oi_data)
        vol = max(vol, 0.0)
        oi = max(oi, 0.0)
        weights[i, j] = 1.0 + np.log1p(vol) + np.log1p(oi)


def calibration_loss_DE(params, target_iv_surface, strike_grid, maturity_grid):
    kappa, theta, sigma, rho, v0 = params
    if 2 * kappa * theta - sigma**2 < 0:
        return 1e10
    r_fixed = 0.043
    q_fixed = 0.0083
    K, T = np.meshgrid(strike_grid, maturity_grid)
    K_flat = K.ravel()
    T_flat = T.ravel()
    predicted_iv = np.zeros_like(K_flat)
    for idx in range(K_flat.shape[0]):
        K_val = K_flat[idx]
        T_val = T_flat[idx]
        price = heston_price(1.0, K_val, r_fixed, T_val, q_fixed, kappa, theta, sigma, rho, v0)
        iv = implied_volatility(price, 1.0, K_val, r_fixed, T_val, q_fixed)
        predicted_iv[idx] = iv
    pred_iv_surface = predicted_iv.reshape(K.shape)
    loss = np.sum(weights * (pred_iv_surface - target_iv_surface)**2)
    return loss


bounds_DE = [(0.5, 5.0), (0.01, 1.0), (0.01, 1.0), (-0.9, 0.1), (0.01, 1.0)]

result_DE = differential_evolution(
    calibration_loss_DE,
    bounds=bounds_DE,
    args=(target_iv_surface, strike_grid, maturity_grid),
    strategy='best1bin',
    maxiter=1000,
    popsize=15,
    tol=1e-6,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    polish=True,
    disp=True
)


if result_DE.success:
    x0 = result_DE.x
    opt_result = minimize(
        calibration_loss_DE,
        x0,
        args=(target_iv_surface, strike_grid, maturity_grid),
        method='L-BFGS-B',
        bounds=bounds_DE
    )
    calibrated_params = opt_result.x if opt_result.success else x0
else:
    print("\nOptimization failed:", result_DE.message)
    calibrated_params = None

print("Calibrated Params:", calibrated_params)

r_fixed = 0.043
q_fixed = 0.0083
K_flat = strike_mesh.ravel()
T_flat = maturity_mesh.ravel()
n_points = K_flat.shape[0]
predicted_iv = np.zeros(n_points)
for idx in range(n_points):
    K_val = K_flat[idx]
    T_val = T_flat[idx]
    price = heston_price(1.0, K_val, r_fixed, T_val, q_fixed, *calibrated_params)
    iv = implied_volatility(price, 1.0, K_val, r_fixed, T_val, q_fixed)
    predicted_iv[idx] = iv
analytical_iv_surface = predicted_iv.reshape(strike_mesh.shape)
residuals = analytical_iv_surface - target_iv_surface

fig = plt.figure(figsize=(20, 6))
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(strike_mesh, maturity_mesh, analytical_iv_surface, cmap='viridis', edgecolor='none')
ax1.set_title("Analytical IV Surface (Calibrated)")
ax1.set_xlabel("Strike")
ax1.set_ylabel("Maturity")
ax1.set_zlabel("IV")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(strike_mesh, maturity_mesh, target_iv_surface, cmap='plasma', edgecolor='none')
ax2.set_title("Market IV Surface")
ax2.set_xlabel("Strike")
ax2.set_ylabel("Maturity")
ax2.set_zlabel("IV")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
ax3 = fig.add_subplot(133)
contour = ax3.contourf(strike_mesh, maturity_mesh, residuals, levels=50, cmap='coolwarm')
ax3.set_title("Residuals (Analytical - Market)")
ax3.set_xlabel("Strike")
ax3.set_ylabel("Maturity")
fig.colorbar(contour, ax=ax3, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
