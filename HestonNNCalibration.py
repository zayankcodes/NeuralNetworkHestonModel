import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import differential_evolution, minimize
from tensorflow import keras
from joblib import load
from pyDOE import lhs
from HestonAnalytical import heston_price, implied_volatility
import matplotlib.pyplot as plt

options_data = pd.read_csv('yfinance_dataset_MSFT_3_1_2025.csv')
dataset_df = pd.read_csv('heston_dataset.csv')

underlying_price = options_data['underlying_price'].iloc[0]
options_data['strike'] = options_data['strike'] / underlying_price

iv_spline = SmoothBivariateSpline(
    x=options_data['strike'],
    y=options_data['T'],
    z=options_data['implied_volatility'],
    kx=3,
    ky=3,
    s=1.0
)

strike_grid = np.linspace(options_data['strike'].min(), options_data['strike'].max(), 30)
maturity_grid = np.linspace(options_data['T'].min(), options_data['T'].max(), 30)
strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturity_grid)

target_iv_surface = iv_spline.ev(
    strike_mesh.ravel(), maturity_mesh.ravel()
).reshape(len(maturity_grid), len(strike_grid))

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

model = keras.models.load_model("heston_model.keras")
X_scaler = load("X_scaler.joblib")

def calibration_loss_DE(params, model, target_iv_surface, strike_grid, maturity_grid, X_scaler, weights=None):
    kappa, theta, sigma, rho, v0 = params
    if 2 * kappa * theta - sigma**2 < 0:
        return 1e10
    r_fixed = 0.043
    q_fixed = 0.0083
    K, T = np.meshgrid(strike_grid, maturity_grid)
    K_flat = K.ravel()
    T_flat = T.ravel()
    param_sets = np.column_stack([
        np.full_like(K_flat, kappa),
        np.full_like(K_flat, theta),
        np.full_like(K_flat, sigma),
        np.full_like(K_flat, rho),
        np.full_like(K_flat, v0),
        np.full_like(K_flat, r_fixed),
        np.full_like(K_flat, q_fixed),
        K_flat,
        T_flat
    ])
    param_scaled = X_scaler.transform(param_sets)
    pred_iv_flat = model.predict(param_scaled, verbose=0).flatten()
    pred_iv_surface = pred_iv_flat.reshape(K.shape)
    if weights is None:
        loss = np.sum((pred_iv_surface - target_iv_surface)**2)
    else:
        loss = np.sum(weights * (pred_iv_surface - target_iv_surface)**2)
    return loss

bounds_DE = [(0.5, 5.0), (0.01, 1.0), (0.01, 1.0), (-0.9, 0.1), (0.01, 1.0)]

result_DE = differential_evolution(
    calibration_loss_DE,
    bounds=bounds_DE,
    args=(model, target_iv_surface, strike_grid, maturity_grid, X_scaler, weights),
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
        args=(model, target_iv_surface, strike_grid, maturity_grid, X_scaler, weights),
        method='L-BFGS-B',
        bounds=bounds_DE
    )
    if opt_result.success:
        calibrated_params = opt_result.x
    else:
        calibrated_params = x0
else:
    print("\nOptimization failed:", result_DE.message)
    calibrated_params = None

print("Calibrated Params:", calibrated_params)

#calibrated_params = np.array([2.83331201, 0.12974377, 0.85744337, -0.8913334 , 0.0845392])  # (kappa, theta, sigma, rho, v0), weighted, 02/21/2025

r_fixed = 0.043
q_fixed = 0.0083


K_flat = strike_mesh.ravel()
T_flat = maturity_mesh.ravel()
n_points = K_flat.shape[0]
input_array = np.column_stack([
    np.full(n_points, calibrated_params[0]),
    np.full(n_points, calibrated_params[1]),
    np.full(n_points, calibrated_params[2]),
    np.full(n_points, calibrated_params[3]),
    np.full(n_points, calibrated_params[4]),
    np.full(n_points, r_fixed),
    np.full(n_points, q_fixed),
    K_flat,
    T_flat
])
input_scaled = X_scaler.transform(input_array)
nn_iv_flat = model.predict(input_scaled, verbose=0).ravel()
nn_iv_surface = nn_iv_flat.reshape(strike_mesh.shape)

residuals = nn_iv_surface - target_iv_surface


fig = plt.figure(figsize=(20,6))

ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(strike_mesh, maturity_mesh, nn_iv_surface, cmap='viridis', edgecolor='none')
ax1.set_title("NN-Predicted IV Surface")
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
ax3.set_title("Residuals (NN - Market)")
ax3.set_xlabel("Strike")
ax3.set_ylabel("Maturity")
fig.colorbar(contour, ax=ax3, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
