import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import differential_evolution, minimize
from tensorflow import keras
from joblib import load
from pyDOE import lhs
import matplotlib.pyplot as plt


options_data = pd.read_csv('yfinance_dataset_MSFT_2_27_2025.csv')
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

strike_array = np.linspace(options_data['strike'].min(), options_data['strike'].max(), 30)


model = keras.models.load_model("heston_model.keras")
X_scaler = load("X_scaler.joblib")


def calibration_loss_single_maturity(params, model, target_iv, strike_array, chosen_maturity, X_scaler, weights=None):
    kappa, theta, sigma, rho, v0 = params

    if 2 * kappa * theta - sigma**2 < 0:
        return 1e10
    r_fixed = 0.043
    q_fixed = 0.0083

    T_fixed = np.full_like(strike_array, chosen_maturity)

    param_sets = np.column_stack([
        np.full_like(strike_array, kappa),
        np.full_like(strike_array, theta),
        np.full_like(strike_array, sigma),
        np.full_like(strike_array, rho),
        np.full_like(strike_array, v0),
        np.full_like(strike_array, r_fixed),
        np.full_like(strike_array, q_fixed),
        strike_array,
        T_fixed
    ])
    param_scaled = X_scaler.transform(param_sets)
    pred_iv = model.predict(param_scaled, verbose=0).flatten()
    if weights is None:
        loss = np.sum((pred_iv - target_iv)**2)
    else:
        loss = np.sum(weights * (pred_iv - target_iv)**2)
    return loss

bounds_DE = [(0.5, 5.0), (0.01, 1.0), (0.01, 1.0), (-0.9, 0.1), (0.01, 1.0)]
maturities_data = []
params = []

for T in options_data['T'].unique():

    maturities_data.append(T)
    target_iv_single = iv_spline.ev(strike_array, np.full_like(strike_array, T))
    result_DE = differential_evolution(
        calibration_loss_single_maturity,
        bounds=bounds_DE,
        args=(model, target_iv_single, strike_array, T, X_scaler),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        disp=True
    )

    if result_DE.success:
        x0 = result_DE.x

        opt_result = minimize(
            calibration_loss_single_maturity, 
            x0,
            args=(model, target_iv_single, strike_array, T, X_scaler),
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

    print("Calibrated Params for maturity", T, ":", calibrated_params)

   
    params.append(calibrated_params)    

        

print(params)
print(maturities_data)
r_fixed = 0.043
q_fixed = 0.0083
n_points = strike_array.shape[0]
T_fixed = np.full(n_points, T)
input_array = np.column_stack([
    np.full(n_points, calibrated_params[0]),
    np.full(n_points, calibrated_params[1]),
    np.full(n_points, calibrated_params[2]),
    np.full(n_points, calibrated_params[3]),
    np.full(n_points, calibrated_params[4]),
    np.full(n_points, r_fixed),
    np.full(n_points, q_fixed),
    strike_array,
    T_fixed
])
input_scaled = X_scaler.transform(input_array)
nn_iv = model.predict(input_scaled, verbose=0).flatten()

plt.figure(figsize=(10,6))
plt.plot(strike_array, target_iv_single, label='Market IV', marker='o')
plt.plot(strike_array, nn_iv, label='NN Predicted IV', marker='x')
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title("Calibration at Maturity = {}".format(T))
plt.legend()
plt.show()
