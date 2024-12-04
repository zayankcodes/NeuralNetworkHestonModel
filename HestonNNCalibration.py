import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import brentq, minimize, least_squares, differential_evolution
from scipy.interpolate import SmoothBivariateSpline
from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt
from joblib import dump, load

X_scaler = load("X_scaler.joblib")
model = keras.models.load_model("heston_model.keras")

dataset_df = pd.read_csv('heston_dataset.csv')
options_data = pd.read_csv('yfinance_dataset.csv')


middle_index = len(options_data) // 2
rows_to_print = 30
start_index = max(0, middle_index - rows_to_print // 2)
end_index = min(len(options_data), middle_index + rows_to_print // 2 + 1)
print(options_data.iloc[start_index:end_index])

underlying_price = options_data['underlying_price'].iloc[0]
options_data['strike'] = options_data['strike'] / underlying_price

features = ['kappa', 'theta', 'sigma', 'rho', 'v0', 'strike', 'maturity']
target = 'implied_volatility'
X = dataset_df[features].values
y = dataset_df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = X_scaler.transform(X_test)

strike_grid = np.linspace(options_data['strike'].min(), options_data['strike'].max(), 30)
maturity_grid = np.linspace(options_data['T'].min(), options_data['T'].max(), 30)  
strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturity_grid)

spline = SmoothBivariateSpline(
    options_data['strike'],
    options_data['T'],  
    options_data['implied_volatility'], 
    kx=3, ky=3
)

target_iv_surface = spline.ev(strike_mesh.ravel(), maturity_mesh.ravel()).reshape(len(maturity_grid), len(strike_grid))


def calibration_loss_DE(params, model, target_iv_surface, strike_grid, maturity_grid, X_scaler):

    kappa, theta, sigma, rho, v0 = params
    if 2 * kappa * theta - sigma**2 < 0:
        return 1e10  
    
    K, T = np.meshgrid(strike_grid, maturity_grid)
    K_flat = K.ravel()
    T_flat = T.ravel()

    param_sets = np.column_stack([
        np.full_like(K_flat, kappa),
        np.full_like(K_flat, theta),
        np.full_like(K_flat, sigma),
        np.full_like(K_flat, rho),
        np.full_like(K_flat, v0),
        K_flat,
        T_flat
    ])
    
    param_scaled = X_scaler.transform(param_sets)
    pred_iv_flat = model.predict(param_scaled, verbose=0).flatten()
    pred_iv_surface = pred_iv_flat.reshape(K.shape)
    loss = np.sum((pred_iv_surface - target_iv_surface) ** 2)
    
    return loss

bounds_DE = [(0.5, 5.0),    # kappa
            (0.01, 1.0),   # theta
            (0.01, 1.0),   # sigma
            (-0.9, 0.0),   # rho
            (0.01, 1.0)]   # v0

result_DE = differential_evolution(
    calibration_loss_DE,
    bounds=bounds_DE,
    args=(model, target_iv_surface, strike_grid, maturity_grid, X_scaler),
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
    print("\nOptimization succeeded!")
    calibrated_params = result_DE.x
    print("Calibrated Parameters:", calibrated_params)
    print(f"Calibration Loss: {result_DE.fun:.4f}\n")
else:
    print("\nOptimization failed:", result_DE.message)
    calibrated_params = None



