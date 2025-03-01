import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
from tensorflow import keras
from joblib import load


model = keras.models.load_model("heston_model.keras")
X_scaler = load("X_scaler.joblib")
kappa, theta, sigma, rho, v0 = 2.0, 0.2, 0.3, -0.5, 0.2
r_fixed = 0.03
q_fixed = 0.01

# Suppose we want strikes from 0.8 to 1.2 * S0
strike_grid = np.linspace(0.8, 1.2, 30)  
# Maturities from 0.1 years to 2 years
maturity_grid = np.linspace(0.1, 2.0, 30)

# Make 2D mesh
K_mesh, T_mesh = np.meshgrid(strike_grid, maturity_grid)
K_flat = K_mesh.ravel()
T_flat = T_mesh.ravel()
# Number of points in the grid
n_points = K_flat.shape[0]

# Build the (n_points, 9) input array
param_array = np.column_stack([
    np.full(n_points, kappa),
    np.full(n_points, theta),
    np.full(n_points, sigma),
    np.full(n_points, rho),
    np.full(n_points, v0),
    np.full(n_points, r_fixed),
    np.full(n_points, q_fixed),
    K_flat,  # strike
    T_flat   # maturity
])
# Scale
param_array_scaled = X_scaler.transform(param_array)

# Predict implied vol
pred_iv_flat = model.predict(param_array_scaled).ravel()

# Reshape back into a 2D surface
pred_iv_surface = pred_iv_flat.reshape(K_mesh.shape)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

# Plot a surface
ax.plot_surface(
    K_mesh,     # X
    T_mesh,     # Y
    pred_iv_surface,  # Z
    cmap='viridis',
    edgecolor='none'
)

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity (years)")
ax.set_zlabel("Predicted Implied Vol")
ax.set_title("NN-Predicted IV Surface")
plt.show()
