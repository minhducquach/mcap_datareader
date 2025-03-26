import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

# Load dataset
data = np.loadtxt("/media/minhducquach/MiduT71/STUDY/SLAM/Internship/mcap_datareader/mcap_plot/light_tab_dis_data_ypr.txt")
distance = data[:, 0]
intensity = data[:, 1]
cosine_a = data[:, 2]
cosine_b = data[:, 3]
cosine_y = data[:, 4]
cosine_p = data[:, 5]
cosine_r = data[:, 6]

# Define function f with learnable weights
def f(w, cosa, cosb, cosy, cosp, cosr):
    # return w[0] * cosa + w[1] * cosb + w[2] * cosy + w[3] * cosp + w[4] * cosr
    return cosa * cosa + 1000*cosb * cosb

# Define model function
def model(params, w, d, cosa, cosb, cosy, cosp, cosr):
    A, B, C = params
    return A * f(w, cosa, cosb, cosy, cosp, cosr) / (B + d**2) + C

# Define residual function
def residuals(params, w, d, cosa, cosb, cosy, cosp, cosr, I):
    return model(params, w, d, cosa, cosb, cosy, cosp, cosr) - I

# Initial guess for A, B, C and weights w
initial_params = [1, 1, 1]
initial_w = [1, 1, 1, 1, 1]

# Optimize A, B, C and weights
result = least_squares(residuals, initial_params, loss='linear', args=(initial_w, distance, cosine_a, cosine_b, cosine_y, cosine_p, cosine_r, intensity))

# Extract fitted parameters
A_fit, B_fit, C_fit = result.x
print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# Fit weights separately for f
def residuals_w(w):
    return f(w, cosine_a, cosine_b, cosine_y, cosine_p, cosine_r) - intensity

w_result = least_squares(residuals_w, initial_w)
w_fit = w_result.x
print(f"Estimated function f coefficients: w1={w_fit[0]:.3f}, w2={w_fit[1]:.3f}, w3={w_fit[2]:.3f}, w4={w_fit[3]:.3f}, w5={w_fit[4]:.3f}")

# Generate fitted values
fitted_intensity = model(result.x, w_fit, distance, cosine_a, cosine_b, cosine_y, cosine_p, cosine_r)

# Compute R² score
r2 = r2_score(intensity, fitted_intensity)
print(f"R² Score: {r2:.3f}")
