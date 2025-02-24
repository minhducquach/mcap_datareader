import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Load dataset
data = np.loadtxt("/media/minhducquach/MiduT71/STUDY/SLAM/Internship/mcap_datareader/mcap_plot/light_tab_dis_data_38.txt")
distance = data[:, 0]
intensity = data[:, 1]

# Define the model function
def model(params, d):
    A, B, C = params
    return A / (B + d**2) + C

# Define residual function for least_squares
def residuals(params, d, I):
    return model(params, d) - I

# Initial guess for A, B, C
initial_guess = [1, 1, 1]

# Perform robust fitting using least_squares with Huber loss
result = least_squares(residuals, initial_guess, loss='soft_l1', args=(distance, intensity))

# Extract fitted parameters
A_fit, B_fit, C_fit = result.x
print(f"Estimated parameters (robust fitting): A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# Generate fitted values
fitted_intensity = model(result.x, distance)

# Plot results
plt.scatter(distance, intensity, label="Data", color="blue", alpha=0.6)
plt.plot(distance, fitted_intensity, color="red", label="Fitted Curve")
plt.xlabel("Distance")
plt.ylabel("Intensity")
plt.legend()
plt.show()
