import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from math import exp

# Load dataset
data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt")
distance = data[:, 0]
intensity = data[:, 1]
cosine_a = data[:, 2]
# cosine_b = data[:, 3]
# cosine_y = data[:, 4]
# cosine_p = data[:, 5]
# cosine_r = data[:, 6]

def model(params, d, cosa):
    A, B, C = params
    return A * (cosa) / (B + d**2) + C

def model_test(d, cosa):
    return 356.505 * (cosa) / (1.976 + d**2) + 3.047

def residuals(params, d, cosa, I):
    return model(params, d, cosa) - I

initial_guess = [1, 1, 1]
result = least_squares(residuals, initial_guess, loss='soft_l1', args=(distance, cosine_a, intensity))

A_fit, B_fit, C_fit= result.x
print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

fitted_intensity = model(result.x, distance, cosine_a)

# fitted_intensity = model_test(distance, cosine_a)

# Compute R² score
r2 = r2_score(intensity, fitted_intensity)
print(f"R² Score: {r2:.3f}")

fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1, 2, 1)
# ax1 = fig.add_subplot(1, 2, 2)
ax = fig.add_subplot(111, projection='3d')

def plot_with_fit(ax, x, y, label):
    ax.scatter3D(x, y, intensity, label="Data", color="blue", alpha=0.6)
    # ax.scatter(x, intensity, label="Data", color="blue", alpha=0.6)
    # ax.scatter(x, fitted_intensity, label="Fitted", color="green", alpha=0.6, marker="x")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Cosine")
    ax.set_zlabel("Intensity")

    ax.legend()

# plot_with_fit(ax, distance, "Distance")
# plot_with_fit(ax1, cosine_a, "Cosine_a")
plot_with_fit(ax, distance, cosine_a, "plot")

plt.show()
