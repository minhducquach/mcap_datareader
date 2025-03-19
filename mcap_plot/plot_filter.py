import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from math import exp

# Load dataset
# data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160500.txt")
#data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160544.txt")
data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt")

distance = data[:, 0]
intensity = data[:, 1]
cosine_a = data[:, 2]
cosine_b = data[:, 3]
# cosine_y = data[:, 4]
# cosine_p = data[:, 5]
# cosine_r = data[:, 6]

def model(params, d, cosa, cosb):
    A, B, C = params
    #return (cosa) * (A / ( d**2 + B ) + C)
    return A * cosa * cosb/ ( d**2 + exp(-B) ) + C
    
def residuals(params, d, cosa, cosb, I):
    return model(params, d, cosa, cosb) - I

initial_guess = [1, 1, 1]
result = least_squares(residuals, initial_guess, loss='soft_l1', args=(distance, cosine_a, cosine_b, intensity))

A_fit, B_fit, C_fit= result.x
print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

fitted_intensity = model(result.x, distance, cosine_a, cosine_b)

# fitted_intensity = model_test(distance, cosine_a)

# Compute R² score
r2 = r2_score(intensity, fitted_intensity)
print(f"R² Score: {r2:.3f}")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2)
ax2 = fig.add_subplot(1, 3, 3)

def plot_with_fit(ax, x, label):
    ax.scatter(x, intensity, label="Data", color="blue", alpha=0.6)
    ax.scatter(x, fitted_intensity, label="Fitted", color="green", alpha=0.6, marker="x")
    ax.set_xlabel(label)
    ax.set_ylabel("Intensity")
    ax.legend()

plot_with_fit(ax, distance, "Distance")
plot_with_fit(ax1, cosine_a, "Cosine_a")
plot_with_fit(ax2, cosine_b, "Cosine_b")

plt.show()
