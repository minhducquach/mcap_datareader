# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import least_squares
# from sklearn.metrics import r2_score
# from math import exp, acos

# # Load dataset
# # data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160500.txt")
# #data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160544.txt")
# data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt/light_tab_dis_data_lr.txt")

# distance = data[:, 0]
# intensity = data[:, 1]
# cosine_a = data[:, 2]
# cosine_b = data[:, 3]
# # cosine_y = data[:, 4]
# # cosine_p = data[:, 5]
# # cosine_r = data[:, 6]

# def model(params, d, cosa, cosb):
#     A, B, C = params
#     #return (cosa) * (A / ( d**2 + B ) + C)
#     return (A * cosb/ ( d**2 + exp(-B) ) + C) * cosa * 0.6 / 3.14
    
# def residuals(params, d, cosa, cosb, I):
#     return model(params, d, cosa, cosb) - I

# initial_guess = [1, 1, 1]
# result = least_squares(residuals, initial_guess, loss='soft_l1', args=(distance, cosine_a, cosine_b, intensity))

# A_fit, B_fit, C_fit= result.x
# print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# # result.x = [219.328, 122.196, 258.175] # diifax
# # result.x = [827.898, 0.006, 175.181] #toute
# result.x = [ 15863.638, -2.762, 192.814]



# fitted_intensity = model(result.x, distance, cosine_a, cosine_b)

# # fitted_intensity = model_test(distance, cosine_a)

# # Compute R² score
# r2 = r2_score(intensity, fitted_intensity)
# print(f"R² Score: {r2:.3f}")

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1, 3, 1)
# ax1 = fig.add_subplot(1, 3, 2)
# ax2 = fig.add_subplot(1, 3, 3)

# def plot_with_fit(ax, x, label):
#     ax.scatter(x, intensity, label="Data", color="blue", alpha=0.6)
#     ax.scatter(x, fitted_intensity, label="Fitted", color="green", alpha=0.6, marker="x")
#     ax.set_xlabel(label)
#     ax.set_ylabel("Intensity")
#     ax.legend()

# plot_with_fit(ax, distance, "Distance")
# plot_with_fit(ax1, cosine_a, "Cosine_a")
# plot_with_fit(ax2, cosine_b, "Cosine_b")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from math import exp

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid", palette="muted", font_scale=1.3)

# Load dataset
# data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160500.txt")
#data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160544.txt")
data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt/light_tab_dis_data_lr.txt")

distance = data[:, 0]
intensity = data[:, 1]
cosine_a = data[:, 2]
cosine_b = data[:, 3]

def model(params, d, cosa, cosb):
    A, B, C = params
    return (A * cosb / (d**2 + exp(-B)) + C) * cosa * 0.6 / 3.14

def residuals(params, d, cosa, cosb, I):
    return model(params, d, cosa, cosb) - I

initial_guess = [1, 1, 1]
result = least_squares(residuals, initial_guess, loss='soft_l1', args=(distance, cosine_a, cosine_b, intensity))

A_fit, B_fit, C_fit = result.x
print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# result.x = [15863.638, -2.762, 192.814]
fitted_intensity = model(result.x, distance, cosine_a, cosine_b)

# Compute R² score
r2 = r2_score(intensity, fitted_intensity)
print(f"R² Score: {r2:.3f}")

# Create a larger figure for better presentation
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

def plot_with_fit(ax, x, label):
    sns.scatterplot(x=x, y=intensity, ax=ax, color="blue", label="Data", alpha=0.7)
    sns.scatterplot(x=x, y=fitted_intensity, ax=ax, color="orange", label="Fitted",  alpha=0.7)
    ax.set_xlabel(label)
    ax.set_ylabel("Intensity")
    ax.legend(title="Legend")
    ax.set_title(f"{label} vs Intensity")

# Plot each subplot with improved aesthetics
plot_with_fit(axs[0], distance, "Distance")
plot_with_fit(axs[1], cosine_a, "Cosine_a")
plot_with_fit(axs[2], cosine_b, "Cosine_b")

# Adjust the layout to avoid overlap
plt.tight_layout()

# Display the plot
plt.show()
