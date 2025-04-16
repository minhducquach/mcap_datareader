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

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.optimize import least_squares
# from sklearn.metrics import r2_score
# from math import exp, log, e
# import os

# path = '/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt'
# files = os.listdit(path)

# # Set Seaborn style for better aesthetics
# sns.set_theme(style="whitegrid", palette="muted", font_scale=1.3)

# # Load dataset
# # data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160500.txt")
# #data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data_160544.txt")
# for file in files:
#     data = np.loadtxt(file)

# distance = data[:, 0]
# intensity = data[:, 1]
# cosine_a = data[:, 2]
# cosine_b = data[:, 3]

# def model(params, d, cosa, cosb):
#     A, B, C = params
#     # arc = np.arccos(cosb)
#     # print(arc)
#     # return (A * (np.exp(-(0.4*cosb)**3)) / (d**2 + exp(-B)) + C) * cosa * 0.6 / 3.14
#     # print(d, cosa, cosb)
#     # return (A * (2*np.exp(-cosb)-0.5) / (d**2 + exp(-B)) + C) * cosa * 0.6 / 3.14
#     # return (A * (2*np.exp(cosb)-0.5) / (d**2 + exp(-B)) + C) * 1 * 0.6 / 3.14
#     # return (A * (1.1*np.exp(0.5*(arc-1))-0.8) / (d**2 + exp(-B)) + C) * 1 * 0.6 / 3.14
#     # return (A * (2*np.exp(-np.cos(arc-3.14))-0.5) / (d**2 + exp(-B)) + C) * 1 * 0.6 / 3.14
#     return ((A*1/log(e+1)*np.log(1+np.exp(cosb)) / (d**2 + B)) + C) * cosa * 0.6 / 3.14

# def residuals(params, d, cosa, cosb, I):
#     total_residuals = []
    
#     return model(params, d, cosa, cosb) - I

# initial_guess = [1, 1, 1]
# bounds = ([0, -np.inf, 0], [100000, np.inf, 1])
# result = least_squares(residuals, initial_guess, bounds=bounds, loss='soft_l1', args=(distance, cosine_a, cosine_b, intensity))

# A_fit, B_fit, C_fit = result.x
# print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# # result.x = [1339.047, 65.894, 0.517]
# # result.x = [26627.970, -0.930, 0.185]
# fitted_intensity = model(result.x, distance, cosine_a, cosine_b)

# # Compute R² score
# r2 = r2_score(intensity, fitted_intensity)
# print(f"R² Score: {r2:.3f}")

# # Create a larger figure for better presentation
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# def plot_with_fit(ax, x, label):
#     sns.scatterplot(x=x, y=intensity, ax=ax, color="blue", label="Data", alpha=0.7)
#     sns.scatterplot(x=x, y=fitted_intensity, ax=ax, color="orange", label="Fitted",  alpha=0.7)
#     ax.set_xlabel(label)
#     ax.set_ylabel("Intensity")
#     ax.legend(title="Legend")
#     ax.set_title(f"{label} vs Intensity")

# # Plot each subplot with improved aesthetics
# plot_with_fit(axs[0], distance, "Distance")
# plot_with_fit(axs[1], cosine_a, "Cosine Alpha")
# plot_with_fit(axs[2], cosine_b, "Cosine Beta")

# # Adjust the layout to avoid overlap
# plt.tight_layout()

# # Display the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from math import exp, log, e, pi
import os

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.3)

# Path to the folder containing the text files
path = '/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt_new'
files = os.listdir(path)  # List all files in the directory

# Function to fit the model to all datasets with the same parameters
def model(params, d, cosa, cosb, blinn):
    A, B, C, D, E = params
    return ((A / log(e + 1)**0.5 * np.log(1 + np.exp(cosb))**0.5 / (d**2 + exp(-B))) + D * blinn**E + C) * cosa * 0.6 / pi

def residuals(params, datasets):
    total_residuals = []
    
    for data in datasets:
        distance = data[0][:, 0]
        intensity = data[0][:, 1]
        cosine_a = data[0][:, 2]
        cosine_b = data[0][:, 3]
        blinn = data[0][:, 4]
        
        # Calculate residuals for this dataset
        residuals_for_data = model(params, distance, cosine_a, cosine_b, blinn) - intensity
        total_residuals.extend(residuals_for_data)  # Add residuals for this dataset to the total residuals
    
    return total_residuals

# Load all datasets
datasets = []
for file in files:
    file_path = os.path.join(path, file)
    
    # Ensure it's a text file before processing
    if file_path.endswith('.txt'):
        # Load dataset
        data = np.loadtxt(file_path)
        datasets.append([data, file])

# for i, data in enumerate(datasets):
#     dataset_array = data[0]
#     filename = data[1]

#     for idx, row in enumerate(dataset_array):
#         if not np.all(np.isfinite(row)):
#             print(f"[{filename}] Non-finite value at line {idx + 1}: {row}")
        
#         blinn_value = row[4]
#         if blinn_value < 0:
#             print(f"[{filename}] Negative blinn value at line {idx + 1}: {blinn_value}")


# Initial guess for parameters [A, B, C]
initial_guess = [100000, 1, 1, 1, 1]
bounds = ([-np.inf, -4, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

# Perform least squares optimization to fit the same parameters across all datasets
result = least_squares(residuals, initial_guess, bounds=bounds, loss='soft_l1', args=(datasets,))

# Extract optimized parameters
A_fit, B_fit, C_fit, D_fit, E_fit = result.x
print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}, D={D_fit:.3f}, E={E_fit:.3f}")

# Calculate fitted intensity for each dataset using the same parameters
fitted_intensities = []
for data in datasets:
    distance = data[0][:, 0]
    cosine_a = data[0][:, 2]
    cosine_b = data[0][:, 3]
    blinn = data[0][:, 4]
    
    fitted_intensity = model(result.x, distance, cosine_a, cosine_b, blinn)
    fitted_intensities.append(fitted_intensity)

# Compute R² score for each dataset
# for i, data in enumerate(datasets):
#     intensity = data[0][:, 1]
#     fitted_intensity = fitted_intensities[i]
#     r2 = r2_score(intensity, fitted_intensity)
#     print(f"R² Score for dataset {data[1]}: {r2:.3f}")

# Create a larger figure for better presentation
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

def plot_with_fit(ax, x, label, fitted_intensity, intensity, r2, name):
    sns.scatterplot(x=x, y=intensity, ax=ax, color="blue", label="Data", alpha=0.7)
    sns.scatterplot(x=x, y=fitted_intensity, ax=ax, color="orange", label="Fitted", alpha=0.7)
    ax.set_xlabel(label)
    ax.set_ylabel("Intensity")
    ax.legend(title="Legend")
    ax.set_title(f"{name}: {r2}")

for i in range(6):
    data = datasets[i]

    distance = data[0][:, 0]
    cosine_a = data[0][:, 2]
    cosine_b = data[0][:, 3]
    intensity = data[0][:, 1]

    fitted_intensity = fitted_intensities[i]
    r2 = r2_score(intensity, fitted_intensity)
    print(f"R² Score for dataset {data[1]}: {r2:.3f}")

    # Plot for each dataset
    plot_with_fit(axs[0], distance, "Distance", fitted_intensities[i], intensity, r2, data[1])
    plot_with_fit(axs[1], cosine_a, "Cosine Alpha", fitted_intensities[i], intensity, r2, data[1])
    plot_with_fit(axs[2], cosine_b, "Cosine Beta", fitted_intensities[i], intensity, r2, data[1])

    # Adjust the layout to avoid overlap
    plt.tight_layout()
    plt.savefig(f'imgs/{data[1]}.png')

    # # Display the plot
    # plt.show()

