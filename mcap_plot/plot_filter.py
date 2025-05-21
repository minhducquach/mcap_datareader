import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from math import exp, log, e, pi, acos
import os

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.3)

# Path to the folder containing the text files
path = '/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt_new_3'
files = os.listdir(path)  # List all files in the directory

# Function to fit the model to all datasets with the same parameters
def model(params, d, alpha, beta, blinn):
    A, B, C, D, E, F, G = params
    # return 255 * ((A * np.exp(-np.log(2) * (np.arccos(beta)/F)**2) * (alpha * G + D * blinn**E))/ (d**2 + exp(-B)) + C)
    return 1 * (A * alpha / (B + d**2) + C)
    # return (A * (alpha * G + D * blinn**E))/ (d**2 + exp(-B)) + C
    # return A * np.cos(alpha) / (B + d**2) + C

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

# Initial guess for parameters [A, B, C]
# initial_guess = [255, 1, 10, 1, 5, pi/6, 1]
initial_guess = [255, 0, 20, 1, 10, 0.1, 1]
bounds = ([0, -np.inf, 0, 0, 1, 0, 0], [np.inf, np.inf, np.inf, np.inf, 100, pi, 1])

print("Checking model outputs at initial guess...")
for i, data in enumerate(datasets):
    distance = data[0][:, 0]
    cosine_a = data[0][:, 2]
    cosine_b = data[0][:, 3]
    blinn = data[0][:, 4]

    model_output = model(initial_guess, distance, cosine_a, cosine_b, blinn)

    # Identify indices of non-finite outputs
    non_finite_mask = ~np.isfinite(model_output)

    if np.any(non_finite_mask):
        print(f"\nNon-finite model outputs in dataset: {data[1]}")
        for idx in np.where(non_finite_mask)[0]:
            print(f"  Index {idx}:")
            print(f"    Distance: {distance[idx]}")
            print(f"    Cosine Alpha: {cosine_a[idx]}")
            print(f"    Cosine Beta: {cosine_b[idx]}")
            print(f"    Blinn: {blinn[idx]}")
            print(f"    Model output: {model_output[idx]}")


# Perform least squares optimization to fit the same parameters across all datasets
result = least_squares(residuals, initial_guess, bounds=bounds, loss='soft_l1', args=(datasets,))

# Extract optimized parameters
A_fit, B_fit, C_fit, D_fit, E_fit, F_fit, G_fit = result.x
# A_fit, B_fit, C_fit, D_fit, E_fit, F_fit, G_fit = [251.751, -2.759, 25.923, 12.211, 0.800, 3.142, 0.262]
# A_fit, B_fit, C_fit, D_fit, E_fit, F_fit, G_fit = [171.330, 0.106, 64.853, 0, 0, 2.767, 1.805]
print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}, D={D_fit:.3f}, E={E_fit:.3f}, F={F_fit:.3f}, G={G_fit:.3f}")

# Calculate fitted intensity for each dataset using the same parameters
fitted_intensities = []
for data in datasets:
    distance = data[0][:, 0]
    cosine_a = data[0][:, 2]
    cosine_b = data[0][:, 3]
    blinn = data[0][:, 4]
    
    fitted_intensity = model(result.x, distance, cosine_a, cosine_b, blinn)
    fitted_intensities.append(fitted_intensity)

# Create a larger figure for better presentation

def plot_with_fit(ax, x, label, fitted_intensity, intensity, r2, name):
    sns.scatterplot(x=x, y=intensity, ax=ax, color="blue", label="Data", alpha=0.7)
    sns.scatterplot(x=x, y=fitted_intensity, ax=ax, color="orange", label="Fitted", alpha=0.7)
    ax.set_xlabel(label)
    ax.set_ylabel("Intensity")
    # ax.legend(title="Legend")
    ax.set_title(f"{name}: {r2}")

for i in range(6): 
    data = datasets[i]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

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

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.optimize import least_squares
# from sklearn.metrics import r2_score
# from math import exp, log, e, pi
# import os
# from itertools import product

# # Set Seaborn style for better aesthetics
# sns.set_theme(style="whitegrid", palette="muted", font_scale=1.3)

# # Path to the folder containing the text files
# path = '/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt_new_1'
# files = os.listdir(path)

# # Model definition
# def model(params, d, alpha, beta, blinn):
#     A, B, C, D, E, F, G = params
#     return (A * np.exp(-np.log(2) * (np.arccos(beta)/F)**2) * (alpha * G + D * blinn**E))/ (d**2 + exp(-B)) + C

# # Residuals function
# def residuals(params, datasets):
#     total_residuals = []
#     for data in datasets:
#         distance = data[0][:, 0]
#         intensity = data[0][:, 1]
#         cosine_a = data[0][:, 2]
#         cosine_b = data[0][:, 3]
#         blinn = data[0][:, 4]
#         residuals_for_data = model(params, distance, cosine_a, cosine_b, blinn) - intensity
#         total_residuals.extend(residuals_for_data)
#     return total_residuals

# # Load datasets
# datasets = []
# for file in files:
#     file_path = os.path.join(path, file)
#     if file_path.endswith('.txt'):
#         data = np.loadtxt(file_path)
#         datasets.append([data, file])

# # Bounds for parameters
# bounds = ([0, -np.inf, 0, 0, 0, 0, 0], [255, np.inf, 255, np.inf, 100, pi, np.inf])

# # Expanded parameter grid
# A_vals = [255]
# B_vals = [-1, 0, 1]
# C_vals = [10, 50, 100]
# D_vals = [0.1, 0.5, 1.0]
# E_vals = [0, 0.5, 1, 2]
# F_vals = [pi/3, pi/4, pi/6]
# G_vals = [0.1, 0.5, 1.0]

# param_grid = list(product(A_vals, B_vals, C_vals, D_vals, E_vals, F_vals, G_vals))
# print(f"Trying {len(param_grid)} combinations of initial guesses...")

# best_result = None
# best_score = -np.inf
# best_params = None

# # Grid search over all combinations
# for idx, initial_guess in enumerate(param_grid):
#     try:
#         result = least_squares(
#             residuals,
#             initial_guess,
#             bounds=bounds,
#             loss='soft_l1',
#             args=(datasets,),
#             verbose=0
#         )

#         # Compute average R² score across all datasets
#         total_r2 = 0
#         for data in datasets:
#             distance = data[0][:, 0]
#             intensity = data[0][:, 1]
#             cosine_a = data[0][:, 2]
#             cosine_b = data[0][:, 3]
#             blinn = data[0][:, 4]

#             fitted = model(result.x, distance, cosine_a, cosine_b, blinn)
#             total_r2 += r2_score(intensity, fitted)

#         avg_r2 = total_r2 / len(datasets)
#         print(f"[{idx+1}/{len(param_grid)}] Initial: {initial_guess} → Avg R² = {avg_r2:.4f}")
#         if avg_r2 > best_score:
#             best_score = avg_r2
#             best_result = result
#             best_params = initial_guess

#     except Exception as e:
#         print(f"[{idx+1}] Error with initial guess {initial_guess}: {e}")

# # Report best result
# print("\nBest result:")
# print(f"Initial guess: {best_params}")
# print(f"Optimized parameters: {best_result.x}")
# print(f"Best average R²: {best_score:.4f}")

# # Plotting the best fit
# fitted_intensities = []
# for data in datasets:
#     distance = data[0][:, 0]
#     cosine_a = data[0][:, 2]
#     cosine_b = data[0][:, 3]
#     blinn = data[0][:, 4]
#     fitted = model(best_result.x, distance, cosine_a, cosine_b, blinn)
#     fitted_intensities.append(fitted)

# # Plot function
# def plot_with_fit(ax, x, label, fitted_intensity, intensity, r2, name):
#     sns.scatterplot(x=x, y=intensity, ax=ax, color="blue", label="Data", alpha=0.7)
#     sns.scatterplot(x=x, y=fitted_intensity, ax=ax, color="orange", label="Fitted", alpha=0.7)
#     ax.set_xlabel(label)
#     ax.set_ylabel("Intensity")
#     ax.set_title(f"{name}: R²={r2:.3f}")

# # Create plots
# for i in range(min(6, len(datasets))): 
#     data = datasets[i]
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6))

#     distance = data[0][:, 0]
#     cosine_a = data[0][:, 2]
#     cosine_b = data[0][:, 3]
#     intensity = data[0][:, 1]

#     fitted_intensity = fitted_intensities[i]
#     r2 = r2_score(intensity, fitted_intensity)
#     print(f"R² Score for dataset {data[1]}: {r2:.3f}")

#     plot_with_fit(axs[0], distance, "Distance", fitted_intensity, intensity, r2, data[1])
#     plot_with_fit(axs[1], cosine_a, "Cosine Alpha", fitted_intensity, intensity, r2, data[1])
#     plot_with_fit(axs[2], cosine_b, "Cosine Beta", fitted_intensity, intensity, r2, data[1])

#     plt.tight_layout()
#     os.makedirs("imgs", exist_ok=True)
#     plt.savefig(f'imgs/{data[1]}.png')
