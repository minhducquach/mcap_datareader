# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.optimize import least_squares
# from sklearn.metrics import r2_score
# from math import exp, log, e, pi, acos
# import os

# # Set Seaborn style for better aesthetics
# sns.set_theme(style="whitegrid", palette="muted", font_scale=1.3)

# # Path to the folder containing the text files
# path = '/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt_new_3'
# files = os.listdir(path)  # List all files in the directory

# # Function to fit the model to all datasets with the same parameters
# def model(params, d, alpha, beta, blinn):
#     A, B, C, D, E, F, G = params
#     return (A * np.exp(-np.log(2) * (np.arccos(beta)/F)**2) * alpha)/ (d**2 + exp(-B)) + C
#     # return (A * (alpha * G + D * blinn**E))/ (d**2 + exp(-B)) + C
#     # return A * np.cos(alpha) / (B + d**2) + C

# def residuals(params, datasets):
#     total_residuals = []
    
#     for data in datasets:
#         distance = data[0][:, 0]
#         intensity = data[0][:, 1] / 255
#         cosine_a = data[0][:, 2]
#         cosine_b = data[0][:, 3]
#         blinn = data[0][:, 4]

#         # print(intensity)
        
#         # Calculate residuals for this dataset
#         residuals_for_data = model(params, distance, cosine_a, cosine_b, blinn) - intensity
#         total_residuals.extend(residuals_for_data)  # Add residuals for this dataset to the total residuals
    
#     return total_residuals

# # Load all datasets
# datasets = []
# for file in files:
#     file_path = os.path.join(path, file)
    
#     # Ensure it's a text file before processing
#     if file_path.endswith('.txt'):
#         # Load dataset
#         data = np.loadtxt(file_path)
#         datasets.append([data, file])

# # Initial guess for parameters [A, B, C]
# # initial_guess = [255, 1, 10, 1, 5, pi/6, 1]
# # initial_guess = [255, 0, 20, 1, 10, 0.1, 4]
# initial_guess = [4.881, -1.993, 0.374, 0.000, 0.000, 2.403, 0.000]
# bounds = ([0, -np.inf, 0, 0, 0, 0, 0], [255, np.inf, 255, np.inf, 100, pi, np.inf])

# print("Checking model outputs at initial guess...")
# for i, data in enumerate(datasets):
#     distance = data[0][:, 0]
#     cosine_a = data[0][:, 2]
#     cosine_b = data[0][:, 3]
#     blinn = data[0][:, 4]

#     model_output = model(initial_guess, distance, cosine_a, cosine_b, blinn)

#     # Identify indices of non-finite outputs
#     non_finite_mask = ~np.isfinite(model_output)

#     if np.any(non_finite_mask):
#         print(f"\nNon-finite model outputs in dataset: {data[1]}")
#         for idx in np.where(non_finite_mask)[0]:
#             print(f"  Index {idx}:")
#             print(f"    Distance: {distance[idx]}")
#             print(f"    Cosine Alpha: {cosine_a[idx]}")
#             print(f"    Cosine Beta: {cosine_b[idx]}")
#             print(f"    Blinn: {blinn[idx]}")
#             print(f"    Model output: {model_output[idx]}")


# # Perform least squares optimization to fit the same parameters across all datasets
# result = least_squares(residuals, initial_guess, bounds=bounds, loss='soft_l1', args=(datasets,))

# # Extract optimized parameters
# A_fit, B_fit, C_fit, D_fit, E_fit, F_fit, G_fit = result.x
# # A_fit, B_fit, C_fit, D_fit, E_fit, F_fit, G_fit = [232.615/255*5.351, -1.993, 95.258/255, 0, 0, 2.403, 5.351]
# # A_fit, B_fit, C_fit, D_fit, E_fit, F_fit, G_fit = [171.330, 0.106, 64.853, 0, 0, 2.767, 1.805]
# print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}, D={D_fit:.3f}, E={E_fit:.3f}, F={F_fit:.3f}, G={G_fit:.3f}")

# # Calculate fitted intensity for each dataset using the same parameters
# fitted_intensities = []
# for data in datasets:
#     distance = data[0][:, 0]
#     cosine_a = data[0][:, 2]
#     cosine_b = data[0][:, 3]
#     blinn = data[0][:, 4]
    
#     fitted_intensity = model(result.x, distance, cosine_a, cosine_b, blinn) * 255
#     fitted_intensities.append(fitted_intensity)

# # Create a larger figure for better presentation

# def plot_with_fit(ax, x, label, fitted_intensity, intensity, r2, name):
#     sns.scatterplot(x=x, y=intensity, ax=ax, color="blue", label="Data", alpha=0.7)
#     sns.scatterplot(x=x, y=fitted_intensity, ax=ax, color="orange", label="Fitted", alpha=0.7)
#     ax.set_xlabel(label)
#     ax.set_ylabel("Intensity")
#     # ax.legend(title="Legend")
#     ax.set_title(f"{name}: {r2}")

# for i in range(6): 
#     data = datasets[i]
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6))

#     distance = data[0][:, 0]
#     cosine_a = data[0][:, 2]
#     cosine_b = data[0][:, 3]
#     intensity = data[0][:, 1]

#     fitted_intensity = fitted_intensities[i]
#     r2 = r2_score(intensity, fitted_intensity)
#     print(f"R² Score for dataset {data[1]}: {r2:.3f}")

#     # Plot for each dataset
#     plot_with_fit(axs[0], distance, "Distance", fitted_intensities[i], intensity, r2, data[1])
#     plot_with_fit(axs[1], cosine_a, "Cosine Alpha", fitted_intensities[i], intensity, r2, data[1])
#     plot_with_fit(axs[2], cosine_b, "Cosine Beta", fitted_intensities[i], intensity, r2, data[1])

#     # Adjust the layout to avoid overlap
#     plt.tight_layout()
#     plt.savefig(f'imgs/{data[1]}.png')

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

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.optimize import least_squares
# from sklearn.metrics import r2_score, root_mean_squared_error
# from math import exp, log, e, pi, acos # math.exp, log, acos not used with numpy arrays directly
# import os

# # Set Seaborn style for better aesthetics
# sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1) # Adjusted font_scale slightly

# # Path to the folder containing the text files
# path = '/media/minhducquach/MiduT73/STUDY/SLAM/Internship/src/mcap_datareader/mcap_plot/txt_new_3'
# # List only .txt files in the directory
# files_in_dir = os.listdir(path)
# txt_files = [f for f in files_in_dir if f.endswith('.txt')]

# # --- Model Definition ---
# def model(params, d, alpha, beta, blinn, epsilon=1e-9): # Added epsilon for F
#     A, B, C, D, E, F, G = params # D, E, G are fitted but not used in this formula

#     # Clip beta to be strictly within [-1, 1] to avoid NaN from arccos
#     beta_clipped = np.clip(beta, -1.0, 1.0)

#     # Angular term with protection against F being zero or too small
#     if abs(F) < epsilon:
#         angular_term = 1.0 # Or 0.0, or handle as error, depending on physical meaning
#                            # If F=0 means no angular falloff, 1.0 is reasonable.
#                            # If F=0 means extremely narrow beam, could be 0 unless beta=1.
#                            # For now, assume 1.0 meaning maximal angular component if F is effectively zero.
#     else:
#         angular_term = np.exp(-np.log(2) * (np.arccos(beta_clipped) / F)**2)

#     # Denominator term: d^2 + exp(-B)
#     # Ensure B doesn't cause overflow/underflow with np.exp if it gets too large/small
#     # least_squares bounds should help manage B
#     denominator = d**2 + np.exp(-B) # Use np.exp for array B

#     # Avoid division by zero in the main term if denominator is zero
#     # (unlikely with d^2 and exp(-B) being non-negative, unless d=0 and exp(-B)=0)
#     if np.any(np.abs(denominator) < epsilon): # Check for near-zero denominator
#         # Handle this case: return a very large number, or zero, or based on context
#         # For simplicity, if this happens, we might return C (ambient) or 0 if it's problematic
#         # This situation needs careful thought based on the physics of the model
#         # For now, let's assume denominator won't be problematic due to d^2 >= 0 and exp(-B) > 0
#         pass

#     return (A * angular_term * alpha) / denominator + C

# # --- Residuals Function ---
# def residuals(params, datasets):
#     total_residuals = []
#     for data_info in datasets:
#         data_array = data_info[0]
#         distance = data_array[:, 0]
#         intensity_raw = data_array[:, 1]
#         cosine_a = data_array[:, 2]
#         cosine_b = data_array[:, 3]
#         blinn = data_array[:, 4] # blinn is passed but not used in the current active model

#         intensity_normalized = intensity_raw / 255.0

#         model_output = model(params, distance, cosine_a, cosine_b, blinn)
#         residuals_for_data = model_output - intensity_normalized
#         total_residuals.extend(residuals_for_data)
#     return total_residuals

# # --- Load Datasets ---
# datasets = []
# for filename in txt_files:
#     file_path = os.path.join(path, filename)
#     try:
#         data = np.loadtxt(file_path)
#         if data.shape[1] < 5:
#             print(f"Warning: File {filename} has fewer than 5 columns. Skipping.")
#             continue
#         datasets.append([data, filename]) # Store data array and filename
#     except Exception as e:
#         print(f"Error loading {filename}: {e}")

# if not datasets:
#     print("No datasets loaded. Exiting.")
#     exit()

# # --- Parameter Initialization and Optimization ---
# # Initial guess for parameters [A, B, C, D, E, F, G]
# initial_guess = [4.881, -1.993, 0.374, 0.000, 0.000, 2.403, 0.000] # D,E,G are placeholders if model changes
# bounds = ([0, -np.inf, 0,      0,     0,     1e-6,  0],      # Lower bound for F > 0
#           [np.inf, np.inf, np.inf, np.inf, 100,   np.pi, np.inf]) # Upper bound for F as pi

# print("Checking model outputs at initial guess for the first dataset...")
# if datasets:
#     first_data_info = datasets[0]
#     distance = first_data_info[0][:, 0]
#     cosine_a = first_data_info[0][:, 2]
#     cosine_b = first_data_info[0][:, 3]
#     blinn = first_data_info[0][:, 4]
#     model_output_initial = model(initial_guess, distance, cosine_a, cosine_b, blinn)
#     if np.any(~np.isfinite(model_output_initial)):
#         print(f"Warning: Non-finite model outputs detected with initial guess for {first_data_info[1]}.")
#         # Further debugging for non-finite outputs could be added here if needed

# # Perform least squares optimization
# result = least_squares(residuals, initial_guess, bounds=bounds, loss='soft_l1', args=(datasets,), verbose=1)
# fitted_params = result.x
# print(f"Estimated parameters: A={fitted_params[0]:.3f}, B={fitted_params[1]:.3f}, C={fitted_params[2]:.3f}, "
#       f"D={fitted_params[3]:.3f}, E={fitted_params[4]:.3f}, F={fitted_params[5]:.3f}, G={fitted_params[6]:.3f}")

# # --- Calculate Fitted Intensities and Plot ---
# fitted_intensities_all_datasets = []
# for data_info in datasets:
#     data_array = data_info[0]
#     distance = data_array[:, 0]
#     cosine_a = data_array[:, 2]
#     cosine_b = data_array[:, 3]
#     blinn = data_array[:, 4]
    
#     fitted_intensity_normalized = model(fitted_params, distance, cosine_a, cosine_b, blinn)
#     fitted_intensities_all_datasets.append(fitted_intensity_normalized * 255.0)

# # --- Plotting Function ---
# def plot_dataset_fits(ax, x_data, y_data_actual, y_data_fitted, xlabel_text, show_legend):
#     sns.scatterplot(x=x_data, y=y_data_actual, ax=ax, color="blue", label="Data", alpha=0.6, s=30)
#     sns.scatterplot(x=x_data, y=y_data_fitted, ax=ax, color="orange", label="Fitted", alpha=0.8, s=20, marker='o')
#     ax.set_xlabel(xlabel_text)
#     ax.set_ylabel("Intensity")
#     if show_legend:
#         ax.legend(loc='best')
#     # Subplot title is now simpler as main info is in suptitle
#     ax.set_title(f"vs. {xlabel_text}")

# # --- Generate Plots for Each Dataset ---
# os.makedirs('imgs', exist_ok=True) # Create imgs directory if it doesn't exist

# for i, data_info in enumerate(datasets):
#     data_array, filename = data_info
    
#     distance = data_array[:, 0]
#     intensity_actual = data_array[:, 1] # Raw intensity for plotting
#     cosine_a = data_array[:, 2]
#     cosine_b = data_array[:, 3]
#     # blinn = data_array[:, 4] # Not directly plotted against, but used in model

#     fitted_intensity_scaled = fitted_intensities_all_datasets[i]
    
#     # Calculate R² score (using actual and fitted scaled intensities)
#     r2 = r2_score(intensity_actual, fitted_intensity_scaled)
#     rmse = root_mean_squared_error(intensity_actual, fitted_intensity_scaled)
#     print(f"R² Score for dataset {filename}: {r2:.3f}")
#     print(f"RMSE Score for dataset {filename}: {rmse:.3f}")

#     # --- Main Fit Plots ---
#     fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True) # sharey is good for comparison
#     fig.suptitle(f"Model Fit for: {filename} (R² = {r2:.3f}, RMSE = {rmse: .3f})", fontsize=16)

#     plot_dataset_fits(axs[0], distance, intensity_actual, fitted_intensity_scaled, "Distance", True) # Show legend on first
#     plot_dataset_fits(axs[1], cosine_a, intensity_actual, fitted_intensity_scaled, "Cosine Alpha", False)
#     plot_dataset_fits(axs[2], cosine_b, intensity_actual, fitted_intensity_scaled, "Cosine Beta", False)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
#     plt.savefig(f'imgs/{os.path.splitext(filename)[0]}_fit.png')
#     plt.close(fig)

#     # --- Residual Plots ---
#     residuals_val = intensity_actual - fitted_intensity_scaled

#     fig_res, axs_res = plt.subplots(1, 4, figsize=(24, 5.5)) # Slightly taller for titles
#     fig_res.suptitle(f"Residual Analysis for: {filename} (R² = {r2:.3f})", fontsize=16)
    
#     # 1. Residuals vs. Fitted
#     axs_res[0].scatter(fitted_intensity_scaled, residuals_val, alpha=0.5, color='green', s=20)
#     axs_res[0].axhline(0, color='red', linestyle='--')
#     axs_res[0].set_xlabel("Fitted Intensity")
#     axs_res[0].set_ylabel("Residuals (Actual - Fitted)")
#     axs_res[0].set_title("Residuals vs. Fitted")
#     axs_res[0].grid(True)

#     # 2. Residuals vs. Distance
#     axs_res[1].scatter(distance, residuals_val, alpha=0.5, color='green', s=20)
#     axs_res[1].axhline(0, color='red', linestyle='--')
#     axs_res[1].set_xlabel("Distance")
#     axs_res[1].set_ylabel("Residuals") # Keep ylabel if not sharey, or first plot if sharey
#     axs_res[1].set_title("Residuals vs. Distance")
#     axs_res[1].grid(True)

#     # 3. Residuals vs. Cosine Alpha
#     axs_res[2].scatter(cosine_a, residuals_val, alpha=0.5, color='green', s=20)
#     axs_res[2].axhline(0, color='red', linestyle='--')
#     axs_res[2].set_xlabel("Cosine Alpha")
#     axs_res[2].set_ylabel("Residuals")
#     axs_res[2].set_title("Residuals vs. Cosine Alpha")
#     axs_res[2].grid(True)

#     # 4. Histogram of Residuals
#     axs_res[3].hist(residuals_val, bins=30, alpha=0.7, color='green', edgecolor='black')
#     axs_res[3].axvline(0, color='red', linestyle='--')
#     axs_res[3].set_xlabel("Residual Value")
#     axs_res[3].set_ylabel("Frequency")
#     axs_res[3].set_title("Histogram of Residuals")
#     axs_res[3].grid(True)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f'imgs/{os.path.splitext(filename)[0]}_residuals.png')
#     plt.close(fig_res)

# print("Processing complete. Plots saved in 'imgs' directory.")

# Let's load and convert the original code to use `curve_fit` instead of `least_squares`.
# Since the original code has a custom residuals function for multiple datasets,
# we'll adapt it for curve_fit which expects xdata and ydata in specific formats.

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_squared_error
from math import log

# Seaborn settings
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# Paths
path = '/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt_new_3'
files_in_dir = os.listdir(path)
txt_files = [f for f in files_in_dir if f.endswith('.txt')]

# Model for curve_fit
def model_func(X, A, B, C, D, E, F, G):
    d, alpha, beta, blinn = X
    epsilon = 1e-9

    beta_clipped = np.clip(beta, -1.0, 1.0)

    if abs(F) < epsilon:
        angular_term = 1.0
    else:
        angular_term = np.exp(-np.log(2) * (np.arccos(beta_clipped) / F)**2)
        # angular_term = 1.0s

    denominator = d**2 + np.exp(-B)

    return (A * angular_term * alpha) / denominator + C

# Load all datasets and concatenate for curve_fit
X_all = []
y_all = []
datasets = []

for filename in txt_files:
    file_path = os.path.join(path, filename)
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] < 5:
            print(f"Warning: File {filename} has fewer than 5 columns. Skipping.")
            continue
        datasets.append((data, filename))
        d = data[:, 0]
        intensity_raw = data[:, 1] / 255.0
        alpha = data[:, 2]
        beta = data[:, 3]
        blinn = data[:, 4]
        for i in range(len(d)):
            X_all.append((d[i], alpha[i], beta[i], blinn[i]))
        y_all.extend(intensity_raw)
    except Exception as e:
        print(f"Error loading {filename}: {e}")

X_all = np.array(X_all).T  # shape: (4, N)
y_all = np.array(y_all)

# Fit with curve_fit
initial_guess = [0.881, -1.993, 0.574, 0.5, 5, 2.403, 0.5]
bounds = (
    [0, -np.inf, 0, 0, 1, 1e-6, 0],      # lower
    [np.inf, np.inf, np.inf, 1, 100, np.pi, 1]  # upper
)

popt, pcov = curve_fit(model_func, X_all, y_all, p0=initial_guess, bounds=bounds, maxfev=10000)
A, B, C, D, E, F, G = popt
print(f"Fitted params: A={A:.3f}, B={B:.3f}, C={C:.3f}, D={D:.3f}, E={E:.3f}, F={F:.3f}, G={G:.3f}")

# Make output directory
os.makedirs('imgs', exist_ok=True)

# Plot results for each dataset
for data_array, filename in datasets:
    d = data_array[:, 0]
    intensity_raw = data_array[:, 1]
    alpha = data_array[:, 2]
    beta = data_array[:, 3]
    blinn = data_array[:, 4]
    
    X_dataset = (d, alpha, beta, blinn)
    y_fitted = model_func(X_dataset, *popt) * 255.0
    
    r2 = r2_score(intensity_raw, y_fitted)
    rmse = np.sqrt(mean_squared_error(intensity_raw, y_fitted))

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f"{filename} (R² = {r2:.3f}, RMSE = {rmse:.3f})", fontsize=16)

    def plot_fit(ax, x_data, label):
        sns.scatterplot(x=x_data, y=intensity_raw, ax=ax, color='blue', label='Data', alpha=0.6, s=30)
        sns.scatterplot(x=x_data, y=y_fitted, ax=ax, color='orange', label='Fitted', alpha=0.8, s=20)
        ax.set_xlabel(label)
        ax.set_ylabel("Intensity")
        ax.set_title(f"vs. {label}")
        ax.legend()

    plot_fit(axs[0], d, "Distance")
    plot_fit(axs[1], alpha, "Cosine Alpha")
    plot_fit(axs[2], beta, "Cosine Beta")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"im2/{os.path.splitext(filename)[0]}_fit_cf.png")
    plt.close(fig)

    # --- Residual Plots ---
    residuals_val = intensity_raw - y_fitted

    fig_res, axs_res = plt.subplots(1, 6, figsize=(24, 5.5)) # Slightly taller for titles
    fig_res.suptitle(f"Residual Analysis for: {filename} (R² = {r2:.3f}, RMSE = {rmse:.3f})", fontsize=16)
    
    # 1. Residuals vs. Fitted
    axs_res[0].scatter(y_fitted, residuals_val, alpha=0.5, color='green', s=20)
    axs_res[0].axhline(0, color='red', linestyle='--')
    axs_res[0].set_xlabel("Fitted Intensity")
    axs_res[0].set_ylabel("Residuals (Actual - Fitted)")
    axs_res[0].set_title("Residuals vs. Fitted")
    axs_res[0].grid(True)

    # 2. Residuals vs. Distance
    axs_res[1].scatter(d, residuals_val, alpha=0.5, color='green', s=20)
    axs_res[1].axhline(0, color='red', linestyle='--')
    axs_res[1].set_xlabel("Distance")
    axs_res[1].set_ylabel("Residuals") # Keep ylabel if not sharey, or first plot if sharey
    axs_res[1].set_title("Residuals vs. Distance")
    axs_res[1].grid(True)

    # 3. Residuals vs. Cosine Alpha
    axs_res[2].scatter(alpha, residuals_val, alpha=0.5, color='green', s=20)
    axs_res[2].axhline(0, color='red', linestyle='--')
    axs_res[2].set_xlabel("Cosine Alpha")
    axs_res[2].set_ylabel("Residuals")
    axs_res[2].set_title("Residuals vs. Cosine Alpha")
    axs_res[2].grid(True)

    # 4. Residuals vs. Cosine Alpha
    axs_res[3].scatter(beta, residuals_val, alpha=0.5, color='green', s=20)
    axs_res[3].axhline(0, color='red', linestyle='--')
    axs_res[3].set_xlabel("Cosine Beta")
    axs_res[3].set_ylabel("Residuals")
    axs_res[3].set_title("Residuals vs. Cosine Beta")
    axs_res[3].grid(True)

    # 4. Residuals vs. Cosine Alpha
    axs_res[4].scatter(beta, residuals_val, alpha=0.5, color='green', s=20)
    axs_res[4].axhline(0, color='red', linestyle='--')
    axs_res[4].set_xlabel("Blinn")
    axs_res[4].set_ylabel("Residuals")
    axs_res[4].set_title("Residuals vs. Blinn")
    axs_res[4].grid(True)

    # 5. Histogram of Residuals
    axs_res[5].hist(residuals_val, bins=50, alpha=0.7, color='green', edgecolor='black')
    axs_res[5].axvline(0, color='red', linestyle='--')
    axs_res[5].set_xlabel("Residual Value")
    axs_res[5].set_ylabel("Frequency")
    axs_res[5].set_title("Histogram of Residuals")
    axs_res[5].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'im2/{os.path.splitext(filename)[0]}_residuals.png')
    plt.close(fig_res)

print("Curve fitting complete and plots saved.")
