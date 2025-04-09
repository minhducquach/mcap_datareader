import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score
from math import exp, log, e, pi
import os

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.3)


# Path to the folder containing the text files
path = '/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt'
files = os.listdir(path)  # List all files in the directory

# Define the objective function
def model(params, distance, cosa, cosb):
    A, B, C = params
    return ((A * 1 / np.log(np.e + 1) * np.log(1 + np.exp(cosb)) / (distance**2 + exp(-B))) + C) * cosa * 0.6 / 3.14

def objective(params, datasets):
    residuals_total = []
    for data in datasets:
        distance = data[:, 0]
        intensity = data[:, 1]
        cosine_a = data[:, 2]
        cosine_b = data[:, 3]
        model_intensity = model(params, distance, cosine_a, cosine_b)
        residuals_total.append(np.sum((model_intensity - intensity)**2))  # Sum of squares of residuals
    return np.sum(residuals_total)

# Load datasets
datasets = []
for file in files:
    file_path = os.path.join(path, file)
    if file_path.endswith('.txt'):
        data = np.loadtxt(file_path)
        datasets.append(data)

# Define bounds for parameters A, B, C
bounds = [(0, 100000), (-10, 10), (0, 1)]

# Perform Differential Evolution optimization
result = differential_evolution(objective, bounds, args=(datasets,), maxiter=1000, popsize=15)

# Get the optimized parameters
A_fit, B_fit, C_fit = result.x
print(f"Estimated parameters: A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# Calculate fitted intensities for each dataset using optimized parameters
fitted_intensities = []
for data in datasets:
    distance = data[:, 0]
    cosine_a = data[:, 2]
    cosine_b = data[:, 3]
    fitted_intensity = model(result.x, distance, cosine_a, cosine_b)
    fitted_intensities.append(fitted_intensity)

# Optionally, calculate R² scores for each dataset
for i, data in enumerate(datasets):
    intensity = data[:, 1]
    fitted_intensity = fitted_intensities[i]
    r2 = r2_score(intensity, fitted_intensity)
    print(f"R² Score for dataset {i+1}: {r2:.3f}")
