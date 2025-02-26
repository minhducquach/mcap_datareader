import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.animation import FuncAnimation
from math import cos, radians

# Load dataset
data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt")
distance = data[:, 0]
intensity = data[:, 1]
rot_x = data[:, 2]
rot_y = data[:, 3]
rot_z = data[:, 4]

# Compute cosine factor (assuming Rot_y is the most important)
cosine = np.array([cos(radians(rot_y[i])) for i in range(len(data))])

# Define the model function
def model(params, d, c):
    A, B, C = params
    return A * c / (B + d**2) + C

# Define residual function for least_squares
def residuals(params, d, c, I):
    return model(params, d, c) - I

# Initial guess for A, B, C
initial_guess = [1, 1, 1]

# Perform fitting using least_squares with 'soft_l1' loss for robustness
result = least_squares(residuals, initial_guess, loss='soft_l1', args=(distance, cosine, intensity))

# Extract fitted parameters
A_fit, B_fit, C_fit = result.x
print(f"Estimated parameters (robust fitting): A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# Generate fitted values
fitted_intensity = model(result.x, distance, cosine)

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for data points and fitted model
sc_data = ax.scatter3D([], [], [], label="Data", color="blue", alpha=0.6)
sc_fit = ax.scatter3D([], [], [], label="Fitted", color="green", alpha=0.6)

ax.set_xlabel("Distance")
ax.set_ylabel("Cosine")
ax.set_zlabel("Intensity")
ax.legend()

count = 0  # Initialize count

# Animation function
def animate(frame):
    global count
    if count >= len(distance):
        count = 0  # Reset animation
    
    ax.cla()  # Clear previous plot
    ax.scatter3D(distance[:count], cosine[:count], intensity[:count], label="Data", color="blue", alpha=0.6)
    ax.scatter3D(distance[:count], cosine[:count], fitted_intensity[:count], label="Fitted", color="green", alpha=0.6)

    ax.set_xlabel("Distance")
    ax.set_ylabel("Cosine")
    ax.set_zlabel("Intensity")
    ax.legend()
    
    count += 10  # Increase step size for smooth animation

# Create animation
ani = FuncAnimation(fig, animate, frames=len(distance)//10, interval=100, repeat=True)

plt.show()
