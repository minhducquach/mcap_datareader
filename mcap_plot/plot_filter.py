import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.animation import FuncAnimation

count = 0  # Initialize count

# Load dataset
data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt")
distance = data[:, 0]
intensity = data[:, 1]
cosine = data[:, 2]

# Define the model function
def model(params, d, cos):
    A, B, C = params
    return A * cos / (B + d**2) + C

# Define residual function for least_squares
def residuals(params, d, cos, I):
    return model(params, d, cos) - I

# Initial guess for A, B, C
initial_guess = [1, 1, 1]

# Perform robust fitting using least_squares with l1 loss
result = least_squares(residuals, initial_guess, loss='linear', args=(distance, cosine, intensity))

# Extract fitted parameters
A_fit, B_fit, C_fit = result.x
print(f"Estimated parameters (robust fitting): A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# Generate fitted values
fitted_intensity = model(result.x, distance, cosine)

# Create 3D scatter plot
fig = plt.figure()
ax = plt.axes(projection='3d')
d = []
c = []
i = []

# Animation function
def animate(v):
    global count  # Declare count as global
    d.append(distance[count])
    c.append(cosine[count])
    i.append(intensity[count])
    count += 1
    ax.cla()  # Clear the previous plot
    # ax.scatter3D(d, c, i, label="Data", color="blue", alpha=0.6)
    ax.scatter3D(distance, cosine, intensity, label="Data", color="blue", alpha=0.6)
    ax.scatter3D(distance, cosine, fitted_intensity, label="Fitted", color="green", alpha=0.6)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Cosine")
    ax.set_zlabel("Intensity")
    ax.legend()

# Create animation
ani = FuncAnimation(fig, animate, frames=1000, interval=10, repeat=True)

plt.show()
