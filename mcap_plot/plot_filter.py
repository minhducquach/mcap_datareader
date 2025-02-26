# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import least_squares
# from matplotlib.animation import FuncAnimation
# from math import cos, radians

# count = 0  # Initialize count

# # Load dataset
# data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt")
# distance = data[:, 0]
# intensity = data[:, 1]
# rot_x = data[:, 2]
# rot_y = data[:, 3]
# rot_z = data[:, 4]
# # cosine = np.array([cos(radians(rot_x[i])) * cos(radians(rot_y[i])) * cos(radians(rot_z[i])) for i in range(len(data))])
# cosine = np.array([cos(radians(rot_y[i])) for i in range(len(data))])

# # Define the model function
# def model(params, d, c):
#     A, B, C = params
#     return A * c / (B + d**2) + C

# # Define residual function for least_squares
# def residuals(params, d, c, I):
#     return model(params, d, c) - I

# # Initial guess for A, B, C
# initial_guess = [1, 1, 1]

# # Perform robust fitting using least_squares with l1 loss
# result = least_squares(residuals, initial_guess, loss='linear', args=(distance, cosine, intensity))

# # Extract fitted parameters
# A_fit, B_fit, C_fit = result.x
# print(f"Estimated parameters (robust fitting): A={A_fit:.3f}, B={B_fit:.3f}, C={C_fit:.3f}")

# # Generate fitted values
# fitted_intensity = model(result.x, distance, cosine)

# # Create 3D scatter plot
# fig = plt.figure()

# ax = plt.axes(projection='3d')
# # ax1 = fig.add_subplot(2, 2, 1)
# # ax2 = fig.add_subplot(2, 2, 2)
# # ax3 = fig.add_subplot(2, 2, 3)
# # ax4 = fig.add_subplot(2, 2, 4)

# d = []
# c = []
# i = []
# rx = []
# ry = []
# rz = []

# # Animation function
# def animate(v):
#     global count  # Declare count as global
#     d.append(distance[count])
#     c.append(cosine[count])
#     i.append(intensity[count])
#     rx.append(rot_x[count])
#     ry.append(rot_y[count])
#     rz.append(rot_z[count])
#     count += 1

#     ax.cla()  # Clear the previous plot
#     ax.scatter3D(distance, cosine, intensity, label="Data", color="blue", alpha=0.6)
#     ax.scatter3D(distance, cosine, fitted_intensity, label="Fitted", color="green", alpha=0.6)
#     ax.set_xlabel("Distance")
#     ax.set_ylabel("Cosine")
#     ax.set_zlabel("Intensity")
#     ax.legend()

#     # ax1.cla()
#     # ax1.scatter(d, i, label="Dist", color="blue", alpha=0.6)
#     # ax1.set_xlabel("Distance")
#     # ax1.set_ylabel("Intensity")
#     # ax1.legend()

#     # ax2.cla()
#     # ax2.scatter(rx, i, label="Rot_x", color="green", alpha=0.6)
#     # ax2.set_xlabel("Rot_x")
#     # ax2.set_ylabel("Intensity")
#     # ax2.legend()

#     # ax3.cla()
#     # ax3.scatter(ry, i, label="Rot_y", color="red", alpha=0.6)
#     # ax3.set_xlabel("Rot_y")
#     # ax3.set_ylabel("Intensity")
#     # ax3.legend()

#     # ax4.cla()
#     # ax4.scatter(rz, i, label="Rot_z", color="orange", alpha=0.6)
#     # ax4.set_xlabel("Rot_z")
#     # ax4.set_ylabel("Intensity")
#     # ax4.legend()
# # Create animation
# ani = FuncAnimation(fig, animate, frames=100, interval=0.1, repeat=True)

# plt.show()

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
    count += 10
    ax.cla()  # Clear the previous plot
    ax.scatter3D(distance, cosine, intensity, label="Data", color="blue", alpha=0.6)
    ax.scatter3D(distance, cosine, fitted_intensity, label="Fitted", color="green", alpha=0.6)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Cosine")
    ax.set_zlabel("Intensity")
    ax.legend()

# Create animation
ani = FuncAnimation(fig, animate, frames=10, interval=10, repeat=True)

plt.show()

