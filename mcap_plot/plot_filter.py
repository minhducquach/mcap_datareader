import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.animation import FuncAnimation

count = 0  # Initialize count

# Load dataset
data = np.loadtxt("/media/minhducquach/MiduT71/STUDY/SLAM/Internship/mcap_datareader/mcap_plot/light_tab_dis_data.txt")
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

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

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
    ax1.cla()
    ax1.scatter(distance, intensity, label="Data", color="blue", alpha=0.6)
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Intensity")
    ax1.legend()

    ax2.cla()
    ax2.scatter(cosine, intensity, label="Fitted", color="green", alpha=0.6)
    ax2.set_xlabel("Cosine")
    ax2.set_ylabel("Intensity")
    ax2.legend()
# Create animation
ani = FuncAnimation(fig, animate, frames=1000, interval=10, repeat=True)

plt.show()
