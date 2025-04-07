import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MLP  # Assuming the model architecture is in model.py
from sklearn.metrics import r2_score

# Load the trained model
model = MLP(input_size=3, hidden_size=64, output_size=1)
model.load_state_dict(torch.load('trained_model.pth'))

# Now, the model is loaded and ready to use for inference

# Assuming you have the same dataset structure and inputs as in your original code
data = np.loadtxt("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/txt/light_tab_dis_data_diffax.txt")
distance = data[:, 0]
intensity = data[:, 1]
cosine_a = data[:, 2]
cosine_b = data[:, 3]

# Convert the data into a PyTorch tensor for the model
inputs = torch.tensor(np.column_stack((distance, cosine_a, cosine_b)), dtype=torch.float32)

# Use the model for prediction
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation to save memory and computations
    predictions = model(inputs).numpy()  # Run the model on the inputs

# Now, you can plot the results using matplotlib
fitted_intensity = predictions.squeeze()  # Remove extra dimensions if needed

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
