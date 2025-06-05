import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the boxes for the diagram
boxes = {
    "Input": ("RGB + Depth Image\nOdometry Pose", (0.1, 0.7)),
    "Init": ("Init Pose / Camera Pose\nGenerate Gaussians", (0.4, 0.7)),
    "Keyframe": ("Add Keyframe\nLaunch Optimization Thread", (0.7, 0.7)),
    "PoseEst": ("IMU Preintegration\n(Predict Camera Pose)", (0.1, 0.4)),
    "Optimize": ("Photometric Optimization\n(Pose Estimation)", (0.4, 0.4)),
    "Covis": ("Check Covisibility\nNew Keyframe?", (0.7, 0.4)),
    "Update": ("Update Gaussians\nDensify & Marginalize", (0.4, 0.1)),
}

# Define arrows between steps
arrows = [
    ("Input", "Init"),
    ("Init", "Keyframe"),
    ("Keyframe", "PoseEst"),
    ("PoseEst", "Optimize"),
    ("Optimize", "Covis"),
    ("Covis", "Update"),
    ("Update", "PoseEst"),  # Loop back for next frame
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

# Draw boxes
for name, (text, pos) in boxes.items():
    ax.add_patch(patches.FancyBboxPatch(
        (pos[0], pos[1]), 0.25, 0.15,
        boxstyle="round,pad=0.02", edgecolor="black", facecolor="#f0f0f0"
    ))
    ax.text(pos[0]+0.125, pos[1]+0.075, text, ha="center", va="center", fontsize=10)

# Draw arrows
for start, end in arrows:
    x1, y1 = boxes[start][1]
    x2, y2 = boxes[end][1]
    ax.annotate("",
                xy=(x2+0.125, y2+0.15 if y2 > y1 else y2), 
                xytext=(x1+0.125, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5))

plt.tight_layout()
plt.show()
