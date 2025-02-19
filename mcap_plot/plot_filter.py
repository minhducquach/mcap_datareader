import matplotlib.pyplot as plt
import numpy as np

file = open('/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt')

point_x = []
point_y = []
flag = 0

for line in file:
    line_split = line.split()
    point_x.append(line_split[0])
    point_y.append(line_split[1])

point_x = np.array(point_x)
point_y = np.array(point_y)

plt.scatter(point_x, point_y)
plt.show()