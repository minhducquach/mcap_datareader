import matplotlib.pyplot as plt
import numpy as np

file_1 = open('/home/manip/ros2_ws/src/mcap_plot/mcap_plot/image_data.txt')
file_2 = open('/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt')

point_x = []
point_y = []
flag = 0
f1 = file_1.readline()

while f1 != "":
    t1 = f1.split()
    cmp = file_2.readline().split()
    while (True):
        if (abs(float(cmp[0]) - float(t1[0])) <= 0.1):
            point_x.append(float(cmp[1]))
            point_y.append(float(t1[1]))
            cmp = file_2.readline().split()
            f1 = file_1.readline()
            break
        else:
            cmp = file_2.readline().split()

point_x = np.array(point_x)
point_y = np.array(point_y)

plt.plot(point_x, point_y)
plt.show()