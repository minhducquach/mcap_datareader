import numpy as np

class Pose:
    def __init__(self, point, quaternion):
        self.position = point
        self.orientation = quaternion

class Data:
    def __init__(self):
        self.datapoints = []