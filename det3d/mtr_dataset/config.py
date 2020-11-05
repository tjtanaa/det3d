import numpy as np 
import os

config = {}

config['class_map'] = {1: "pedestrian", 2: "others"}
config['train_test_split'] = 0.9

config['point_cloud'] = {}
config['point_cloud']['height'] = 128
config['point_cloud']['width'] = 2048
config['point_cloud']['channels'] = 7
# point cloud attributes of channels
config['point_cloud']['attributes'] = "x, y, z, intensity, range, ambient, reflectivity"

# statistics to reposition the point cloud upright
config['point_cloud']['xyz_offset'] = [0, 0, 3.3215672969818115]
config['point_cloud']['rxyz_offset'] = [np.pi, 0, 0]

