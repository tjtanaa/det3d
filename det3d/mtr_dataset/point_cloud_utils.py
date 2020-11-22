from __future__ import division
import os
import numpy as np
from numpy.linalg import multi_dot
import cv2
import open3d as o3d
import os
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List



'''
This augmentation is performed instance-wised, not for batch-processing.

'''

def scale(scale_range, mode='g', scale_xyz=None, T=None):
    if T is None:
        T = np.eye(3)
    if scale_xyz is None:
        if mode == 'g':
            scale_factor_xz = gauss_dist(1., scale_range)
            scale_factor_y = gauss_dist(1., scale_range)
        elif mode == 'u':
            scale_factor_xz = uni_dist(1., scale_range)
            scale_factor_y = uni_dist(1., scale_range)
        else:
            raise ValueError("Undefined scale mode: {}".format(mode))
    else:
        scale_factor_x, scale_factor_y, scale_factor_z = scale_xyz
    T = np.dot(T, np.array([[scale_factor_xz, 0, 0],
                            [0, scale_factor_y, 0],
                            [0, 0, scale_factor_xz]]))
    return T, [scale_factor_xz, scale_factor_y]


def flip(flip=False, T=None):
    if T is None:
        T = np.eye(3)
    if not flip:
        return np.dot(T, np.eye(3)), 1.
    else:
        flip_y = -1 if np.random.rand() > 0.5 else 1
        # T = np.dot(T, np.array([[1, 0, 0],
        #                         [0, flip_y, 0],
        #                         [0, 0, 1]])) # flip along y axis
        T = np.dot(T, np.array([[flip_y, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])) # flip along z axis (kitti camera frame)
        return T, flip_y


def rotation_matrix(angle: List[float]) -> Any:
    """ This is a function that generates the rotation matrix.

    Args:
        angle (List[float]): List of [rx, ry, rz] in radians

    Returns:
        Tuple[Any, Any]: The rotation matrix
    """
    
    T = np.eye(3)
    
    rx = angle[0]
    Rx = np.array([[1, 0 , 0],
                  [0, np.cos(rx),  -np.sin(rx)],
                  [0, np.sin(rx), np.cos(rx)]]) # x axis rotation
    ry = angle[1]
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]]) # y axis rotation kitti camera frame
    rz = angle[2]
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                  [np.sin(rz), np.cos(rz), 0],
                  [0, 0, 1]]) # z axis rotation
    
    T = multi_dot([Rz, Ry, Rx, T])
    return T


def shear(shear_range, mode, shear_xy=None, T=None):
    if T is None:
        T = np.eye(3)
    # TODO: Need to change the angles_z into uniform_dist for ModelNet_40
    if shear_xy is None:
        if mode == 'g':
            lambda_x = gauss_dist(0., shear_range)
            lambda_y = gauss_dist(0., shear_range)
        elif mode == 'u':
            lambda_x = np.random.uniform(0., shear_range)
            lambda_y = np.random.uniform(0., shear_range)
        else:
            raise ValueError("Undefined shear mode: {}".format(mode))
    else:
        lambda_x, lambda_y = shear_xy
    Sx = np.array([[1, 0, lambda_x],
                   [0, 1, 0],
                   [0, 0, 1]])
    Sy = np.array([[1, 0, 0],
                   [0, 1, lambda_y],
                   [0, 0, 1]])
    T = multi_dot([Sx, Sy, T])
    return T, [lambda_x, lambda_y]


def transform(data, T):
    transformed = np.transpose(np.dot(T, np.transpose(data)))
    return transformed


def denoising_point_cloud_index(point_cloud:Any, nb_neighbors=20, std_ratio=1.0):
    """
        This is a function that returns the indices of the inliers

    Args:
        point_cloud (Any): [description]
        nb_neighbors (int, optional): [description]. Defaults to 20.
        std_ratio (float, optional): [description]. Defaults to 1.0.

    Returns:
        [list]: list of indices of inliers
    """
    xyz = point_cloud[:,:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                            std_ratio=std_ratio)
    return np.array(ind)