from __future__ import division

import numpy as np
from numpy.linalg import multi_dot
import cv2
from typing import List, Set, Dict, Tuple, Optional
from typing import Callable, Iterator, Union, Optional, List, Any

from numba import cuda


'''
This augmentation is performed instance-wised, not for batch-processing.

'''


def shuffle(data):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, :]


def gauss_dist(mean, shift):
    if shift != 0:
        std = shift * 0.5
        return_value = np.clip(np.random.normal(mean, std), mean - shift, mean + shift)
        return return_value
    else:
        return mean

def uni_dist(mean, shift):
    if shift != 0:
        return_value = (np.random.rand() - 0.5) * 2 * shift + mean
        return return_value
    else:
        return mean

def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

def scale_matrix(scale: List[float]) -> Any:
    """
       This function generates the scale matrix for 2D and 3D

    Args:
        scale (List[float]): [sx, sy] for 2D or [sx, sy, sz] for 3D

    Returns:
        Any: the rotation matrix
    """

    T = None
    if len(scale) == 2:
        T = np.array([[scale[0], 0],
                [0, scale[1]]])

    elif len(scale) == 3:
        T = np.array([[scale[0], 0, 0],
                    [0, scale[1], 0],
                    [0, 0, scale[2]]])
    return T


def flip_matrix(flip: List[Union[float, int]]) -> Any:
    """
       This function generates the flipping matrix

       [Note]
       In 3D:
       
       Examples from https://www.gatevidyalay.com/tag/3d-reflection-matrix/:
       for flip = [1,1,-1], the point cloud will be mirror along z axis/
       reflect relative to XY plane

    Args:
        flip (List[Union[float, int]]): [fy, fx] for 2D or [fyz, fxz, fxy] for 3D.
                                        The values should either be 1 or -1

    Returns:
        Any: The flipping matrix
    """

    T = None

    if len(flip) == 2:
        T = np.array([[flip[0], 0],
                        [0, flip[2]]])

    elif len(flip) == 3:
        T = np.array([[flip[0], 0, 0],
                                [0, flip[1], 0],
                                [0, 0, flip[2]]])
    return T



def rotation_matrix(angle: List[float]) -> Any:
    """
       This function generates a rotation matrix

    Args:
        angle (List[float]): [theta] in 2D or [rx, ry, rz] in 3D
                            angles are in radian

    Returns:
        Any: The rotation matrix
    """
    T = None

    if len(angle) == 1: # 2D
        theta = angle[0]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))


    if len(angle) == 3: # 3D

        rx, ry, rz = angle

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]]) # x axis rotation


        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]]) # y axis rotation kitti camera frame

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]]) # z axis rotation
    T = multi_dot([Rz,Ry,Rx])
    return T 


def shear_matrix(shear_range: List[float]) -> Any:

    raise NotImplementedError

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


def perspective_transformation(map, scale, padding_list=None, mode='g'):
    assert len(map.shape) == 3
    height = map.shape[0]
    width = map.shape[1]
    channels = map.shape[-1]
    if padding_list is None:
        padding_list = np.zeros(channels)
    else:
        assert len(padding_list) == channels
    if mode == 'g':
        rand_method = gauss_dist
    elif mode == 'u':
        rand_method = uni_dist
    else:
        raise NameError("Unsupported random mode: {}".format(mode))
    original_corners = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    new_corners = np.float32([[rand_method(0, scale * width), rand_method(0, scale * height)],
                              [rand_method(width, scale * width), rand_method(0, scale * height)],
                              [rand_method(0, scale * width), rand_method(height, scale * height)],
                              [rand_method(width, scale * width), rand_method(height, scale * height)]])
    M = cv2.getPerspectiveTransform(new_corners, original_corners)
    for c in range(channels):
        map[:, :, c] = cv2.warpPerspective(map[:, :, c], M, (width, height),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=padding_list[c])

    return map


def horizontal_flip(map):
    return map[:, ::-1, :]

@cuda.jit
def allocate_bev(points, bev_buffer, voxel_size, coor_range):
    
    pass

def points_to_bev(points, voxel_size, coor_range):
    assert points.ndim == 3

    bev_x_len = np.ceil((coor_range[3] - coor_range[0])/voxel_size[0])
    bev_y_len = np.ceil((coor_range[4] - coor_range[1])/voxel_size[1])
    bev_z_len = np.ceil((coor_range[5] - coor_range[2])/voxel_size[2])


    bev_buffer = np.zeros(shape=(bev_x_len, bev_y_len, bev_z_len))

    bev_map = allocate_bev(points, bev_buffer, voxel_size, coor_range)
    return bev_map


