from __future__ import division

import numpy as np
from numpy.linalg import multi_dot
import cv2

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


# def scale(scale_range, mode='g', scale_xyz=None, T=None):
#     if T is None:
#         T = np.eye(3)
#     if scale_xyz is None:
#         if mode == 'g':
#             scale_factor_xy = gauss_dist(1., scale_range)
#             scale_factor_z = gauss_dist(1., scale_range)
#         elif mode == 'u':
#             scale_factor_xy = uni_dist(1., scale_range)
#             scale_factor_z = uni_dist(1., scale_range)
#         else:
#             raise ValueError("Undefined scale mode: {}".format(mode))
#     else:
#         scale_factor_x, scale_factor_y, scale_factor_z = scale_xyz
#     T = np.dot(T, np.array([[scale_factor_xy, 0, 0],
#                             [0, scale_factor_xy, 0],
#                             [0, 0, scale_factor_z]]))
#     return T, [scale_factor_xy, scale_factor_z]


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



def rotate(rotate_range, mode, angle=None, T=None):
    if T is None:
        T = np.eye(3)
    if angle is None:
        if mode == 'g':
            angle = gauss_dist(0., rotate_range)
        elif mode == 'u':
            angle = np.random.uniform(0., rotate_range)
        else:
            raise ValueError("Undefined rotate mode: {}".format(mode))

    # R = np.array([[np.cos(angle), -np.sin(angle), 0],
    #               [np.sin(angle), np.cos(angle), 0],
    #               [0, 0, 1]]) # z axis rotation

    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]]) # y axis rotation kitti camera frame
    T = multi_dot([R, T])
    return T, angle


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


def drop(data):
    raw_length = len(data)
    length = int(np.clip(-raw_length//8 * np.abs(np.random.randn()) + raw_length, raw_length-raw_length//4, raw_length))
    if length < raw_length:
        left_data = data[:length, ...]
        padding_idx = np.random.randint(length, size=[raw_length-length])
        data = np.concatenate([left_data, left_data[padding_idx, ...]], axis=0)
        return data
    else:
        return data


def ones_padding(raw_input):
    features = np.ones(shape=(raw_input.shape[0], raw_input.shape[1], 1), dtype=np.float32)
    return features


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


def img_feature_value_perturbation(map, noise_scale=0, offset_scale=0, depth_scale=0, mode='g'):
    assert len(map.shape) == 3
    channels = map.shape[-1]
    height = map.shape[0]
    width = map.shape[1]
    mask = map[:, :, 0] != -1
    depth_span = np.percentile(map[mask, 0], 95) - np.percentile(map[mask, 0], 5)
    for c in range(channels):
        std = np.std(map[mask, c])
        span = np.percentile(map[mask, c], 95) - np.percentile(map[mask, c], 5)
        if mode == 'g':
            noise = np.random.randn(height, width) * std * noise_scale
            offset = gauss_dist(mean=0, shift=offset_scale * span)
            depth_wise = gauss_dist(mean=0, shift=span * depth_scale) * map[mask, 0] / depth_span
        elif mode == 'u':
            noise = (np.random.rand(height, width) - 0.5) * 2 * std * noise_scale
            offset = uni_dist(mean=0, shift=offset_scale * span)
            depth_wise = uni_dist(mean=0, shift=span * depth_scale) * map[mask, 0] / depth_span
        else:
            raise NameError("Unsupported random mode: {}".format(mode))
        map[mask, c] =  map[mask, c] + noise[mask] + offset + depth_wise

    return map


def point_feature_value_perturbation(features, depth, noise_scale=0, offset_scale=0, depth_scale=0, mode='g'):
    assert len(features.shape) == 2
    assert len(depth.shape) == 1
    assert features.shape[0] == depth.shape[0]
    channels = features.shape[-1]
    npoint = features.shape[0]
    depth_span = np.percentile(depth, 95) - np.percentile(depth, 5)
    for c in range(channels):
        std = np.std(features[:, c])
        span = np.percentile(features[:, c], 95) - np.percentile(features[:, c], 5)
        if mode == 'g':
            noise = np.random.randn(npoint) * std * noise_scale
            offset = gauss_dist(mean=0, shift=offset_scale * span)
            depth_wise = gauss_dist(mean=0, shift=span * depth_scale) * depth / depth_span
        elif mode == 'u':
            noise = (np.random.rand(npoint) - 0.5) * 2 * std * noise_scale
            offset = uni_dist(mean=0, shift=offset_scale * span)
            depth_wise = uni_dist(mean=0, shift=span * depth_scale) * depth / depth_span
        else:
            raise NameError("Unsupported random mode: {}".format(mode))
        features[:, c] = features[:, c] + noise + offset + depth_wise

    return features





