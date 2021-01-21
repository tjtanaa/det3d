import os
import sys
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List

import det3d.kitti_dataset.utils.calibration as calibration
import det3d.kitti_dataset.utils.kitti_utils as kitti_utils
from PIL import Image


class KittiDatasetBase(object):
    """ This is a base class of kitti dataset.
        You have to extend this class to create your own KITTI Dataset Loader.
        Currently it only supports mono image.
       Its functionality is to primitive data from KITTI detection dataset:
       Files related:
       1) image
       2) lidar
       3) calibration file
       4) annotations/ label
       5) plane

        Note: 
        The coordinate system of the point cloud and the labels are different
        Point cloud is in a Z-up coordinate system.
        Label is in a Y-down coordinate system.

        Point cloud   (Velodyne Coordinate System)    Yaw_pc = -Yaw_label                      

                        (z) height
                            ^
                            |
                width       |
        Front   (x) <-------|
                           / 
                          /
                         /
                        L
                        (y) length  Left


        Label  (CAM 0 Coordinate System)     Yaw_label = -Yaw_pc                      

                   
                length   (Right)    
                    (x) <-------|
                               /| 
                              / |
                             /  |
                            L   |
                width   (z)     |
                Front           v
                                (y) height (Bottom)


        IMPORTANT:
        The center of label (x,y,z) computed from transformation matrix is not the
        true center.
        In Point Cloud (Velodyne) coordinate, you have to offset the z by (+h/2)
        In Camera (CAM 0) coordinate, you have to offset the y by (-h/2)

    """
    def __init__(self, root_dir: str, split: str='train'):
        """ Constructor

        Args:
            root_dir (str): The absolute path to the KITTI dataset
            split (str, optional): Possible values are 'train', 'val' and 'test'. 
                                Defaults to 'train'.
        """

        self.split = split
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')
        self.image_idx_list = []
        if split == 'train_val':
            split_list = split.split('_')
            for s in split_list:
                split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', s + '.txt')
                self.image_idx_list.extend([x.strip() for x in open(split_dir).readlines()])
                # print(self.image_idx_list[0], "\t", print(self.image_idx_list[-1]))
            self.image_idx_list= self.image_idx_list[:-784]
            # print(len(self.image_idx_list))
            self.split = 'train'
            # exit()
        elif split == 'train_val_test':
            split_list = split.split('_')[:-1]
            for s in split_list:
                split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', s + '.txt')
                self.image_idx_list.extend([x.strip() for x in open(split_dir).readlines()])
            self.image_idx_list= self.image_idx_list[-784:]
            # print(len(self.image_idx_list))
            self.split = 'val'
        else:
            split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
            self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        print(len(self.image_idx_list))
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')

    def get_image(self, idx):
        # assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        # im = np.array(Image.open(img_file))
        # print("img shape: ", im.shape)
        # return im
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        # print("calib file: ", calib_file, " Found: ", os.path.exists(calib_file))
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane



    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
