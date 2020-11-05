"""
    This is a module to prepare dataset for point cloud only 3d detection
    It includes:
    1) Ground Truth Database Generator
    2) Data Loader
"""



import os
import numpy as np 
from det3d.kitti_dataset.utils import kitti_utils
from det3d.kitti_dataset.kitti_dataset import KittiDataset

class PCKittiDataset(KittiDataset):
    """
        This is a

    Args:
        KittiDataset ([type]): [description]
    """

    def __init__(self, root_dir:str, split:str = 'train'):
        super().__init__(root_dir, split)


    