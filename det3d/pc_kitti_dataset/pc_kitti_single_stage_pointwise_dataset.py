"""
    This is a module to prepare dataset for point cloud only 3d detection
    It includes:
    1) Data Loader without gt augmentation and with gt augmentation
"""



import os
import numpy as np 
import pickle
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List
from det3d.kitti_dataset.utils import kitti_utils
from det3d.pc_kitti_dataset import PCKittiAugmentedDataset
from det3d.pc_kitti_dataset.config import cfg
from det3d.point_cloud_utils.transformation import limit_period

class PCKittiSingleStagePointwiseDataset(PCKittiAugmentedDataset):
    """
        This is a database that generates information that are necessary 
        for pointwise object detection

    Args:
        PCKittiAugmentedDataset ([type]): [description]
    """

    def __init__(self, root_dir:str, npoints:int =16384, split: str ='train', 
                classes:List[str] =['Car'], random_select:bool =True,
                gt_database_dir=None, aug_hard_ratio:float=0.5, **kwargs):
        super().__init__(root_dir = root_dir, 
                npoints=npoints, 
                split=split, 
                classes=classes, 
                random_select=random_select,
                gt_database_dir=gt_database_dir, 
                aug_hard_ratio=aug_hard_ratio, **kwargs)
        self.classes = self.classes
        self.num_class = self.classes.__len__()

        self.npoints = npoints
        self.random_select = random_select
        self.aug_hard_ratio = aug_hard_ratio
        # self.split = super().split
        assert split in ['train', 'val', 'train_val', 'train_val_test', 'test'], 'Invalid mode: %s' % split
        # self.split = split
        # print("PCKittiSingleStagePointwiseDataset ", self.split)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        return self.get_rpn_sample(index)

    @staticmethod
    def cls_type_to_id(cls_type):
        type_to_id = cfg.CLASSES_ID_MAPPING
        if cls_type not in type_to_id.keys():
            return -1
        return type_to_id[cls_type]

    @staticmethod
    def id_to_cls_type(id_type:int):
        id_to_type = cfg.ID_CLASSES_MAPPING
        if id_type not in id_to_type.keys():
            return ""
        return id_to_type[str(id_type)]

    def get_rpn_sample(self, index):
        sample_info = self.get_sample(index)

        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((sample_info['pts_rect'], sample_info['pts_features']), axis=1)  # (N, C)
        else:
            pts_input = sample_info['pts_rect']

        sample_info['pts_input'] = pts_input
        if self.split == "test":
            return sample_info

        # generate training labels
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(sample_info['pts_rect'], 
                                                                    sample_info['gt_boxes3d'], 
                                                                    sample_info['gt_cls_type_list'])
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        return sample_info

    
    def generate_rpn_training_labels(self, pts_rect, gt_boxes3d, gt_cls_type_list):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32) # integer and the class_type to id mapping is from config.py
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l (without anchor)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        # extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        # extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = self.cls_type_to_id(gt_cls_type_list[k])

            # enlarge the bbox3d, ignore nearby points (WHY?)
            # extend_box_corners = extend_gt_corners[k]
            # fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            # ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            # cls_label[ignore_flag] = -1

            # pixel offset of object center 
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2 # (y - h/2) This is true
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928 by KITTI

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

        return cls_label, reg_label

    