
import os
import numpy as np 
import pickle
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List
from det3d.kitti_dataset.utils import kitti_utils
from det3d.pc_kitti_dataset.template_config import cfg

from det3d.point_cloud_utils.transformation import flip_matrix, rotation_matrix, transform, limit_period

class PCKittiTemplateDataLoaderTemplate():

    def __init__(self, root_dir, split='train', classes:List[str]=['Car'], **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.template_database_path = os.path.join(self.root_dir, "gt_database")
        self.template_database = None
        self.classes = ['Background']
        classes.sort()
        self.classes.extend(classes)

        self.template_pkl_file_path = os.path.join(self.template_database_path, 
                '%s_template_gt_database_level_%s.pkl' % (self.split, '-'.join(self.classes)))

        with open(self.template_pkl_file_path, 'rb') as f:
            self.template_database =  pickle.load(f)

    def __len__(self):
        return len(self.template_database)

    def __getitem__(self, item):
        raise NotImplementedError

class PCKittiTemplateSingleDataLoader(PCKittiTemplateDataLoaderTemplate):

    def __init__(self, root_dir, split='train', classes:List[str]=['Car'], **kwargs):
        super().__init__(root_dir=root_dir, split=split, classes=classes)


    def generate_bbox_augmentation_params(self):
        param_dict = {
            "dxyz": np.array([0,0,0], dtype=np.float), # translate
            "dhwl": np.array([0,0,0], dtype=np.float), # scale
            "rxyz": np.array([0,0,0], dtype=np.float), # rotate
            "fxyz": np.array([1,1,1], dtype=np.float), # flip
        }

        return param_dict

    def bbox_apply_augmentation_params(self, box3d, param_dict=None):
        """augment the parameter of the bounding bboxes

        Args:
            box3d (np.array): [x,y,z,h,w,l,ry] 
            param_dict ([dict], optional): The dictionary will contain all the 
                                            augmentation parameters. 
                                            Defaults to generate_bbox_augmentation_params().
        """
        if param_dict is None:
            param_dict = self.generate_bbox_augmentation_params()

        aug_box3d = np.copy(box3d)
        # print("before augmentation: ", box3d)
        aug_box3d[:3] += param_dict['dxyz']
        aug_box3d[3:6] += param_dict['dhwl']
        aug_box3d[6] += param_dict['rxyz'][1]
        aug_box3d[:3] *= param_dict['fxyz']
        # print("after augmentation: ", box3d)

        return aug_box3d

    def bbox_augmentation(self, aug_gt_boxes3d, sample_id=None, mustaug=False):

        aug_list = cfg.AUG_BBOX_METHOD_LIST
        aug_enable = 1 - np.random.rand(4)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []

        aug_box_params = self.generate_bbox_augmentation_params()
        if 'translation' in aug_list and aug_enable[0] < cfg.AUG_BBOX_METHOD_PROB[0]:
            # print(np.random.uniform(-cfg.AUG_BBOX_X_RANGE, cfg.AUG_BBOX_X_RANGE, 1))
            aug_box_params['dxyz'][0] = np.random.uniform(-cfg.AUG_BBOX_X_RANGE, cfg.AUG_BBOX_X_RANGE, 1)
            aug_box_params['dxyz'][1] = np.random.uniform(-cfg.AUG_BBOX_Y_RANGE, cfg.AUG_BBOX_Y_RANGE, 1)
            aug_box_params['dxyz'][2] = np.random.uniform(-cfg.AUG_BBOX_Z_RANGE, cfg.AUG_BBOX_Z_RANGE, 1)
            
            # print("AUGMENT_TRANSLATION: ", aug_box_params['dxyz'])

        
        if 'rotation' in aug_list and aug_enable[1] < cfg.AUG_BBOX_METHOD_PROB[1]:
            ry_angle = np.random.uniform(-cfg.AUG_BBOX_YROT_RANGE, cfg.AUG_BBOX_YROT_RANGE)
            aug_box_params['rxyz'][1] = ry_angle
            # print("AUGMENT_ROTATE")
            

        if 'scaling' in aug_list and aug_enable[2] < cfg.AUG_BBOX_METHOD_PROB[2]:
            aug_box_params['dhwl'][0] = np.random.uniform(-cfg.AUG_BBOX_H_RANGE, cfg.AUG_BBOX_H_RANGE)
            aug_box_params['dhwl'][1] = np.random.uniform(-cfg.AUG_BBOX_W_RANGE, cfg.AUG_BBOX_W_RANGE)
            aug_box_params['dhwl'][2] = np.random.uniform(-cfg.AUG_BBOX_L_RANGE, cfg.AUG_BBOX_L_RANGE)
            # print("AUGMENT_SCALING")


        if 'flip' in aug_list and aug_enable[3] < cfg.AUG_BBOX_METHOD_PROB[3]:
            if cfg.AUG_BBOX_XFLIP_ENABLE  == 1:
                aug_box_params['fxyz'][0] = np.random.choice([-1,1]) 
            if cfg.AUG_BBOX_YFLIP_ENABLE  == 1:
                aug_box_params['fxyz'][1] = np.random.choice([-1,1]) 
            if cfg.AUG_BBOX_ZFLIP_ENABLE  == 1:
                aug_box_params['fxyz'][2] = np.random.choice([-1,1]) 
            # print("AUGMENT_FLIPPTING")
        
        aug_gt_boxes3d = self.bbox_apply_augmentation_params(aug_gt_boxes3d, aug_box_params)

        return aug_gt_boxes3d, aug_box_params

    

    def generator(self):
        
        while True:
            for idx, kitti_obj in enumerate(self.template_database):
                bbox3d = kitti_obj['gt_box3d']

                aug_bbox3d, aug_box_params = self.bbox_augmentation(bbox3d)
                kitti_obj['aug_gt_box3d'] = aug_bbox3d
                kitti_obj['aug_box_params'] = aug_box_params
                # print("aug_bbox3d: ", aug_bbox3d)

                yield kitti_obj


    def bounding_box_pc_pairs_generator(self):

        with open(self.template_pkl_file_path, 'rb') as f:
            self.template_database =  pickle.load(f)

        
        while True:
            for idx, kitti_obj in enumerate(self.template_database):
                bbox3d = kitti_obj['gt_box3d']

                aug_bbox3d, aug_box_params = self.bbox_augmentation(bbox3d)
                kitti_obj['aug_gt_box3d'] = aug_bbox3d
                kitti_obj['aug_box_params'] = aug_box_params
                # print("aug_bbox3d: ", aug_bbox3d)
                
                pts_rect = kitti_obj['points']

                bboxes3d_rotated_corners = kitti_utils.boxes3d_to_corners3d(bbox3d[np.newaxis,:])
                aug_bboxes3d_rotated_corners = kitti_utils.boxes3d_to_corners3d(aug_bbox3d[np.newaxis,:])

                boxes_pts_mask_list = []
                aug_boxes_pts_mask_list = []
                for i, bbox3d_corners in enumerate(bboxes3d_rotated_corners):
                    box3d_roi_inds = kitti_utils.in_hull(pts_rect[:,:3], bbox3d_corners)
                    boxes_pts_mask_list.append(box3d_roi_inds)

                
                for i, bbox3d_corners in enumerate(aug_bboxes3d_rotated_corners):
                    box3d_roi_inds = kitti_utils.in_hull(pts_rect[:,:3], bbox3d_corners)
                    aug_boxes_pts_mask_list.append(box3d_roi_inds)
                
                n_points_in_bbox3d = np.sum(boxes_pts_mask_list[0])
                n_points_in_aug_bbox3d = np.sum(aug_boxes_pts_mask_list[0])

                if(n_points_in_aug_bbox3d == 0 or n_points_in_bbox3d == 0):
                    continue
                if(n_points_in_aug_bbox3d < 40 or n_points_in_bbox3d < 40):
                    continue

                cropped_bbox3d_pts = pts_rect[boxes_pts_mask_list[0],:]
                cropped_aug_bbox3d_pts = pts_rect[aug_boxes_pts_mask_list[0],:]

                kitti_obj['gt_box3d_pts'] = cropped_bbox3d_pts
                kitti_obj['aug_gt_box3d_pts'] = cropped_aug_bbox3d_pts

                yield kitti_obj
                