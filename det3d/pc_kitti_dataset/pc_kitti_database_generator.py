"""
    This is a module to prepare dataset for point cloud only 3d detection
    It includes:
    1) Ground Truth Database Generator
    2) Data Loader
"""



import os
import numpy as np 
import pickle
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List
from det3d.kitti_dataset.utils import kitti_utils
from det3d.kitti_dataset.kitti_dataset_base import KittiDatasetBase


class PCKittiDatabaseGenerator(KittiDatasetBase):
    """[summary]

    Args:
        KittiDatasetBase ([type]): [description]
    """
    def __init__(self, root_dir, split='train', classes:List[str]=['Car'], **kwargs):
        print("PCKittiDatabaseGenerator\t: root\t:", root_dir)
        super().__init__(root_dir, split=split)
        self.gt_database = None
        self.classes = ['Background']
        classes.sort()
        self.classes.extend(classes)
        # if classes == 'Car':
        #     self.classes = ('Background', 'Car')
        # elif classes == 'People':
        #     self.classes = ('Background', 'Pedestrian', 'Cyclist')
        # elif classes == 'Pedestrian':
        #     self.classes = ('Background', 'Pedestrian')
        # elif classes == 'Cyclist':
        #     self.classes = ('Background', 'Cyclist')
        # else:
        #     assert False, "Invalid classes: %s" % classes

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def generate_gt_database(self, save_path:str) -> Any:
        gt_database = []
        for idx, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)
            print('process gt sample (id=%06d)' % sample_id)

            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3] # [0 ~ 1]

            obj_list = self.filtrate_objects(self.get_label(sample_id)) # are labels

            gt_boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
            for k, obj in enumerate(obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry

            if gt_boxes3d.__len__() == 0:
                print('No gt object')
                continue

            
            bboxes3d = kitti_utils.objs_to_boxes3d(obj_list)

            boxes_pts_mask_list = []

            bboxes3d_rotated_corners = kitti_utils.boxes3d_to_corners3d(bboxes3d)

            for i, bbox3d_corners in enumerate(bboxes3d_rotated_corners):
                box3d_roi_inds = kitti_utils.in_hull(pts_rect[:,:3], bbox3d_corners)
                boxes_pts_mask_list.append(box3d_roi_inds)

            total_mask = boxes_pts_mask_list[0]
            for k in range(boxes_pts_mask_list.__len__()):
                pt_mask_flag = boxes_pts_mask_list[k]
                total_mask |= pt_mask_flag
                if(np.sum(total_mask) == 0):
                    print("Bbox \t:", k, " does not enclose any points")
                cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                cur_pts_intensity = pts_intensity[pt_mask_flag].astype(np.float32)
                sample_dict = {'sample_id': sample_id,
                               'cls_type': obj_list[k].cls_type,
                               'gt_box3d': gt_boxes3d[k],
                               'points': cur_pts,
                               'intensity': cur_pts_intensity,
                               'obj': obj_list[k]}
                # print(cur_pts.shape)
                gt_database.append(sample_dict)

            # print(np.max(pts_intensity[total_mask]))
            # return pts_rect, pts_intensity, total_mask
            
        save_file_name = os.path.join(save_path, '%s_gt_database_level_%s.pkl' % (self.split, '-'.join(self.classes)))
        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        print('Save refine training sample info file to %s' % save_file_name)

    