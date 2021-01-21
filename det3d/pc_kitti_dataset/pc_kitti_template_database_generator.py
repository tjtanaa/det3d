"""
    This is a module to prepare dataset for icp point cloud 3d detection.
    There will be two types of generators:
    It includes:
    1) Ground Truth Database Generator # with background
      - Requirements:
        1. Enlarge the bounding box to include more points
        2. Still save the original Ground Truth Bounding boxes
    2) Purturbed Database Generator # with points cropped from perturbed bounding box
      - Purpose:
        1. Only for offline augmentation usage
    3) SingleDatabase Loader
      - Requirements:
        1. Randomize and purturb the bounding boxes
        2. Extract only points within the bounding boxes
        3. Return the purturbation parameters, xyzwlh,ry
    4) PairDatabase Loader
      - Requirements:
        1. Randomize and purturb the bounding boxes
        2. Extract only points within the bounding boxes
        3. Return the purturbation parameters, xyzwlh,ry, 
        pairs of point clouds (gt point cloud, purturbed point cloud)
      - Possible usecase:
        1. Test whether registration from gt point cloud to purturbed point cloud works
"""



import os
import numpy as np 
import pickle
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List
from det3d.kitti_dataset.utils import kitti_utils
from det3d.kitti_dataset.kitti_dataset_base import KittiDatasetBase


class PCKittiTemplateDatabaseGenerator(KittiDatasetBase):
    """
        This is a class to generate instance-wise database of specific classes: e.g. Car
        It will be used to test the effectiveness of registration under different 
        policies as specified in the https://github.com/tjtanaa/icp-cuda-pytorch-extension.git.

        It generates a database with following contents:
        sample_dict = {'sample_id': sample_id,
                        'cls_type': obj_list[k].cls_type,
                        'gt_box3d': gt_boxes3d[k],
                        'points': cur_pts, # points with background
                        'fg_mask': fg_mask,
                        'intensity': cur_pts_intensity,
                        'obj': obj_list[k]}


    Args:
        KittiDatasetBase ([type]): [description]
    """
    def __init__(self, root_dir, split='train', classes:List[str]=['Car'], **kwargs):
        print("PCKittiTemplateDatabaseGenerator\t: root\t:", root_dir)
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
            # for k, obj in enumerate(obj_list):
            #     gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
            #         = obj.pos, obj.h, obj.w, obj.l, obj.ry

            gt_boxes3d = kitti_utils.objs_to_boxes3d(obj_list)
            if gt_boxes3d.__len__() == 0:
                print('No gt object')
                continue

            

            # print("gt_boxes3d: ", gt_boxes3d)
            enlarged_bboxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, 2)
            # print("enlarged_bboxes3d: ", enlarged_bboxes3d)
            # exit()

            boxes_pts_mask_list = []
            enlarged_boxes_pts_mask_list = []

            bboxes3d_rotated_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d)
            enlarged_bboxes3d_rotated_corners = kitti_utils.boxes3d_to_corners3d(enlarged_bboxes3d)


            for i, bbox3d_corners in enumerate(enlarged_bboxes3d_rotated_corners):
                box3d_roi_inds = kitti_utils.in_hull(pts_rect[:,:3], bbox3d_corners)
                enlarged_boxes_pts_mask_list.append(box3d_roi_inds)

            
            for i, bbox3d_corners in enumerate(bboxes3d_rotated_corners):
                box3d_roi_inds = kitti_utils.in_hull(pts_rect[enlarged_boxes_pts_mask_list[i],:3], bbox3d_corners)
                boxes_pts_mask_list.append(box3d_roi_inds)

            # total_mask = boxes_pts_mask_list[0]

            for k in range(boxes_pts_mask_list.__len__()):
                pt_mask_flag = enlarged_boxes_pts_mask_list[k]
                # total_mask |= pt_mask_flag
                if(np.sum(pt_mask_flag) == 0):
                    print("Bbox \t:", k, " does not enclose any points")
                
                # print("number of pt_mask_flag: ", np.sum(pt_mask_flag), "\t length of fg_mask: ",
                #          len(boxes_pts_mask_list[k]), "\t number of valid fg_mask: ", np.sum(boxes_pts_mask_list[k]))

                cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                cur_pts_intensity = pts_intensity[pt_mask_flag].astype(np.float32)
                sample_dict = {'sample_id': sample_id,
                               'cls_type': obj_list[k].cls_type,
                               'gt_box3d': gt_boxes3d[k],
                               'points': cur_pts,
                               'fg_mask': boxes_pts_mask_list[k],
                               'intensity': cur_pts_intensity,
                               'obj': obj_list[k]}
                # print(cur_pts.shape)
                gt_database.append(sample_dict)
            # print(np.max(pts_intensity[total_mask]))
            # return pts_rect, pts_intensity, total_mask
            
        save_file_name = os.path.join(save_path, '%s_template_gt_database_level_%s.pkl' % (self.split, '-'.join(self.classes)))
        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        print('Save refine training sample info file to %s' % save_file_name)

    