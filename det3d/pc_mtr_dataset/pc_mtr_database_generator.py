"""
    This is a module to prepare dataset for point cloud only 3d detection
    It includes:
    1) Ground Truth Database Generator
    2) Data Loader
"""



import os
import numpy as np 
import pickle
from det3d.mtr_dataset.utils import mtr_utils
from det3d.mtr_dataset.mtr_dataset_base import MTRDatasetBase
from det3d.point_cloud_utils.transformation import transform


class PCMTRDatabaseGenerator(MTRDatasetBase):
    """[summary]

    Args:
        MTRDatasetBase ([type]): [description]
    """
    def __init__(self, root_dir, 
                split='train', 
                point_cloud_statistics_path:str = "det3d/mtr_dataset/point_cloud_statistics",
                classes='pedestrian'):
        # print("PCMTRDatabaseGenerator\t: root\t:", root_dir)
        super().__init__(root_dir, split=split, point_cloud_statistics_path=point_cloud_statistics_path)
        self.gt_database = None
        if classes == 'pedestrian':
            self.classes = ('background', 'pedestrian')
        # elif classes == 'People':
        #     self.classes = ('Background', 'Pedestrian', 'Cyclist')
        # elif classes == 'Pedestrian':
        #     self.classes = ('Background', 'Pedestrian')
        # elif classes == 'Cyclist':
        #     self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        return obj_list
        raise NotImplementedError
        # TO BE IMPLEMENTED
        # remove the invalid bounding boxes
        # remove the bounding boxes that are too small
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def generate_gt_database(self, save_path:str) -> None:
        gt_database = []

        # generate a text file containing the name of files being used as train and test


        for sample_id, sample_filename in enumerate(self.sample_list):
            sample_id = int(sample_id)
            print('process gt sample (id=%06d) \t %s' % (sample_id, sample_filename))

            # pts_lidar = self.get_lidar(sample_id)
            pts_rect = self.get_lidar(sample_id)

            # pts_rect = np.stack([pts_rect[:,1],pts_rect[:,2], pts_rect[:,0], pts_rect[:,3]], axis=-1)
            # print(pts_rect.shape)
            # calib = self.get_calib(sample_id)
            # pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_features = pts_rect[:, 3:]

            obj_list = self.filtrate_objects(self.get_label(sample_id)) # are labels

            gt_boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
            for k, obj in enumerate(obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry

            if gt_boxes3d.__len__() == 0:
                print('No gt object')
                continue

            # if(sample_id == 203):
            #     print(gt_boxes3d.shape)

            bboxes3d = mtr_utils.objs_to_boxes3d(obj_list)

            boxes_pts_mask_list = []

            bboxes3d_rotated_corners = mtr_utils.boxes3d_to_corners3d(bboxes3d)

            for i, bbox3d_corners in enumerate(bboxes3d_rotated_corners):
                box3d_roi_inds = mtr_utils.in_hull(pts_rect[:,:3], bbox3d_corners)
                boxes_pts_mask_list.append(box3d_roi_inds)

            # total_mask = boxes_pts_mask_list[0]
            for k in range(boxes_pts_mask_list.__len__()):
                pt_mask_flag = boxes_pts_mask_list[k]
                # total_mask |= pt_mask_flag
                # print("Sum of points: ", np.sum(pt_mask_flag))
                if(np.sum(pt_mask_flag) == 0):
                    print("Zero Point Bounding Boxes")
                    continue
                cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                cur_pts_features = pts_features[pt_mask_flag].astype(np.float32)
                sample_dict = {'sample_id': sample_id,
                               'cls_type': obj_list[k].cls_type,
                               'gt_box3d': gt_boxes3d[k],
                               'points': cur_pts,
                               'features': cur_pts_features,
                               'obj': obj_list[k]}
                # print(cur_pts.shape)
                gt_database.append(sample_dict)

        save_file_name = os.path.join(save_path, '%s_mtr_gt_database_level_%s.pkl' % (self.split, self.classes[-1]))
        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        print('Save refine training sample info file to %s' % save_file_name)

    