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
from tqdm import tqdm

class PCOptimKittiNpyDatabaseGenerator(PCKittiAugmentedDataset):
    """
        This is a database that generates information that are necessary 
        for pointwise object detection

        optimized data pipeline should have the following 
        project directories:

        path_to_training_data:
        np.save(join(output_home, 'lidar_points_{}.npy'.format(task)), np.array(output_lidar_points, dtype=object))
        np.save(join(output_home, 'bbox_labels_{}.npy'.format(task)), np.array(output_bbox, dtype=object))
        np.save(join(output_home, 'trans_matrix_{}.npy'.format(task)), np.array(output_trans_matrix, dtype=object))
        np.save(join(output_home, 'ground_plane_{}.npy'.format(task)), np.array(output_plane, dtype=object))
        np.save(join(output_home, 'img_size_{}.npy'.format(task)), np.array(output_image_size, dtype=object))
        np.save(join(output_home, 'img_{}.npy'.format(task)), np.array(output_image, dtype=object))
        np.save(join(output_home, "object_collections_{}.npy".format(task)), np.array(output_object_points, dtype=object))
        np.save(join(output_home, "bbox_collections_{}.npy".format(task)), np.array(output_object_bboxes, dtype=object))

                
            



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

    def generate_database_npy(self, save_npy_path):
        # this function has to save the database element into the following format
        
        # lidar_pts : [x,y,z,intensity]
        #   filename: lidar_pts_<split>_<ClassName>_{sample_id wrt to the class}
        # bbox : [x,y,z,w,l,h,ry]
        #   filename: bbox_<split>_<ClassName>_{sample_id wrt to the class}

        # np.save(join(output_home, 'lidar_points_{}.npy'.format(task)), np.array(output_lidar_points, dtype=object))
        # np.save(join(output_home, 'bbox_labels_{}.npy'.format(task)), np.array(output_bbox, dtype=object))
        # np.save(join(output_home, 'trans_matrix_{}.npy'.format(task)), np.array(output_trans_matrix, dtype=object))
        # np.save(join(output_home, 'ground_plane_{}.npy'.format(task)), np.array(output_plane, dtype=object))
        # np.save(join(output_home, 'img_size_{}.npy'.format(task)), np.array(output_image_size, dtype=object))
        # np.save(join(output_home, 'img_{}.npy'.format(task)), np.array(output_image, dtype=object))
        # np.save(join(output_home, "object_collections_{}.npy".format(task)), np.array(output_object_points, dtype=object))
        # np.save(join(output_home, "bbox_collections_{}.npy".format(task)), np.array(output_object_bboxes, dtype=object))

        sample_id_list = []
        lidar_points_list = [] # [x, y, z, intensity]
        bbox_labels_list = [] # [x,y,z,w,l,h,ry,category, difficulty]
        trans_matrix_list = [] # []
        ground_plane_list = [] # []
        img_size_list = []
        img_list = []
        object_collections_list = [] # [x, y, z, intensity]
        bbox_collections_list = [] # [x,y,z,w,l,h,ry,category]
        
        object_collections_list = [[], [], [], []] #
        bbox_collections_list = [[], [], [], []]

        for index in tqdm(range(100)):
        # for index in tqdm(range(len(self.sample_id_list))):
            # print(len(self.sample_id_list))
            sample_id = int(self.sample_id_list[index])
            # print("sample_id: ", sample_id)
            # if sample_id < 10000:
            calib = self.get_calib(sample_id)
            
            Tr_velo_to_cam = np.concatenate([calib.V2C, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

            R0_rect = np.eye(4).astype('float32')
            R0_rect[:3, :3] = calib.R0

            calib_list = [calib.P2, R0_rect, Tr_velo_to_cam]

            img = self.get_image(sample_id)
            img_shape = self.get_image_shape(sample_id)
            img_size = img_shape

            pts_lidar = self.get_lidar(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]

            # get valid point (projected points should be in image)
            pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
            pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
            pts_rect = pts_rect[pts_valid_flag][:, 0:3]
            pts_intensity = pts_intensity[pts_valid_flag]
            pts_rect_lidar_coordinate = calib.rect_to_lidar(pts_rect)

            # get road plane
            ground_plane = self.get_road_plane(sample_id)

            if self.split == 'train' or self.split == 'val':
                gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
                gt_boxes3d = kitti_utils.objs_to_boxes3d_include_cls_diff(gt_obj_list)
                bbox_labels_list.append(gt_boxes3d)

            sample_id_list.append(sample_id)
            trans_matrix_list.append(calib_list)
            # print(len(trans_matrix_list[0]))
            # for j in range(len(calib_list)):
            #     print(j, "\t", calib_list[j].shape)
            img_list.append(img)
            img_size_list.append(img_size)
            lidar_points_list.append(np.concatenate([pts_rect_lidar_coordinate, pts_intensity[:,np.newaxis]], axis=-1))
            ground_plane_list.append(ground_plane)

        task = self.split
        np.save(os.path.join(save_npy_path, 'sample_id_list_{}.npy'.format(task)), np.array(sample_id_list, dtype=object))
        np.save(os.path.join(save_npy_path, 'lidar_points_{}.npy'.format(task)), np.array(lidar_points_list, dtype=object))
        np.save(os.path.join(save_npy_path, 'bbox_labels_{}.npy'.format(task)), np.array(bbox_labels_list, dtype=object))
        np.save(os.path.join(save_npy_path, 'trans_matrix_{}.npy'.format(task)), np.array(trans_matrix_list, dtype=object))
        np.save(os.path.join(save_npy_path, 'ground_plane_{}.npy'.format(task)), np.array(ground_plane_list, dtype=object))
        np.save(os.path.join(save_npy_path, 'img_size_{}.npy'.format(task)), np.array(img_size_list, dtype=object))
        np.save(os.path.join(save_npy_path, 'img_{}.npy'.format(task)), np.array(img_list, dtype=object))
        # np.save(os.path.join(save_npy_path, 'trans_matrix_{}.npy'.format(task)), np.array(trans_matrix_list, dtype=object))
        

        # if self.gt_database is not None:
        #     for i, (k, v) in enumerate(self.gt_database.items()): 
        #         # convert the gt database


        # np.save(join(output_home, "object_collections_{}.npy".format(task)), np.array(output_object_points, dtype=object))
        # np.save(join(output_home, "bbox_collections_{}.npy".format(task)), np.array(output_object_bboxes, dtype=object))
