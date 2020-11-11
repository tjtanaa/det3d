import os
from det3d.pc_kitti_dataset import PCKittiSingleStagePointwiseDataset

import numpy as np
# Import package.
from point_viz.converter import PointvizConverter

if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = '/media/data3/tjtanaa/kitti_dataset'
    database_path = os.path.join(dataset_path, "gt_database")
    # gt_database_dir = os.path.join(database_path, "train_gt_database_level_Car.pkl")
    aug_dataset = PCKittiSingleStagePointwiseDataset(root_dir=dataset_path, split='train', 
                npoints =16384,
                classes =['Car'], mode='TRAIN', random_select =True,
                gt_database_dir=database_path, aug_hard_ratio=0.7)

    print("============ Grab Aug Sample =====================")
    save_viz_path = "/home/tan/tjtanaa/det3d/demos/pc_kitti_single_stage_pointwise_dataset"
    # Initialize and setup output directory.
    Converter = PointvizConverter(save_viz_path)
    for i in range(20):
        sample_info = aug_dataset.get_rpn_sample(i) 
        # sample_info['pts_input'] = pts_input
        # sample_info['pts_rect'] = aug_pts_rect
        # sample_info['pts_features'] = ret_pts_features
        # sample_info['rpn_cls_label'] = rpn_cls_label
        # sample_info['rpn_reg_label'] = rpn_reg_label
        # sample_info['gt_boxes3d'] = aug_gt_boxes3d
        # Pass data and create html files.
        sample_info['pts_rect'][:,1] *= -1 # mirror the y axis
        coors = sample_info['pts_rect']
        bbox_params = np.stack([sample_info['gt_boxes3d'][:,5], sample_info['gt_boxes3d'][:,3], sample_info['gt_boxes3d'][:,4],
                                sample_info['gt_boxes3d'][:,0], -(sample_info['gt_boxes3d'][:,1] - sample_info['gt_boxes3d'][:,3] / 2) , 
                                sample_info['gt_boxes3d'][:,2],
                                sample_info['gt_boxes3d'][:,6]], axis=1)
        Converter.compile("aug_sample_{}".format(i), coors=coors, intensity=sample_info['pts_features'][:,0],
                        bbox_params=bbox_params)

    
    print("============ Grab W/O Sample =====================")
    wo_aug_dataset = PCKittiSingleStagePointwiseDataset(root_dir=dataset_path, split='train', 
                npoints =16384,
                classes =['Car'], mode='TRAIN', random_select =True,
                gt_database_dir=None, aug_hard_ratio=0.7)
    for i in range(20):
        sample_info = wo_aug_dataset.get_rpn_sample(i) 
        # Pass data and create html files.
        sample_info['pts_rect'][:,1] *= -1 # mirror the y axis
        coors = sample_info['pts_rect']
        bbox_params = np.stack([sample_info['gt_boxes3d'][:,5], sample_info['gt_boxes3d'][:,3], sample_info['gt_boxes3d'][:,4],
                                sample_info['gt_boxes3d'][:,0], -(sample_info['gt_boxes3d'][:,1] - sample_info['gt_boxes3d'][:,3] / 2) , 
                                sample_info['gt_boxes3d'][:,2],
                                sample_info['gt_boxes3d'][:,6]], axis=1)
        Converter.compile("wo_aug_sample_{}".format(i), coors=coors, intensity=sample_info['pts_features'][:,0],
                        bbox_params=bbox_params)




    