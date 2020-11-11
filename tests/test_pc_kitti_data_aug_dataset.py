import os
from det3d.pc_kitti_dataset import PCKittiAugmentedDataset

import numpy as np
# Import package.
from point_viz.converter import PointvizConverter

if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = '/media/data3/tjtanaa/kitti_dataset'
    database_path = os.path.join(dataset_path, "gt_database")
    # gt_database_dir = os.path.join(database_path, "train_gt_database_level_Car.pkl")
    aug_dataset = PCKittiAugmentedDataset(root_dir=dataset_path, split='train', 
                npoints =16384,
                classes =['Car'], mode='TRAIN', random_select =True,
                gt_database_dir=database_path, aug_hard_ratio=0.7)

    print("============ Grab Aug Sample =====================")
    save_viz_path = "/home/tan/tjtanaa/det3d/demos/pc_kitti_data_aug_dataset"
    # Initialize and setup output directory.
    Converter = PointvizConverter(save_viz_path)
    for i in range(20):
        sample_info = aug_dataset.get_sample(i) 
        # sample_info['pts_rect'] = aug_pts_rect # just [x y z]
        # sample_info['pts_features'] = ret_pts_features # [i]
        # sample_info['gt_boxes3d'] = aug_gt_boxes3d # note that the height is not the true height, you have to - h/2
        # sample_info['gt_cls_type_list'] = objs_to_cls_type_list # Object3d
        # Pass data and create html files.
        sample_info['pts_rect'][:,1] *= -1 # mirror the y axis
        coors = sample_info['pts_rect']
        bbox_params = np.stack([sample_info['gt_boxes3d'][:,5], sample_info['gt_boxes3d'][:,3], sample_info['gt_boxes3d'][:,4],
                                sample_info['gt_boxes3d'][:,0], -(sample_info['gt_boxes3d'][:,1] - sample_info['gt_boxes3d'][:,3] / 2) , 
                                sample_info['gt_boxes3d'][:,2],
                                sample_info['gt_boxes3d'][:,6]], axis=1)
        Converter.compile("aug_sample_{}".format(i), coors=coors, intensity=sample_info['pts_features'][:,0],
                        bbox_params=bbox_params)






    