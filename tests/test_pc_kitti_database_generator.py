import os
from det3d.pc_kitti_dataset.pc_kitti_database_generator import PCKittiDatabaseGenerator

import numpy as np
# Import package.
# from point_viz.converter import PointvizConverter

if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = '/media/data3/tjtanaa/kitti_dataset'
        
    dataset = PCKittiDatabaseGenerator(root_dir=dataset_path, split='train', classes=['Car', 'Pedestrian', 'Person_sitting'])
    dataset = PCKittiDatabaseGenerator(root_dir=dataset_path, split='train', classes=['Car'])

    database_path = os.path.join(dataset_path, "gt_database")
    os.makedirs(database_path, exist_ok=True)
    print("============ Generate Database =====================")
    dataset.generate_gt_database(database_path) # we only need to generate database to augment the scene during training

    # pts_rect, pts_intensity, total_mask  = dataset.generate_gt_database(database_path)

    # pts_rect[:,1] *= -1 
    # # print(np.sum(total_mask))

    # # pts_intensity[total_mask] = 0

    # save_viz_path = "/home/tan/tjtanaa/det3d/demos"
    # # Initialize and setup output directory.
    # Converter = PointvizConverter(save_viz_path)
    # # Pass data and create html files.
    # Converter.compile("pc_kitti_dataset", coors=pts_rect, intensity=pts_intensity)
