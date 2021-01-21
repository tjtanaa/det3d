import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from det3d.pc_kitti_dataset import PCOptimKittiNpyDatabaseGenerator

import numpy as np
import time
# Import package.
from point_viz.converter import PointvizConverter

if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = '/media/data3/tjtanaa/kitti_dataset'
    database_path = os.path.join(dataset_path, "gt_database")
    # gt_database_dir = os.path.join(database_path, "train_gt_database_level_Car.pkl")
    npy_gt_database_path = '/media/data3/tjtanaa/kitti_dataset/kitti_npy'

    print("============ Generate NPY Database =====================")
    start = time.time()
    database_generator = PCOptimKittiNpyDatabaseGenerator(root_dir=dataset_path, split='train', 
                npoints =16384,
                classes =['Car'], random_select =True,
                gt_database_dir=database_path, aug_hard_ratio=0.7)
    database_generator.generate_database_npy(npy_gt_database_path)

    end = time.time()
    print("Takes (s): ", (end - start))




    