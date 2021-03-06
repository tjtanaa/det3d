import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from det3d.kitti_dataset.kitti_dataset_base import KittiDatasetBase



if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = '/media/data3/tjtanaa/kitti_dataset'

    train_dataset = KittiDatasetBase(dataset_path, split='train')
    print("================== Training dataset =====================")
    print("Number of samples\t:", train_dataset.num_sample)

    val_dataset = KittiDatasetBase(dataset_path, split='val')
    print("================== Validation dataset ===================")
    print("Number of samples\t:", val_dataset.num_sample)

    test_dataset = KittiDatasetBase(dataset_path, split='test')
    print("================== Test dataset =========================")
    print("Number of samples\t:", test_dataset.num_sample)