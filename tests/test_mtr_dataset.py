from det3d.mtr_dataset.mtr_dataset_base import MTRDatasetBase



if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = "/home/tjtanaa/Documents/AKK/Project4-MTR"
    point_cloud_statistics_path = "/home/tjtanaa/Documents/Github/det3d/det3d/mtr_dataset/point_cloud_statistics"

    dataset = MTRDatasetBase(dataset_path, 'train', point_cloud_statistics_path)
    print("================== Dataset ======================")
    train_data_filename_list, train_directory_index = dataset.load_filenames_by_directory()
    print("Number of samples\t:", dataset.num_sample)

    # print("========= Generate Training Samples Metadata ====")
    # metadata_path = "/home/tjtanaa/Documents/AKK/Project4-MTR"
    # print("Save to \t:", metadata_path)
    # dataset.generate_training_meta_data(metadata_path)

    # val_dataset = MTRDatasetBase(dataset_path, split='val')
    # print("================== Validation dataset ===================")
    # print("Number of samples\t:", val_dataset.num_sample)

    # test_dataset = MTRDatasetBase(dataset_path, split='test')
    # print("================== Test dataset =========================")
    # print("Number of samples\t:", test_dataset.num_sample)