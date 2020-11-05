from det3d.pc_mtr_dataset.pc_mtr_database_generator import PCMTRDatabaseGenerator



if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = "/home/tjtanaa/Documents/AKK/Project4-MTR"
    point_cloud_statistics_path = "/home/tjtanaa/Documents/Github/det3d/det3d/mtr_dataset/point_cloud_statistics"

    train_dataset = PCMTRDatabaseGenerator(dataset_path, 'train', point_cloud_statistics_path)
    print("================== Dataset ======================")
    print("Number of samples\t:", train_dataset.num_sample)

    print("====== Generate Train-Ground Truth Database ===========")
    database_path = "/home/tjtanaa/Documents/AKK/Project4-MTR/Database"
    print("Save to \t:", train_dataset)
    train_dataset.generate_gt_database(database_path)

    test_dataset = PCMTRDatabaseGenerator(dataset_path, 'test', point_cloud_statistics_path)
    print("================== Dataset ======================")
    print("Number of samples\t:", test_dataset.num_sample)

    print("====== Generate Test-Ground Truth Database ===========")
    database_path = "/home/tjtanaa/Documents/AKK/Project4-MTR/Database"
    print("Save to \t:", database_path)
    test_dataset.generate_gt_database(database_path)

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