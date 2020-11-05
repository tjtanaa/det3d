from det3d.pc_kitti_dataset.pc_kitti_database_generator import PCKittiDatabaseGenerator

if __name__ == "__main__":
    # Path to the kitti dataset
    dataset_path = '/media/data3/tjtanaa/kitti_dataset'
        
    dataset = PCKittiDatabaseGenerator(root_dir=dataset_path, split='train')
    
    # os.makedirs(args.save_dir, exist_ok=True)

    dataset.generate_gt_database()

