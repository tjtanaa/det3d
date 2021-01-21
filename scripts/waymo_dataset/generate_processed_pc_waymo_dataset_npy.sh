# python generate_waymo_dataset_npy.py --num_workers=15 --split='train' --input_file_pattern='/home/tan/waymo2021/raw/training/segment-*.tfrecord'  --output_filebase='/home/tan/waymo2021/processed/training'
python generate_waymo_dataset_npy.py --num_workers=20 --split='val' --input_file_pattern='/home/tan/waymo2021/raw/validation/segment-*.tfrecord'  --output_filebase='/home/tan/waymo2021/processed/validation'

