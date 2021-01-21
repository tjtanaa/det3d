import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from det3d.pc_kitti_dataset.pc_kitti_template_single_dataloader import PCKittiTemplateSingleDataLoader

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Select mode to do')
parser.add_argument('--mode', type=int, default=0,
                    help='mode 0: generate database mode 1: visualize generated database')
parser.add_argument('--dataset_path', type=str, default='/media/data3/tjtanaa/kitti_dataset',
                    help='path to kitti dataset')

args = parser.parse_args()
# Import package.
import pickle
from point_viz.converter import PointvizConverter

if __name__ == "__main__":
    # Path to the kitti dataset
    # dataset_path = '/media/data3/tjtanaa/kitti_dataset'
    dataset_path = args.dataset_path
    database_path = os.path.join(dataset_path, "gt_database")
    dataset = PCKittiTemplateSingleDataLoader(root_dir=dataset_path, split='train', classes=['Car'])
    if args.mode == 0:
        # dataset = PCKittiTemplateDatabaseGenerator(root_dir=dataset_path, split='train', classes=['Car', 'Pedestrian', 'Person_sitting'])
        

        # os.makedirs(database_path, exist_ok=True)
        # print("============ Generate Database =====================")
        # dataset.generate_gt_database(database_path) # we only need to generate database to augment the scene during training
        pass

    elif args.mode == 1:
        save_viz_path = os.path.join(currentdir, 'visualization/test_pc_kitti_template_single_dataloader')
        Converter = PointvizConverter(save_viz_path)

        train_gen = dataset.generator()

        for idx in range(10):
            kitti_obj = next(train_gen)
            
            pts_rect = kitti_obj['points']
            fg_mask = kitti_obj['fg_mask']
            gt_box3d = kitti_obj['gt_box3d']
            aug_gt_box3d = kitti_obj['aug_gt_box3d']

            kitti_bbox_params = [[ gt_box3d[5],
                                    gt_box3d[3],
                                    gt_box3d[4],
                                    gt_box3d[0],
                                    -gt_box3d[1] + gt_box3d[3] / 2,
                                    gt_box3d[2],
                                    gt_box3d[6],
                                    "Green" ],
                                [ aug_gt_box3d[5],
                                    aug_gt_box3d[3],
                                    aug_gt_box3d[4],
                                    aug_gt_box3d[0],
                                    -aug_gt_box3d[1] + aug_gt_box3d[3] / 2,
                                    aug_gt_box3d[2],
                                    aug_gt_box3d[6],
                                    "Magenta" ]]


            # gt_box3d_xyz = [gt_box3d[0],
            #                 -gt_box3d[1] + gt_box3d[3] / 2,
            #                 gt_box3d[2]]
            # kitti_bbox_params = [[ gt_box3d[5],
            #                         gt_box3d[3],
            #                         gt_box3d[4],
            #                         0,
            #                         0,
            #                         0,
            #                         gt_box3d[6],
            #                         "Green" ]]

            pts_rect[:,1] *= -1
            # pts_rect -= gt_box3d_xyz

            intensity = kitti_obj['intensity']
            masked_intensity = intensity * fg_mask

            Converter.compile("pc_kitti_template_generator_sample_{}".format(idx), coors=pts_rect, intensity=masked_intensity, 
                                bbox_params = kitti_bbox_params)
            if idx == 10:
                exit()

    # pts_rect, pts_intensity, total_mask  = dataset.generate_gt_database(database_path)

    # pts_rect[:,1] *= -1 
    # # print(np.sum(total_mask))

    # # pts_intensity[total_mask] = 0

    # save_viz_path = "/home/tan/tjtanaa/det3d/demos"
    # Initialize and setup output directory.
    # Pass data and create html files.
