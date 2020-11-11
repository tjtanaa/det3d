""" Remove points outside the image coordinates

    This script is from https://github.com/qianguih/voxelnet/blob/master/data/crop.py
    "credited to https://github.com/dtczhl/dtc-KITTI-For-Beginners"
"""

import numpy as np
# from scipy.misc import imread
import imageio
import os
import sys
import json
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# print(BASE_DIR)
sys.path.append('/home/tan/tjtanaa/RandLADet')


import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d_python as iou3d

# from helper_tool import ConfigKitti as cfg
CAM = 2

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    # points = points[:, :3]  # exclude luminance
    return points


def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [line.split()[1:] for line in lines][:-1]
    # print(
    #     "calibration line: ", lines
    # )
    #
    P = np.array(lines[CAM]).reshape(3, 4)
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')

    # print("P")
    # print(P)

    # print("Tr_velo_to_cam")
    # print(Tr_velo_to_cam)

    # print("R_cam_to_rect")
    # print(R_cam_to_rect)

    return P, Tr_velo_to_cam, R_cam_to_rect


def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
        points can be directly multiplied by the camera projection matrix'''
    pts3d = pts3d_raw
    # Reflectance > 0
    indices = pts3d[:, 3] > 0
    pts3d = pts3d[indices, :]
    pts3d[:, 3] = 1
    return pts3d.transpose(), indices


def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
        numpy array. Returns the 2D projection of the points that
        are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2, :] >= 0)
    pts2d_cam = Prect.dot(pts3d_cam[:, idx])
    return pts3d_cam[:, idx], pts2d_cam / pts2d_cam[2, :], idx
    # return pts3d[:, idx], pts2d_cam / pts2d_cam[2, :], idx


def filter_object(bbox_objs, level_difficulty, class_of_interest):
    # return bbox_objs
    objects = []

    for obj in bbox_objs:
        if obj.level in level_difficulty and obj.cls_id in class_of_interest:
            objects.append(obj)
    return objects
    

def load_label(frame, level_difficulty, class_of_interest, verbose=False):
    """
        Params:
        frame: frame ID
        level_difficulty: 1, 2, 3, 4 (Easy, Moderate, Hard, Unknown)
        class_of_interest: [1,2,3,4] {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}

        label_stat = {'frame_id', 'truncated', 'occlusion' , 'labels', '3D_bboxes', '2D_bboxes'}

        'truncated' : Float from 0 (non-truncated) to 1 (truncated), 
                        where truncated refers to the object leaving image boundaries
        'occlusion' : Integer (0,1,2,3) indicating occlusion state: 
                        0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
        'labels' : class label
        '3D_bboxes' : [[x,y,z,w,h,l,ry]] in camera coordinates
        '2D_bboxes' : 2D bounding box of object in the image (0-based index): 
                        contains left, top, right, bottom pixel coordinates
    """

    # label_stat = {'frame_id': frame}
    label_filename = os.path.join(LABEL_ROOT, '{0:06d}.txt'.format(frame))
    assert os.path.exists(label_filename)
    objs =  kitti_utils.get_objects_from_label(label_filename)
    objs = filter_object(objs, level_difficulty, class_of_interest)
    # for obj in objs:
    #     if obj is not None:
    #         print(obj)
    
    return objs

    # with open(label_filename) as f_label:
    #     lines = f_label.readlines()
        
    #     for line in lines:
    #         line = line.strip('\n').split()
            
    #         print("line : ", line)
    #         label_stat[line[0]] = label_stat.setdefault(line[0], 0) + 1
    #         kitti_get_objects_from_label
    # return label_stat

def align_img_and_pc(img_dir, pc_dir, calib_dir):
    img = imageio.imread(img_dir)
    # img = imread(img_dir)
    pts = load_velodyne_points(pc_dir)
    P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir)

    pts3d, indices = prepare_velo_points(pts)
    # pts3d_ori = pts3d.copy()
    reflectances = pts[indices, 3]
    pts3d, pts2d_normed, idx = project_velo_points_in_img(pts3d, Tr_velo_to_cam, R_cam_to_rect, P)
    # print reflectances.shape, idx.shape
    reflectances = reflectances[idx]
    # print reflectances.shape, pts3d.shape, pts2d_normed.shape
    # assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

    rows, cols = img.shape[:2]

    points = []
    for i in range(pts2d_normed.shape[1]):
        c = int(np.round(pts2d_normed[0, i]))
        r = int(np.round(pts2d_normed[1, i]))
        if c < cols and r < rows and r > 0 and c > 0:
            color = img[r, c, :]
            point = [pts3d[0, i], pts3d[1, i], pts3d[2, i], reflectances[i], color[0], color[1], color[2],
                     pts2d_normed[0, i], pts2d_normed[1, i]]
            points.append(point)

    points = np.array(points)
    return points



if __name__ == '__main__':

    with open('./lib/datasets/explore_config.json', 'r') as f:
        json_obj = json.load(f)

        BASE_DIR = os.path.join(json_obj['DATA_PATH'], 'training')
        
        # path to data_object_image_2/training/image_2
        IMG_ROOT = BASE_DIR + '/image_2/'
        # path to data_object_velodyne/training/velodyne
        PC_ROOT = BASE_DIR + '/velodyne/'
        # path to data_object_calib/training/calib
        CALIB_ROOT = BASE_DIR + '/calib/'

        LABEL_ROOT = BASE_DIR + '/label_2/'

        # path to the folder for saving cropped point clouds
        SAVE_ROOT = os.path.join(json_obj['SAVE_PATH'], 'training')
        SAVE_PC_PATH = os.path.join(SAVE_ROOT, 'point_cloud')
        SAVE_LABELS_PATH = os.path.join(SAVE_ROOT, 'labels')
        if not os.path.exists(SAVE_ROOT):
            os.makedirs(SAVE_ROOT)
        if not os.path.exists(SAVE_PC_PATH):
            os.makedirs(SAVE_PC_PATH)
        if not os.path.exists(SAVE_LABELS_PATH):
            os.makedirs(SAVE_LABELS_PATH)

        sub_sampling_ratio = json_obj['SUB_SAMPLING_RATIO']
        num_points = json_obj['NUM_POINTS']
        level_difficulty = json_obj['DIFFICULTY']
        class_of_interest = json_obj['CLASS_OF_INTEREST']
        resampling_frequency = json_obj['RESAMPLING_FREQUENCY']
        interested_sample_ratio = json_obj['INTERESTED_SAMPLE_RATIO']
        print("level_difficulty ", level_difficulty)
    downsampling_factor = 1
    for factor in sub_sampling_ratio:
        downsampling_factor *= factor
    
    num_pts_left = int(num_points * downsampling_factor)

    total_objs = []
    x_list = []
    y_list = []
    z_list = []
    l_list = []
    w_list = []
    h_list = []
    ry_list = [] # 7481 int(7481*0.1)
    num_pts_per_scene = []
    num_pts_per_object = []
    skip_count = 0
    num_labels_retained = []
    proportion_fgbg_sub_points = []
    num_fgbg_sub_points = []
    # avg of how many object has been lost due to downsampling

    for frame in range( int(7481*interested_sample_ratio)):

        print('--- processing {0:06d}'.format(frame))

        img_dir = os.path.join(IMG_ROOT,  '{0:06d}.png'.format(frame))
        pc_dir = os.path.join(PC_ROOT, '{0:06d}.bin'.format(frame))
        calib_dir = os.path.join(CALIB_ROOT, '{0:06d}.txt'.format(frame))

        labels = load_label(frame, level_difficulty=level_difficulty, class_of_interest = class_of_interest)
        # print(labels)
        if len(labels) == 0:
            skip_count += 1
            print("skip: ", skip_count)
            continue
        total_objs.append(len(labels))
        for label in labels:
            x_list.append(label.pos[0])
            y_list.append(label.pos[1])
            z_list.append(label.pos[2])
            l_list.append(label.l)
            w_list.append(label.w)
            h_list.append(label.h)
            ry_list.append(label.ry)
        points = align_img_and_pc(img_dir, pc_dir, calib_dir)
        # print("numpoints: ", len(points))
        num_pts_per_scene.append(len(points))


        # Get the foreground and background label
        bboxes3d = kitti_utils.objs_to_boxes3d(labels)
        # print("Number of bboxes: ",len(bboxes3d))
        bboxes3d_rotated_corners = kitti_utils.boxes3d_to_corners3d(bboxes3d)
        box3d_roi_inds_overall = None
        sub_box3d_roi_inds_overall = None
        valid_labels = []
        for i, bbox3d_corners in enumerate(bboxes3d_rotated_corners):
            # print("bboxes3d_rotated_corners: ", bboxes3d_rotated_corners[i])
            box3d_roi_inds = kitti_utils.in_hull(points[:,:3], bbox3d_corners)
            iou_3d, iou_2d = iou3d.box3d_iou(bbox3d_corners, bbox3d_corners - 0.0001)
            print("iou_3d: ", iou_3d, "\t iou_2d: ", iou_2d)
            # box3d_roi_inds = kitti_utils.in_hull(bbox3d_corners[:,:3], bbox3d_corners)
            # print("xmin: ", np.min(points[:,0]), " xmax: ", np.max(points[:,0]))
            # print("ymin: ", np.min(points[:,1]), " ymax: ", np.max(points[:,1]))
            # print("zmin: ", np.min(points[:,2]), " zmax: ", np.max(points[:,2]))
            # pc_filter = kitti_utils.extract_pc_in_box3d(points[:,:3], bbox3d_corners)
            # sub_pc_filter = kitti_utils.extract_pc_in_box3d(points[:,:3], bbox3d_corners)
            # print("pc_filter.shape: ", pc_filter[0].shape)
            # print("interested indices shape: ", box3d_roi_inds.shape)
            # print("interested indices: ", box3d_roi_inds)
            # print("number of obj points: ", np.sum(box3d_roi_inds))
            if  np.sum(box3d_roi_inds) < 10:
                continue
            valid_labels.append(labels[i])
            # print(type(box3d_roi_inds))
            if box3d_roi_inds_overall is None:
                box3d_roi_inds_overall = box3d_roi_inds
            else:
                box3d_roi_inds_overall = np.logical_or(box3d_roi_inds_overall,box3d_roi_inds)

            num_pts_per_object.append(np.sum(box3d_roi_inds))

            avg_times_label_retained = 0
            for i in range(resampling_frequency):
                indices = np.arange(points.shape[0])
                if num_points > points.shape[0]:
                    num_pad_pts = num_points - points.shape[0]
                    pad_indices = np.random.choice(indices, size=num_pad_pts)
                    indices = np.hstack([indices, pad_indices])
                    np.random.shuffle(indices)
                else:
                    np.random.shuffle(indices)
                    indices = indices[:num_points]

                output_indices = indices[:num_pts_left]
                sub_points = points[output_indices,:]
                sub_box3d_roi_inds = kitti_utils.in_hull(sub_points[:,:3], bbox3d_corners)

                if np.sum(sub_box3d_roi_inds) > 0:
                    avg_times_label_retained+=1
            
            num_labels_retained.append(avg_times_label_retained/resampling_frequency)

        if box3d_roi_inds_overall is None:
            skip_count += 1
            print("skip: ", skip_count)
            continue
        print("number of obj points: ", np.sum(box3d_roi_inds_overall.astype(np.uint32)))
        print("number of valid labels: ", len(valid_labels))
        if np.sum(box3d_roi_inds_overall.astype(np.uint32)) < 1e-5:
            skip_count += 1
            print("skip: ", skip_count)
            continue

        # get the foreground/background label
        fgbg = np.zeros([points.shape[0],1])
        fgbg[box3d_roi_inds_overall,0] = 1
        print("fgbg: ", np.sum(fgbg))

        stacked_points = np.hstack([points[:,:4], fgbg]) # [x,y,z, reflectance, fgbg]
        print("stacked points.shape: ", stacked_points.shape)

        indices = np.arange(points.shape[0])
        if num_points > points.shape[0]:
            num_pad_pts = num_points - points.shape[0]
            pad_indices = np.random.choice(indices, size=num_pad_pts)
            indices = np.hstack([indices, pad_indices])
            np.random.shuffle(indices)
        else:
            np.random.shuffle(indices)
            indices = indices[:num_points]

        output_indices = indices[:num_pts_left]
        sub_points = stacked_points[output_indices,:]
        num_fg_points = np.sum(sub_points[:,4])
        proportion_fgbg_sub_points.append(num_fg_points/num_pts_left * 100.0)
        num_fgbg_sub_points.append(num_fg_points)
        print("num_fg_points: ", num_fg_points)

        output_pc_name = os.path.join(SAVE_ROOT, 'point_cloud/{0:06d}.bin'.format(frame))
        output_label_name = os.path.join(SAVE_ROOT, 'labels/{0:06d}.pkl'.format(frame))
        # points[:, :4].astype('float32').tofile(output_pc_name)
        # stacked_points.astype('float32').tofile(output_pc_name)
        sub_points.astype('float32').tofile(output_pc_name)
        with open(output_label_name, 'wb') as f:
            pickle.dump(valid_labels, f)

    num_labels_loss = np.ones_like(np.array(num_labels_retained)) - np.array(num_labels_retained)
    stats_list = [total_objs, x_list, y_list, z_list, l_list, w_list, h_list, ry_list , \
                num_pts_per_scene, num_pts_per_object, num_labels_retained, \
                num_labels_loss,
                num_fgbg_sub_points, proportion_fgbg_sub_points]
    headers = ['N','X', 'Y', 'Z', 'L', "W", "H", "RY", 
                "NPts/Scene", "NPts/Obj", "NObjsRetained",
                "NObjsLoss", 
                "NfgPts", "percentage FgPts"]
    columnHeaders = ["Attribute",
                    "Total Number Of Items",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "median",
                    "Q1",
                    "Q3",
                    "IQR"
    ]
    exit()

    import csv
    import shutil
    import datetime
    current_time = str(datetime.datetime.now()).replace(':', '-').replace('.','-')
    # make a copy of the current config file to exploration_stats

    shutil.copy('lib/datasets/explore_config.json', 'lib/datasets/exploration_stats/config_' + str(current_time) +'.json')

    output_file = "lib/datasets/exploration_stats/stats_" + str(current_time) + '.csv'

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columnHeaders)
        for j, h_index_list  in enumerate(stats_list):
            mean = np.mean(np.array(h_index_list))
            print(headers[j], "-index Statistics:")
            print("total number of objects: ", len(h_index_list))
            print("mean: ", mean)
            std = np.std(np.array(h_index_list))
            print("std: ", std)
            min_v = np.min(np.array(h_index_list))
            print("min: ", min_v)
            max_v = np.max(np.array(h_index_list))
            print("max: ", max_v)
            median_v = np.median(np.array(h_index_list))
            print("median: ", median_v)
            # First quartile (Q1) 
            Q1 = np.percentile(np.array(h_index_list), 25, interpolation = 'midpoint') 
            print("Q1: ", Q1)
            # Third quartile (Q3) 
            Q3 = np.percentile(np.array(h_index_list), 75, interpolation = 'midpoint') 
            print("Q3: ", Q3)
            # Interquaritle range (IQR) 
            IQR = Q3 - Q1 
            print("IQR: ", IQR) 
            row_entry = [headers[j], len(h_index_list),  mean, std, min_v, max_v, median_v, Q1, Q3, IQR]
            
            writer.writerow(row_entry)
            print()
