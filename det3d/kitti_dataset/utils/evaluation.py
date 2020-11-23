import os
import numpy as np
from det3d.kitti_dataset.utils import kitti_utils


def save_kitti_format_for_evaluation(save_sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape, classes, labels_obj):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    # kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)

    kitti_output_dir_prediction =   os.path.join(kitti_output_dir, 'Predictions')
    kitti_output_dir_labels =  os.path.join(kitti_output_dir, 'Labels')

    if not os.path.exists(kitti_output_dir_prediction):
        os.makedirs(kitti_output_dir_prediction)

    if not os.path.exists(kitti_output_dir_labels):
        os.makedirs(kitti_output_dir_labels)

    # print(kitti_output_dir_labels)
    # print(kitti_output_dir_prediction)

    kitti_output_file = os.path.join(kitti_output_dir_prediction, '%06d.txt' % save_sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (classes[k], alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)

    kitti_label_output_file = os.path.join(kitti_output_dir_labels, '%06d.txt' % save_sample_id)
    with open(kitti_label_output_file, 'w') as f:

        for obj in labels_obj:
            print(obj.src.rstrip('\n'), file=f)

            # print(obj.src)
        # for k in range(bbox3d.shape[0]):
        #     if box_valid_mask[k] == 0:
        #         continue
        #     x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
        #     beta = np.arctan2(z, x)
        #     alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        #     print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
        #           (classes[k], alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
        #            bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
        #            bbox3d[k, 6], scores[k]), file=f)




def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape, classes):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    # kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (classes[k], alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)


