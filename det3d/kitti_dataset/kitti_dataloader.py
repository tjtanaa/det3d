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

from utils.helper_tool import DataProcessing as DP
from config.config import ConfigKitti as cfg
import dataloader.kitti_dataset.utils.calibration as calibration
import dataloader.kitti_dataset.utils.kitti_utils as kitti_utils
import tensorflow as tf
from dataloader.kitti_dataset.augmentation import rotate, scale, transform, flip
from numpy.linalg import multi_dot



class KittiLoader(object):
    
    def __init__(self):
        # with open('./lib/datasets/dataloader_config.json', 'r') as f:
        #     json_obj = json.load(f)
        # self.root = json_obj['TRAINING_DATA_PATH']
        # self.num_pts = json_obj['NUM_POINTS']
        # self.pc_path = os.path.join(self.root, 'point_cloud')
        # self.label_path = os.path.join(self.root, 'labels')
        # self.frames = [s.split('.')[0] for s in os.listdir(self.pc_path) if '.bin' in s ]
        self.name = 'KITTI'
        self.root = cfg.train_data_path
        self.num_pts = cfg.num_points
        self.pc_path = os.path.join(self.root, 'point_cloud')
        self.label_path = os.path.join(self.root, 'labels')
        self.frames = [s.split('.')[0] for s in os.listdir(self.pc_path) if '.bin' in s ]
        self.num_classes = cfg.num_classes
        if self.num_classes ==1:
            self.num_classes = 0
        self.num_features = cfg.num_features
        self.num_target_attributes = cfg.num_target_attributes
        self.split_ratio = cfg.split_ratio
        self.augmentation = cfg.augmentation
        
        self.rotate_range=np.pi/4
        self.rotate_mode='u'
        self.scale_range=0.05
        self.scale_mode='u'
        self.flip_flag=True
        self.num_samples = len(self.frames)
        # self.num_samples = 500
        assert np.abs(np.sum(self.split_ratio) - 1.0) < 1e-5
        train_split = int(self.num_samples * self.split_ratio[0])
        val_split = int(self.num_samples * np.sum(self.split_ratio[:2]))

        self.frames_indices = np.arange(len(self.frames))
        # self.train_list = self.frames[:train_split]
        # self.val_list = self.frames[train_split:val_split]
        # self.test_list = self.frames[val_split:]
        self.train_list = self.frames_indices[:train_split]
        self.val_list = self.frames_indices[train_split:]
        self.test_list = self.frames_indices[train_split:]

        self.train_list = DP.shuffle_list(self.train_list)
        self.val_list = DP.shuffle_list(self.val_list)
        self.num_points_left = int(cfg.num_points * np.prod(cfg.sub_sampling_ratio))


    def resampling_pc_indices(self, pc):
        # resampling the point cloud to NUM_POINTS
        indices = np.arange(pc.shape[0])
        if self.num_pts > pc.shape[0]:
            num_pad_pts = self.num_pts - pc.shape[0]
            pad_indices = np.random.choice(indices, size=num_pad_pts)
            indices = np.hstack([indices, pad_indices])
            np.random.shuffle(indices)
        else:
            np.random.shuffle(indices)
            indices = indices[:self.num_pts]

        return indices


    def get_sample(self, idx):
        _idx = idx
        while(True):
            with open(os.path.join(self.pc_path, self.frames[_idx] + '.bin'), 'rb') as f:
                pc = np.fromfile(f, dtype=np.float32)
                pc = pc.reshape(pc.shape[0]//4, 4)
            
            # with open(os.path.join(self.label_path, self.frames[idx] + '.pkl'), 'rb') as f:
            #     label = pickle.load(f)

            with open(os.path.join(self.label_path, self.frames[_idx] + '.bin'), 'rb') as f:
                
                target = np.fromfile(f, dtype=np.float32)
                target = target.reshape(target.shape[0]//(self.num_target_attributes)
                                , self.num_target_attributes)
            

            # start to precompute the randlanet input

            # Random resampling the points to NUM_POINTS self.num_pts
            indices = self.resampling_pc_indices(pc)

            resampled_pc = pc[indices,:3]
            resampled_features = pc[indices,:4]
            resampled_bboxes = target[indices[:self.num_points_left], :self.num_target_attributes - 2] # [x,y,z,h,w,l,ry]
            resampled_fgbg = target[indices[:self.num_points_left], self.num_target_attributes-2] #.reshape(-1,1) # [fgbg]
            resampled_cls = ((np.logical_or(target[indices[:self.num_points_left], self.num_target_attributes-1] == 1, target[indices[:self.num_points_left], self.num_target_attributes-1] == 4))).astype(np.float32) #.reshape(-1,1) # [cls]
            # resampled_cls_one_hot = None
            # if self.num_classes > 1:
            #     # print("num class > 1")
            #     resampled_cls_one_hot = target[indices, self.num_target_attributes:self.num_target_attributes + self.num_classes]
            
            # print(resampled_pc.shape)
            # print(resampled_features.shape)
            # print(resampled_target.shape)
            # print(resampled_fgbg.shape)
            # print(resampled_cls_one_hot.shape)
            # if np.sum(resampled_fgbg[:self.num_points_left]) > 0:
            if np.sum(resampled_cls[:self.num_points_left]) > 0:
            # return resampled_pc, resampled_features, resampled_target, resampled_fgbg, resampled_cls_one_hot
                if self.augmentation:
                    T_rotate, angle = rotate(self.rotate_range, self.rotate_mode)
                    T_scale, scale_xyz = scale(self.scale_range, self.scale_mode)
                    T_flip, flip_y = flip(flip=self.flip_flag)
                    T_coors = multi_dot([T_scale, T_flip, T_rotate])
                    resampled_pc = transform(resampled_pc, T_coors)
                    resampled_bboxes_xyz = resampled_bboxes[:,:3]
                    # h w l
                    resampled_bboxes_hwl = resampled_bboxes[:,3:-1]
                    resampled_bboxes_ry = resampled_bboxes[:,-1:]
                    # resampled_bboxes_ry
                    # w, l, h, x, y, z, r = box[:7]
                    resampled_bboxes_xyz = transform(resampled_bboxes_xyz, T_coors)
                    resampled_bboxes_hwl = transform(resampled_bboxes_hwl, T_scale)
                    resampled_bboxes_w = resampled_bboxes_hwl[:,1,np.newaxis]
                    resampled_bboxes_l = resampled_bboxes_hwl[:,2,np.newaxis]
                    resampled_bboxes_h = resampled_bboxes_hwl[:,0,np.newaxis]
                    resampled_bboxes_ry += angle
                    for i in range(self.num_points_left):
                        if flip_y == -1:
                            resampled_bboxes_ry[i] = resampled_bboxes_ry[i] + 2 * (np.pi / 2 - resampled_bboxes_ry[i])
                        if np.abs(resampled_bboxes_ry[i]) > np.pi:
                            resampled_bboxes_ry[i] = (2 * np.pi - np.abs(resampled_bboxes_ry[i])) * \
                                                    ((-1)**(resampled_bboxes_ry[i]//np.pi))
                            
                    
                    # angle_cat_1_part1 = np.logical_and(resampled_bboxes_ry >= (-7 * np.pi/6), resampled_bboxes_ry <= (np.pi/6)).astype(np.float32)
                    # angle_cat_1_part2 = np.logical_and(resampled_bboxes_ry >= (-7 * np.pi/6), resampled_bboxes_ry <= (np.pi/6)).astype(np.float32)
                    
                    # angle_cat_2 = np.logical_and(resampled_bboxes_ry >= (- np.pi/6), resampled_bboxes_ry <= (7 * np.pi/6)).astype(np.float32)

                    
                    
                    # print(angle_cat_1.shape)
                    # print(angle_cat_2.shape)
                    # print(np.sum(angle_cat_1))

                    # resampled_bboxes_xyz = transform(resampled_bboxes_xyz, T_coors)
                    # # [x, y, z, h, w, l, ry, cat_1, cat_2]
                    # resampled_bboxes = np.concatenate([resampled_bboxes_xyz, resampled_bboxes_hwl, resampled_bboxes_ry, angle_cat_1, angle_cat_2],axis=1)
                    # [w, l, h, x, y, z, ry]
                    resampled_bboxes = np.concatenate([ resampled_bboxes_w, resampled_bboxes_l, resampled_bboxes_h, 
                                        resampled_bboxes_xyz,resampled_bboxes_ry], axis=1)
                    # print("resampled_bboxes.shape: ", resampled_bboxes.shape)
                # resampled_bboxes = None
                return resampled_pc, resampled_features, resampled_bboxes[:self.num_points_left], resampled_fgbg[:self.num_points_left], resampled_cls[:self.num_points_left]
            else:
                if _idx -1 < 0:
                    _idx = _idx + 1
                else:
                    _idx = _idx - 1
        
    def get_num_per_epoch(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.train_list) / cfg.batch_size) * cfg.batch_size
            # path_list = self.train_list
        elif split == 'validation':
            num_per_epoch = int(len(self.val_list) / cfg.val_batch_size) * cfg.val_batch_size
            cfg.val_steps = int(len(self.val_list) / cfg.batch_size)
            # path_list = self.val_list
        elif split == 'test':
            num_per_epoch = int(len(self.test_list) / cfg.val_batch_size) * cfg.val_batch_size * 4
            # path_list = self.test_list   
        return num_per_epoch

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.train_list) / cfg.batch_size) * cfg.batch_size
            path_list = self.train_list
        elif split == 'validation':
            num_per_epoch = int(len(self.val_list) / cfg.val_batch_size) * cfg.val_batch_size
            cfg.val_steps = int(len(self.val_list) / cfg.batch_size)
            path_list = self.val_list
        elif split == 'test':
            num_per_epoch = int(len(self.test_list) / cfg.val_batch_size) * cfg.val_batch_size * 4
            path_list = self.test_list        

        def spatially_regular_gen():
            for i in range(num_per_epoch):
                sample_id = path_list[i]
                pc, features, bboxes, fgbg, cls_label = self.get_sample(sample_id)
                yield (pc.astype(np.float32), 
                    features.astype(np.float32), 
                    bboxes.astype(np.float32), 
                    fgbg.astype(np.int32), 
                    cls_label.astype(np.int32))

        gen_func = spatially_regular_gen            
        gen_types = (tf.float32, tf.float32, tf.float32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 4], [None, 7], [None], [None])
        return gen_func, gen_types, gen_shapes



    @staticmethod
    def get_tf_mapping():

        def tf_map(batch_pc, batch_features, batch_bboxes, batch_fgbg, batch_cls):
            features = batch_features
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            # print(batch_pc.numpy().shape)
            for i in range(cfg.num_layers):
                # print("Number of layer: ", i)
                neighbour_idx = tf.py_function(DP.knn_search, [batch_pc, batch_pc, cfg.k_n], tf.int32)
                index_limit = tf.cast(tf.cast(tf.shape(batch_pc)[1], dtype=tf.float32) * cfg.sub_sampling_ratio[i], dtype = tf.int32)
                # print("index_limit ", index_limit)
                sub_points = batch_pc[:, :index_limit, :]
                # print("jks")
                pool_i = neighbour_idx[:, :index_limit, :]
                up_i = tf.py_function(DP.knn_search, [sub_points, batch_pc, 1], tf.int32)
                input_points.append(batch_pc)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_pc = sub_points


            # follow the anchor strategy from the SECOND
            # set the anchor to -1
            # batch_pc[:,:,2].assign(tf.ones(shape=tf.shape(batch_pc)[:2]) * -1.0)
            # batch_pc = tf.concat([batch_pc[:,:,:2], tf.ones(shape=(tf.shape(batch_pc)[0], tf.shape(batch_pc)[1], 1)) * -1.0],axis=-1 )

            batch_bboxes = batch_bboxes[:, :int(tf.shape(batch_pc)[1]),:]
            
            batch_fgbg = batch_fgbg[:, :int(tf.shape(batch_pc)[1])]
            batch_cls = batch_cls[:, :int(tf.shape(batch_pc)[1])]

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [features, batch_pc, batch_bboxes, batch_fgbg, batch_cls]

            # print('len(input_list): ', len(input_list))

            return input_list

        return tf_map


    def init_input_pipeline(self):
        print('Initiating input pipelines')
        # cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)



if __name__ == '__main__':

    dataset = KittiLoader()
    # dataset.init_input_pipeline()
    
    # with tf.Session() as sess:
    #     one_batch = sess.run(dataset.train_init_op)
    #     one_batch = sess.run(dataset.flat_inputs)
    #     print("len(one_batch): ", len(one_batch))
    #     for tensor in one_batch:
    #         print(tensor.shape)
    #     print("tensor")
    #     print(np.sum(one_batch[-2]))
    #     print(np.sum(one_batch[-1]))
    #     fgbg_value_set = set()
    #     for i in range(len(one_batch[-2][0])):
    #         fgbg_value_set.update([one_batch[-2][0][i]])
    #     print("fgbg_value_set: ", fgbg_value_set)
        
    #     cls_value_set = set()
    #     for j in range(len(one_batch[-1])):
    #         per_scene_cls_value_set = set()
    #         for i in range(len(one_batch[-1][0])):
    #             cls_value_set.update([one_batch[-1][j][i]])
    #             per_scene_cls_value_set.update([one_batch[-1][j][i]])
    #     print("cls_value_set: ", cls_value_set)
    #     print(tensor)

    # # print(len(dataset.frames))
    # # pc, features, target, fgbg, cls_one_hot = dataset.get_sample(0)
    # # print(pc.shape)
    # # print(pc[0])

    gen_func, gen_type, gen_shape = dataset.get_batch_gen('training')


    ds_counter = tf.data.Dataset.from_generator(gen_func, output_types=gen_type, output_shapes = gen_shape, )
    for count_batch in ds_counter.repeat().batch(10).prefetch(20):
        print(len(count_batch), ": ", count_batch[0].numpy().shape)