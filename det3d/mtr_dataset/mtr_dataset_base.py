import os
import sys
import json
import numpy as np 

from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List

from det3d.mtr_dataset.config import config
from det3d.mtr_dataset.utils.fileio import load_directory_list_from_path, load_filenames_from_directory, load_absolute_directory_list_from_path
from det3d.mtr_dataset.utils.fileio import load_annotations_from_file_in_mtr_format, convert_mtr_to_kittimot_format, convert_kittimot_to_ab3dmot_format
from det3d.mtr_dataset.point_cloud_utils import rotation_matrix, transform
from det3d.mtr_dataset.utils import mtr_utils

class MTRDatasetBase(object):

    def __init__(self, root_dir:str, 
                split: str = 'train',
                point_cloud_statistics_path:str = "det3d/mtr_dataset/point_cloud_statistics"):
        self._root_dir = root_dir
        self.split = split

        self._config = config
        self._config['dataset_path'] = root_dir
        self._config['point_cloud_statistics_path'] = point_cloud_statistics_path
        self._data_path = os.path.join(root_dir, 'Data')
        self._annotation_path = os.path.join(root_dir, 'Label')

        self._directory_list = load_directory_list_from_path(self._annotation_path)
        # print(self._directory_list)
        self._point_cloud_statistics_path = config['point_cloud_statistics_path']
        self._point_cloud_shape = [config['point_cloud']['height'], 
                                    config['point_cloud']['width'],
                                    config['point_cloud']['channels']]
        self._directory_index = 0
        self._frame_index = 0
        # check if there are training metadata, create one if there isn't
        split_dir = os.path.join(root_dir, split + '.txt')
        if(not os.path.exists(split_dir)):
            self.generate_training_meta_data(root_dir)
        self.sample_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.sample_list.__len__()

    def get_point_cloud_shape(self) -> List[int]:
        return self._point_cloud_shape

    def get_point_cloud_attributes(self) -> str:
        return self._config['point_cloud']['attributes']

    def get_sequences_list(self):
        return self.sample_list

    def get_number_of_sequences(self) -> int:
        return len(self.sample_list)

    def get_current_sequence_number(self):
        return self._directory_index

    def get_current_directory_index(self):
        return self._directory_index

    def increment_directory_index(self):
        self._directory_index +=1
    
    def decrement_directory_index(self):
        self._directory_index -=1
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def _load_single_annotation_in_mtr_format(self, filepath: str):
        if(os.path.exists(filepath)):
            sample = load_annotations_from_file_in_mtr_format(filepath)
            return sample
        else:
            raise FileNotFoundError


    def get_label(self, idx):
        label_file = os.path.join(self._annotation_path, self.sample_list[idx] + '.json')
        assert os.path.exists(label_file)
        sample = self._load_single_annotation_in_mtr_format(label_file)
        sample = convert_mtr_to_kittimot_format(sample, idx)
        return mtr_utils.get_objects_from_label(sample)

    def _load_single_point_cloud(self, filepath: str):
        if(os.path.exists(filepath)):
            # print("filepath: ", filepath)
            point_cloud_np = np.fromfile(filepath, '<f4')
            # print(point_cloud_np.shape)
            # print(len(point_cloud_np.tobytes()))
            point_cloud_np = np.reshape(point_cloud_np, (-1, self._point_cloud_shape[2]))
            xyz = point_cloud_np[:,:3]
            features = point_cloud_np[:,3:]

            T_rotate = rotation_matrix(config['point_cloud']['rxyz_offset'] )
            xyz = transform(xyz, T_rotate)
            xyz[:,0] += config['point_cloud']['xyz_offset'][0]
            xyz[:,1] += config['point_cloud']['xyz_offset'][1]
            xyz[:,2] += config['point_cloud']['xyz_offset'][2]
            with open(os.path.join(self._point_cloud_statistics_path, 'min_range.npy'), 'rb') as f:
                min_array = np.load(f)
                # print(min_array[:10])
                
            with open(os.path.join(self._point_cloud_statistics_path, 'max_range.npy'), 'rb') as f:
                max_array = np.load(f)
                # print(max_array[:10])

            invalid_point_mask = np.greater(features[:,1], min_array) & np.less(features[:,1], max_array)

            # crop the region that is not interested
            invalid_region_mask1 = (xyz[:,0] > -6.949255957496236) & (xyz[:,0] < 3.251014678502448)
            invalid_region_mask1 &= (xyz[:,1] > 7.899438643537479) & (xyz[:,1] < 11.001353179972943)

            # crop the region that is not interested
            invalid_region_mask2 = (xyz[:,0] > -13.027740395930584) & (xyz[:,0] < -9.21177287224803)
            invalid_region_mask2 &= (xyz[:,1] > 7.428958051420843) & (xyz[:,1] < 10.554803788903929)

            point_cloud_np = np.concatenate([ xyz.astype('<f4'), features.astype('<f4')], axis=1).astype('<f4')
            point_cloud_np[invalid_point_mask,:] = 0
            point_cloud_np[invalid_region_mask1,:] = 0
            point_cloud_np[invalid_region_mask2,:] = 0
            point_cloud_np = np.reshape(point_cloud_np, (-1, self._point_cloud_shape[2]))
            return point_cloud_np
        else:
            raise FileNotFoundError


    def get_lidar(self, idx):
        lidar_file = os.path.join(self._data_path, self.sample_list[idx])
        assert os.path.exists(lidar_file)
        return self._load_single_point_cloud(lidar_file)


    def load_filenames_by_directory(self):
        """
            This function will load the data by directory

            @return:
                The generator returns 
                A list of data with MTR format [object_type h w l x y z ry]
        """
        current_directory = self._directory_list[self._directory_index]
        # print("current directory\t: ", current_directory)
        current_directory_path = os.path.join(self._data_path, current_directory)
        data_filename_list = load_filenames_from_directory(current_directory_path, extension='.bin')
        return data_filename_list, self._directory_index

    def load_annotations_by_directory(self):
        """
            This function will load the data by directory

            @return:
                The generator returns 
                A list of data with MTR format [object_type h w l x y z ry]
        """
        current_directory = self._directory_list[self._directory_index]
        # print("current directory\t: ", current_directory)
        current_directory_path = os.path.join(self._annotation_path, current_directory)
        annotation_filename_list = load_filenames_from_directory(current_directory_path, extension='.json')
        # print(len(annotation_filename_list))
        # print("annotation_filename_list\t: ", annotation_filename_list)
        seq_data = []
        for filename in annotation_filename_list:
            sample = load_annotations_from_file_in_mtr_format(filename)
            sample = convert_mtr_to_kittimot_format(sample, self._frame_index)
            sample = convert_kittimot_to_ab3dmot_format(sample)
            # print(len(sample))
            seq_data += sample
            self._frame_index += 1
            # if self._frame_index == 561 + 1:
            #     print(filename)
            #     print(len(sample))
        
        # self._directory_index += 1
        
        return seq_data, annotation_filename_list, self._directory_index

    

    def generate_training_meta_data(self, save_path:str) -> None:

        current_directory = self._directory_list[self._directory_index]
        # print("current directory\t: ", current_directory)
        current_data_path = os.path.join(self._data_path, current_directory)
        current_annotation_path = os.path.join(self._annotation_path, current_directory)

        data_filename_list = load_filenames_from_directory(current_data_path, extension='.bin')
        annotation_filename_list = load_filenames_from_directory(current_annotation_path, extension='.json')
        
        train_start, train_end = 0, int(self._config['train_test_split'] * len(annotation_filename_list))
        test_start, test_end = train_end, len(annotation_filename_list)

        # generate a text file containing the name of files being used as train and test

        train_txt_filepath = os.path.join(save_path, 'train.txt')
        with open(train_txt_filepath, 'w') as f:
            f.writelines("%s\n" % '/'.join(filename.split('/')[-2:])[:-5] for filename in annotation_filename_list[train_start: train_end])

        test_txt_filepath = os.path.join(save_path, 'test.txt')
        with open(test_txt_filepath, 'w') as f:
            f.writelines("%s\n" % '/'.join(filename.split('/')[-2:])[:-5] for filename in annotation_filename_list[test_start: test_end])
    