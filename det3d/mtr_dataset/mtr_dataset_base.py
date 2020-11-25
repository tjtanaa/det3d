import os
import sys
import json
import numpy as np 

from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List

from det3d.mtr_dataset.config import config
from det3d.mtr_dataset.utils.fileio import load_directory_list_from_path, load_filenames_from_directory, load_absolute_directory_list_from_path
from det3d.mtr_dataset.utils.fileio import load_annotations_from_file_in_mtr_format, convert_mtr_to_kittimot_format, convert_kittimot_to_ab3dmot_format
from det3d.mtr_dataset.point_cloud_utils import rotation_matrix, transform, denoising_point_cloud_index
from det3d.mtr_dataset.utils import mtr_utils

class MTRDatasetBase(object):
    """
       This is the Base Cases that provide methods to
       1. load point cloud (with background removed)
       2. labels
       3. invalid_region_mask

       
       Note:
       The coordinate system of the point cloud and labels are the same. (Z-up)
       Angle around Z axis

        Description of the labels:
        
                    (z) height
                    ^
                    |
        length      |
        (x) <-------|
                   /
                  /
                 / 
                L
               (y) width


        


    Args:
        object ([type]): [description]
    """

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
        assert split =='train' or split =='test' or split =='real'
        if split == 'train' or split == 'test' or split =='real':
            split_dir = os.path.join(root_dir, split + '.txt')
            if(not os.path.exists(split_dir)):
                self.generate_training_meta_data(root_dir)
            self.sample_list = [x.strip() for x in open(split_dir).readlines()]
        # elif split == 'real':
        #     self.sample_list = 
        
        self.num_sample = self.sample_list.__len__()

    def get_point_cloud_shape(self) -> List[int]:
        return self._point_cloud_shape

    def get_point_cloud_attributes(self) -> str:
        return self._config['point_cloud']['attributes']


    # to be depreciated
    def get_sequences_list(self):
        return self.sample_list

    def get_sample_list(self):
        return self.sample_list

    # to be depreciated
    def get_number_of_sequences(self) -> int:
        return len(self.sample_list)


    # to be depreciated
    def get_current_sequence_number(self):
        return self._directory_index


    # to be depreciated
    def get_current_directory_index(self):
        return self._directory_index


    # to be depreciated
    def increment_directory_index(self):
        self._directory_index +=1
    
    # to be depreciated
    def decrement_directory_index(self):
        self._directory_index -=1
    
    def __len__(self):
        return self.num_sample
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

    def _get_invalid_region_mask(self, point_cloud_np: Any):
        xyz = point_cloud_np[:,:3]
        
        # crop the region that is NOT interested
        invalid_region_mask1 = (xyz[:,0] > -6.949255957496236) & (xyz[:,0] < 3.251014678502448)
        invalid_region_mask1 &= (xyz[:,1] > 7.899438643537479) & (xyz[:,1] < 11.001353179972943)

        # crop the region that is NOT interested
        invalid_region_mask2 = (xyz[:,0] > -13.027740395930584) & (xyz[:,0] < -9.21177287224803)
        invalid_region_mask2 &= (xyz[:,1] > 7.428958051420843) & (xyz[:,1] < 10.554803788903929)


        # crop the region that IS interested
        valid_region_mask3 = (xyz[:,1] > -5) & (xyz[:,1] < 1)
        valid_region_mask3 &= (xyz[:,0] > -10) & (xyz[:,0] < -0.6)
        # print(np.sum(valid_region_mask3))

        # crop the region that IS interested
        valid_region_mask4 = (xyz[:,1] > -5) & (xyz[:,1] < 7.5)
        valid_region_mask4 &= (xyz[:,0] > -0.4) & (xyz[:,0] < 4.4)

        # crop the region that IS interested
        valid_region_mask5 = (xyz[:,1] > -5) & (xyz[:,1] < 2.57)
        valid_region_mask5 &= (xyz[:,0] > 4.3) & (xyz[:,0] < 10.8)

        total_invalid_mask = invalid_region_mask1 | invalid_region_mask2 | ~(valid_region_mask3 | valid_region_mask4 | valid_region_mask5)

        return total_invalid_mask

    def _get_denoising_point_cloud_index(self, point_cloud:Any, nb_neighbors=10, std_ratio=0.4):
        return denoising_point_cloud_index(point_cloud, nb_neighbors, std_ratio)

    def _get_remove_background_using_statistics_mask(self, point_cloud_np):

        with open(os.path.join(self._point_cloud_statistics_path, 'min_range.npy'), 'rb') as f:
                min_array = np.load(f)
                # print(min_array[:10])
                
        with open(os.path.join(self._point_cloud_statistics_path, 'max_range.npy'), 'rb') as f:
            max_array = np.load(f)
            # print(max_array[:10])

        invalid_point_mask = np.greater(point_cloud_np[:,4], min_array) & np.less(point_cloud_np[:,4], max_array)
        
        return invalid_point_mask

    def _normalize_features(self, features):
        # Range [32 bit unsigned int - only 20 bits used] - 1048575

        # Signal Photons [16 bit unsigned int] - 65536

        # Reflectivity [16 bit unsigned int] - 65536

        # Ambient Photons [16 bit unsigned int] - 65536

        features /= np.array([1048575, 65536-1, 65536-1, 65536-1])

        return features

    def _load_single_point_cloud_without_background(self, filepath: str):
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


            point_cloud_np = np.concatenate([ xyz.astype('<f4'), features.astype('<f4')], axis=1).astype('<f4')

            invalid_point_mask = self._get_remove_background_using_statistics_mask(point_cloud_np)
            
            invalid_region_mask = self._get_invalid_region_mask(point_cloud_np)

            total_mask = invalid_point_mask | invalid_region_mask
            
            filtered_point_cloud_np = point_cloud_np[~total_mask, :]

            # get the mask of those that are located at (x,y,z) = (0,0,z) (in bird eye view)
            mask = np.all(np.abs(filtered_point_cloud_np[:,:2]) < 0.1, axis=1)

            # normalize feature

            filtered_point_cloud_np[:,3:] = self._normalize_features(filtered_point_cloud_np[:,3:])

            return filtered_point_cloud_np[~mask,:]
        else:
            raise FileNotFoundError

    
    def get_lidar_without_background(self, idx):
        lidar_file = os.path.join(self._data_path, self.sample_list[idx])
        assert os.path.exists(lidar_file)
        return self._load_single_point_cloud_without_background(lidar_file)



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


            filtered_point_cloud_np = np.concatenate([ xyz.astype('<f4'), features.astype('<f4')], axis=1).astype('<f4')

            # invalid_point_mask = self._get_remove_background_using_statistics_mask(point_cloud_np)
            
            # invalid_region_mask = self._get_invalid_region_mask(point_cloud_np)

            # total_mask = invalid_point_mask | invalid_region_mask
            
            # filtered_point_cloud_np = point_cloud_np[~total_mask, :]

            # get the mask of those that are located at (x,y,z) = (0,0,z) (in bird eye view)
            mask = np.all(np.abs(filtered_point_cloud_np[:,:2]) < 0.1, axis=1)

            # normalize feature

            filtered_point_cloud_np[:,3:] = self._normalize_features(filtered_point_cloud_np[:,3:])

            return filtered_point_cloud_np[~mask,:]
        else:
            raise FileNotFoundError


    def get_lidar(self, idx):
        """
        Load raw point cloud data (without filtering)

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        lidar_file = os.path.join(self._data_path, self.sample_list[idx])
        assert os.path.exists(lidar_file)
        return self._load_single_point_cloud(lidar_file)


    # to be depreciated
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

    # to be depreciated
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
        
        data_filename_list = []
        annotation_filename_list = []
        for current_directory in self._directory_list:
        # current_directory = self._directory_list[self._directory_index]
        # print("current directory\t: ", current_directory)
            current_data_path = os.path.join(self._data_path, current_directory)
            current_annotation_path = os.path.join(self._annotation_path, current_directory)

            data_filename_list.extend(load_filenames_from_directory(current_data_path, extension='.bin'))
            annotation_filename_list.extend(load_filenames_from_directory(current_annotation_path, extension='.json'))

        print("=============== Generating Training-Test Meta Data ==================")
        print("Data availables: ", len(data_filename_list))
        print("Label availables: ", len(annotation_filename_list))
        print("Labeled Data Percentage: ", len(annotation_filename_list)/ len(data_filename_list) * 100, "%")
        
        train_start, train_end = 0, int(self._config['train_test_split'] * len(annotation_filename_list))
        test_start, test_end = train_end, len(annotation_filename_list)

        # generate a text file containing the name of files being used as train and test

        train_txt_filepath = os.path.join(save_path, 'train.txt')
        with open(train_txt_filepath, 'w') as f:
            f.writelines("%s\n" % '/'.join(filename.split('/')[-2:])[:-5] for filename in annotation_filename_list[train_start: train_end])

        test_txt_filepath = os.path.join(save_path, 'test.txt')
        with open(test_txt_filepath, 'w') as f:
            f.writelines("%s\n" % '/'.join(filename.split('/')[-2:])[:-5] for filename in annotation_filename_list[test_start: test_end])

        
        real_txt_filepath = os.path.join(save_path, 'real.txt')
        with open(real_txt_filepath, 'w') as f:
            f.writelines("%s\n" % '/'.join(filename.split('/')[-2:]) for filename in data_filename_list[test_end:])

    # ========== The following codes are use to load no label data ===========
    # The split setting is called 'real'



