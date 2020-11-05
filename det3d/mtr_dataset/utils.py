import os
import sys
import json
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List

from det3d.mtr_dataset.config import config

def id2str(id: int) -> str:
    """ This is a function that maps the id to the string. 
        Its definition is declared in the config.py

    Args:
        id (int): The id of the object type

    Returns:
        str: The string of the object type
    """
    return config['class_map'][id]

def str2id(classname: str) -> int:
    """This is a function that maps the string of the object type to its id.
       The mapping is defined in the config.py

    Args:
        classname (str): The string of the object type  

    Returns:
        int: The id of the object type
    """
    str2id_map = {v: k for k, v in config['class_map'].items()}
    return str2id_map[classname]

def sort_list(directory_list: List[str], charbefore:int = 20) -> List[str]:
    """This is a custom sort function that is used to sort the filename
        of the AKK dataset, which has a variable filename.

    Args:
        directory_list (list[str]): [description]
        charbefore (int, optional): [description]. Defaults to 20.
    """
    def func(x):
        return x[:charbefore]+x[charbefore:][:-4].zfill(4)
    return sorted(directory_list,key=func)


def load_absolute_directory_list_from_path(path: str, suffix: str ='_dir') -> List[str]:
    """This is a function to aggregate the directories with suffix in the 'path'
       into a list, in absolute path format.
       
       [Note]
       The difference between this function `load_directory_list_from_path`
       and `load_filenames_from_path` is that
       `load_directory_list_from_path` only aggregates the path to the child directory
       `load_filenames_from_path` aggregates the path to all the files containing in the
        child directory

    Args:
        path (str): Absolute path of the directory of interest
        suffix (str, optional): The directory with suffix is aggregated, else discarded. 
                                Defaults to '_dir'.

    Raises:
        FileNotFoundError: The `path` does not exists

    Returns:
        List[str]: A list of absolute path to the directories with suffix.
    """
    directory_list = []
    if(os.path.exists(path)):
        directory_list = [os.path.join(path, directory) for directory in os.listdir(path)
                                if (os.path.isdir(os.path.join(path, directory)) and suffix in directory) ]
    else:
        raise FileNotFoundError
    
    return directory_list

def load_directory_list_from_path(path: str, suffix: str ='_dir') -> List[str]:
    """This is a function to aggregate the directories with suffix in the 'path'
       into a list, in absolute path format.
       
       [Note]
       The difference between this function `load_directory_list_from_path`
       and `load_filenames_from_path` is that
       `load_directory_list_from_path` only aggregates the path to the child directory
       `load_filenames_from_path` aggregates the path to all the files containing in the
        child directory

    Args:
        path (str): Absolute path of the directory of interest
        suffix (str, optional): The directory with suffix is aggregated, else discarded. 
                                Defaults to '_dir'.

    Raises:
        FileNotFoundError: The `path` does not exists

    Returns:
        List[str]: A list of absolute path to the directories with suffix.
    """
    directory_list = []
    if(os.path.exists(path)):
        directory_list = [directory for directory in os.listdir(path)
                                if (os.path.isdir(os.path.join(path, directory)) and suffix in directory) ]
    else:
        raise FileNotFoundError
    
    return directory_list

def load_filenames_from_path(path: str, extension: str ='.bin') -> List[str]:
    """This is a function to aggregate the filenames with suffix in the child 
        directory of 'path' into a list, in absolute path format.

        [Note]
        The difference between this function `load_directory_list_from_path`
        and `load_filenames_from_path` is that
        `load_directory_list_from_path` only aggregates the path to the child directory
        `load_filenames_from_path` aggregates the path to all the files containing in the
        child directory

    Args:
        path (str): Absolute path of the directory of interest
        extension (str, optional): The filenames with suffix is aggregated, else discarded. 
                                Defaults to '.bin'.

    Raises:
        FileNotFoundError: The `path` does not exists

    Returns:
        List[str]: A list of absolute path to the filenames 
                    (within the child directories of `path`) with suffix.
    """
    sorted_filenames_list = []
    if(os.path.exists(path)):
        directory_list = load_directory_list_from_path(path)
        
        for directory in directory_list:
            filename_list = [filename for filename in os.listdir(os.path.join(path, directory))
                                        if (os.path.isfile(
                                                os.path.join(path, 
                                                os.path.join(directory, filename)
                                                )) and extension in filename)  ]
            
            filename_list = sort_list(filename_list)

            sorted_filenames_list += [os.path.join(path, os.path.join(directory, filename)) for filename in filename_list]
    else:
        raise FileNotFoundError
    
    return sorted_filenames_list

def load_data_filenames_from_path(path: str) -> List[str]:
    """ This is a function to load a list of absolute path to the filenames 
        (within the child directories of `path`) with suffix `.bin`.

        [Note]
        `data_filenames` refers to the binary data storing the point cloud
        ALL the input point cloud are store in binary

    Args:
        path (str): Absolute path to the directory which stores the binary of
                    point cloud. Expected path : `<path to the database>/Data/`

    Returns:
        List[str]: A list of absolute path to the filenames 
                    (within the child directories of `path`) with suffix `.bin`.
    """
    return load_filenames_from_path(path, extension='.bin')

def load_annotation_filenames_from_path(path: str) -> List[str]:
    """This is a function to load a list of absolute path to the filenames 
        (within the child directories of `path`) with suffix `.json`.

        [Note]
        `annotation` refers to the ground truth label of bounding boxes

    Args:
        path (str): Absolute path to the directory which stores the binary of
                    point cloud. Expected path : `<path to the database>/Label/`

    Returns:
        List[str]: A list of absolute path to the filenames 
                    (within the child directories of `path`) with suffix `.json`.
    """
    return load_filenames_from_path(path, extension='.json')

def load_filenames_from_directory(directory_path: str, extension: str='.bin') -> List[str]:
    """This is a function that takes in an absolute path to directories
       and aggregates the files with the extension (e.g. `/bin`) from the directory
       into a list , in absolute path format

    Args:
        directory_path ([type]): [description]
        extension (str, optional): [description]. Defaults to '.bin'.

    Raises:
        FileNotFoundError: The directory path does not exists

    Returns:
        List[str]: A list of absolute path to the filenames 
                    (within the child directories of `path`) with suffix `.bin`.
    """
    sorted_filenames_list = []
    if(os.path.exists(directory_path)):
        filename_list = [filename for filename in os.listdir(directory_path)
                                    if (os.path.isfile(
                                            os.path.join(directory_path, filename)
                                            ) and extension in filename)  ]
        
        filename_list = sort_list(filename_list)

        sorted_filenames_list += [os.path.join(directory_path, filename) for filename in filename_list]
    else:
        raise FileNotFoundError
    
    return sorted_filenames_list


    """
    """
def load_annotations_from_file_in_mtr_format(filepath: str) -> List[Union[str, int, float]]:
    """ This is a function to load the annotations

        [Note]

        MTR data format
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    Args:
        filepath (str): The absolute path to the annotations


    Returns:
        List[Union[str, int, float]]: List of annotations with the attribute show in the Note
    """
    with open(filepath, 'r') as f:
        json_obj = json.load(f)
        # print(json_obj)
        bounding_boxes = json_obj['bounding_boxes']
        
        # filter out noisy annotations
        # and convert the data to kitti MOTS data format
        
        # []
        annotation_list = []
        track_id = -1
        for bboxes in bounding_boxes:
            if bboxes['center']['z'] is None or bboxes['height'] is None or bboxes['height'] < 0.001 \
                or bboxes['width'] < 0.001 or bboxes['length'] < 0.001:
                continue
            # annotation = [frame_id, -1]
            annotation = []
            # print("type: ", str2id(bboxes['object_id']))
            # object_type = bboxes['object_id'] # suppress as 'pedestrian'
            object_type = 'pedestrian'
            # truncated = -1
            # occluded = -1
            # alpha = -1
            # bbox2d = [-1, -1, -1, -1]
            dimensions = [bboxes['height'], bboxes['width'], bboxes['length']]
            location = [bboxes['center']['x'], bboxes['center']['y'], bboxes['center']['z']]
            rotation_y = bboxes['angle']

            annotation.append(object_type)
            # annotation.append(truncated)
            # annotation.append(occluded)
            # annotation.append(alpha)
            # annotation += bbox2d
            annotation += dimensions
            annotation += location
            annotation.append(rotation_y)
            annotation_list.append(annotation)
        return annotation_list

    """
    """
def load_annotations_from_file_in_kittimot_format(filepath: str, frame_id: int) -> List[Union[str, int, float]]:
    """ This is the function that directly loads the MTR annotations into kitti mot format

        [Note]
        From https://github.com/pratikac/kitti/blob/master/readme.tracking.txt
        kitti MOTS data format
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    frame        Frame within the sequence where the object appearers
        1    track id     Unique tracking id of this object within this sequence
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries.
                    Truncation 2 indicates an ignored object (in particular
                    in the beginning or end of a track) introduced by manual
                    labeling.
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
    Args:
        filepath (str): The absolute path to the annotations
        frame_id (int): The assigned frame number to the annotation files loaded

    Returns:
        List[Union[str, int, float]]: List of annotations with the attribute show in the Note
    """
    with open(filepath, 'r') as f:
        json_obj = json.load(f)
        # print(json_obj)
        bounding_boxes = json_obj['bounding_boxes']
        
        # filter out noisy annotations
        # and convert the data to kitti MOTS data format
        
        # []
        annotation_list = []
        track_id = -1
        for bboxes in bounding_boxes:
            if bboxes['center']['z'] is None or bboxes['height'] is None or bboxes['height'] < 0.001 \
                or bboxes['width'] < 0.001 or bboxes['length'] < 0.001:
                continue
            annotation = [frame_id, -1]
            # print("type: ", str2id(bboxes['object_id']))
            # object_type = bboxes['object_id'] # suppress as 'pedestrian'
            object_type = 'pedestrian'
            truncated = -1
            occluded = -1
            alpha = -1
            bbox2d = [-1, -1, -1, -1]
            dimensions = [bboxes['height'], bboxes['width'], bboxes['length']]
            location = [bboxes['center']['x'], bboxes['center']['y'], bboxes['center']['z']]
            rotation_y = bboxes['angle']

            annotation.append(object_type)
            annotation.append(truncated)
            annotation.append(occluded)
            annotation.append(alpha)
            annotation += bbox2d
            annotation += dimensions
            annotation += location
            annotation.append(rotation_y)
            annotation_list.append(annotation)
        return annotation_list



    """
        filepath is the absolute path to the annotations

    """

def convert_mtr_to_kittimot_format(data_list: List[Union[str, int, float]], frame_id: int) -> List[Union[str, int, float]]:
    """ This is the function that converts the MTR annotations into kitti mot format

        [Note]
        From https://github.com/pratikac/kitti/blob/master/readme.tracking.txt
        kitti MOTS data format
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    frame        Frame within the sequence where the object appearers
        1    track id     Unique tracking id of this object within this sequence
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries.
                    Truncation 2 indicates an ignored object (in particular
                    in the beginning or end of a track) introduced by manual
                    labeling.
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
    Args:
        data_list (List[Union[str, int, float]]): List of annotations with the attribute show in 
                                                the Note of function `load_annotations_from_file_in_mtr_format`
        frame_id (int): The assigned frame number to the annotation files loaded

    Returns:
        List[Union[str, int, float]]: List of annotations with the attribute show in the Note
    """
    annotation_list = []
    track_id = -1
    for data in data_list:
        annotation = [frame_id, -1]
        # print("type: ", str2id(bboxes['object_id']))
        object_type = data[0]
        truncated = -1
        occluded = -1
        alpha = -1
        bbox2d = [-1, -1, -1, -1]
        dimensions = data[1:4]
        location = data[4:7]
        rotation_y = data[7]

        annotation.append(object_type)
        annotation.append(truncated)
        annotation.append(occluded)
        annotation.append(alpha)
        annotation += bbox2d
        annotation += dimensions
        annotation += location
        annotation.append(rotation_y)
        annotation_list.append(annotation)
    return annotation_list



    """
        convert KITTI MOTS format to AB3DMOT format

        
        @params:
        data_list: a list containing data in KITTI MOTs format
    """

def convert_kittimot_to_ab3dmot_format(data_list: List[Union[str, int, float]]) -> List[Union[str, int, float]]:
    """This is the function that converts the kitti mot format annotations into ab3dmot format

        [Note]
        From https://github.com/xinshuoweng/AB3DMOT.git
        AB3DMOT format
        =============================================================================================
        Frame	Type	2D BBOX (x1, y1, x2, y2)	Score	3D BBOX (h, w, l, x, y, z, rot_y)	Alpha
        0	2 (car)	726.4, 173.69, 917.5, 315.1	13.85	1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56	-1.82

    Args:
        data_list (List[Union[str, int, float]]): List of annotations with the attribute show in the Note
                                                of function `convert_mtr_to_kittimot_format`

    Returns:
        List[Union[str, int, float]]: List of annotations with the attribute show in the Note
    """
    ab3dmot_data_list = []

    for data in data_list:
        annotation = []
        annotation.append(data[0])
        annotation.append(str2id(data[2]))
        annotation.append(80) # max scores as it is human annotated
        annotation += data[6:17]
        annotation.append(data[5])
        ab3dmot_data_list.append(annotation)

    return ab3dmot_data_list


if __name__ == "__main__":
    datasource_path = "/home/tjtanaa/Documents/AKK/Project4-MTR"
    datapath = os.path.join(datasource_path, 'Data')
    labelpath = os.path.join(datasource_path, 'Label')

    print("========================== Data Loading =======================")
    print("data path \t: ", datapath)
    print("annotation path \t: ", labelpath)

    data_filename_list = load_data_filenames_from_path(datapath)
    annotation_filename_list = load_annotation_filenames_from_path(labelpath)
    print("Total number of data filenames \t: ", len(data_filename_list))
    print("Total number of anno filneames \t: ", len(annotation_filename_list))


    print("================= MTR format to Kitti MOTS format =============")
    data_sample_mtr_format = load_annotations_from_file_in_mtr_format(annotation_filename_list[100])
    data_sample_kittimot_format = np.array(convert_mtr_to_kittimot_format(data_sample_mtr_format, 100))

    print("Before converting MTR format to Kitti format:")
    # print(data_sample_mtr_format)
    print("After convertion:")
    print(data_sample_kittimot_format.shape)

    print("================= Kitti MOTS format to AB3DMOT format ==========")
    data_sample_kittimot_format = load_annotations_from_file_in_kittimot_format(annotation_filename_list[100], 100)
    data_sample_ab3dmot_format = np.array(convert_kittimot_to_ab3dmot_format(data_sample_kittimot_format))

    print("Before converting Kitti format to AB3DMOT format:")
    # print(data_sample_kittimot_format)
    print("After convertion:")
    print(data_sample_ab3dmot_format.shape)


