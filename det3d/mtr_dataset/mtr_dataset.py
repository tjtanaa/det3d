import os
import sys
import json
import numpy as np 

from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List

from det3d.point_cloud_utils.transformation import rotation_matrix, transform

from det3d.mtr_dataset.config import config
from det3d.mtr_dataset.utils import load_directory_list_from_path, load_filenames_from_directory, load_absolute_directory_list_from_path
from det3d.mtr_dataset.utils import load_annotations_from_file_in_mtr_format, convert_mtr_to_kittimot_format, convert_kittimot_to_ab3dmot_format
from det3d.mtr_dataset.mtr_dataset_base import MTRDatasetBase

class MTRDataset(MTRDatasetBase):
    def __init__(self, root_dir):
        super().__init__(root_dir)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class MTRDatasetRandom(MTRDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
