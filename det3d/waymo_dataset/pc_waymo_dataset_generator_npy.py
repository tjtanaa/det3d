"""Tool to convert Waymo Open Dataset to tf.Examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT_DIR = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(ROOT_DIR)

import multiprocessing
from absl import app
from absl import flags
import glob

import time
import tensorflow.compat.v2 as tf
import det3d.waymo_dataset.utils.waymo_decoder_npy as waymo_decoder_custom
from waymo_open_dataset import dataset_pb2
import numpy as np

tf.enable_v2_behavior()

flags.DEFINE_string('input_file_pattern', None, 'Path to read input')
flags.DEFINE_string('output_filebase', None, 'Path to write output')
flags.DEFINE_string('split', 'train', 'train or val or test')
flags.DEFINE_integer('num_workers', 1, 'Your age in years.', lower_bound=1)

FLAGS = flags.FLAGS


def main(unused_argv):
  # gpus = tf.config.experimental.list_physical_devices('GPU')
  # for gpu in gpus:
  #   tf.config.experimental.set_memory_growth(gpu, True)
  
  FLAGS.output_filebase = os.path.join(FLAGS.output_filebase, FLAGS.split + "_" )
  assert FLAGS.input_file_pattern
  assert FLAGS.output_filebase

  def process(processes_id, starting_idx, list_of_files):
    from tqdm import tqdm
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #   tf.config.experimental.set_memory_growth(gpu, True)
    starting_idx = starting_idx
    pbar = tqdm(total=len(list_of_files)+1)
    total_num_frames = 0
    for idx, fname in enumerate(list_of_files):
      t1 = time.time()
      dataset = tf.data.TFRecordDataset(fname, compression_type='')
      num_frames = 0
      # print(('processes_id_' + str(processes_id)))
      # with tf.device('/device:GPU:0' ):
      with tf.device('/device:CPU:0' ):
        output_scene_name_list = []
        output_frame_name_list = []
        output_lidars_list = []
        output_labels_list = []
        output_raw_object_id_list = []

        for data in dataset:
          num_frames += 1
          total_num_frames+= 1
          start = time.time()
          frame = dataset_pb2.Frame()
          frame.ParseFromString(bytearray(data.numpy()))
          decoded_frame = waymo_decoder_custom.decode_frame(frame)
          # print(decoded_frame.keys())
          # writer.write(decoded_frame)
          # print("decoded_frame['lidars'].shape: ", decoded_frame['lidars'].shape)
          output_scene_name_list.append(decoded_frame['scene_name'])
          output_frame_name_list.append(decoded_frame['frame_name'])
          output_lidars_list.append(decoded_frame['lidars'])
          output_labels_list.append(decoded_frame['output_labels'])
          output_raw_object_id_list.append(decoded_frame['output_raw_object_id_list'])
          if num_frames == 10:
            break

        print("Saving...")
        task = str(starting_idx + idx ) # sequence idx
        np.save(FLAGS.output_filebase + 'scene_name_{}.npy'.format(task), np.array(output_scene_name_list, dtype=object))
        np.save(FLAGS.output_filebase + 'frame_name_{}.npy'.format(task), np.array(output_frame_name_list, dtype=object))
        np.save(FLAGS.output_filebase + 'lidars_{}.npy'.format(task), np.array(output_lidars_list, dtype=object))
        np.save(FLAGS.output_filebase + 'output_labels_{}.npy'.format(task), np.array(output_labels_list, dtype=object))
        np.save(FLAGS.output_filebase + 'output_raw_object_id_list_{}.npy'.format(task), np.array(output_raw_object_id_list, dtype=object))

        
        end = time.time()
        pbar.update(1)
        print("CPU: ", str(processes_id), "idx: ", str(starting_idx + idx), " function time taken (s): ", end-start)
        # exit()
      t2 = time.time()
      print('processes_id:',  str(processes_id), ' idx:', str(starting_idx + idx), ' time:', t2 - t1, 'filename:', fname,'\t total num frames: ', num_frames)
    print("total_num_frames: ", total_num_frames)
    pbar.close()
    
  threads_list = []
  processes = [i for i in range(FLAGS.num_workers)]
  # processes = [i for i in range(1)]
  input_file_name = list(glob.glob(FLAGS.input_file_pattern))
  print("There are total ", len(input_file_name), " tfrecords")
  print("Total ", len(processes), " processes")
  file_indices = []
  num_files_per_processes = len(input_file_name) // len(processes)
  for i in range(len(processes)):
    file_indices.append(i * num_files_per_processes)
  file_indices.append(len(input_file_name))
  print("file_indices: ", file_indices)
  try:
    for i in range(len(processes)):
      file_for_each_processes = input_file_name[file_indices[i]:file_indices[i+1]]
      # print("file_for_gpu ", str(i), " : ", file_for_each_gpu)
      thread = multiprocessing.Process(target=process, args=(i,file_indices[i], file_for_each_processes))
      thread.daemon = True
      threads_list.append(thread)
      thread.start()
    while True:
      time.sleep(0.05)
  except:
      pass
  for i, thread in enumerate(threads_list):
      thread.terminate()
      thread.join()

      
if __name__ == '__main__':
  app.run(main)
