import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandgrandparentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0,grandgrandparentdir) 
import pickle
import numpy as np
import tqdm
import det3d
import argparse

# parser = argparse.ArgumentParser(description='Analayse the kitti dataset dimensions:\
#     mode: 0 := generate the numpy file ')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()


if __name__ == "__main__":
    print("Current Directory: ", currentdir)
    print("Grand Grand Parent Directory: ", grandgrandparentdir)
    dataset_path = '/media/data3/tjtanaa/kitti_dataset'
    database_path = os.path.join(dataset_path, "gt_database")
    classes=['Car']
    pkl_file_name = os.path.join(database_path, '%s_gt_database_level_%s.pkl' % ('train', '-'.join(classes)))

    with open(pkl_file_name, 'rb') as f:
        db = pickle.load(f) 
        print("There are %d objects in the %s database" % (len(db), '%s_gt_database_level_%s.pkl' % ('train', '-'.join(classes))))
        print(len(db[0].keys()), "Keys Available: ", db[0].keys())
        # for keys in db: 
        #     print(keys, '=>', db[keys]) 
        #     break

        # save the points as [h, w, l] to be plotted as 3d points later
        dimensions_np = np.zeros((len(db), 3))

        for obj_ind in tqdm.tqdm(range(len(db))):
            obj = db[obj_ind]
            dimensions_np[obj_ind, 0] = obj['gt_box3d'][3]
            dimensions_np[obj_ind, 1] = obj['gt_box3d'][4]
            dimensions_np[obj_ind, 2] = obj['gt_box3d'][5]

        # save to current directory
        with open(os.path.join(currentdir, '%s_dimensions_np.npy' % ('-'.join(classes))), 'wb') as numpyfile:
            np.save(numpyfile, dimensions_np)

    # with open(pkl_file_name, 'rb') as f:
    #     db = pickle.load(f) 
    #     print("There are %d objects in the %s database" % (len(db), '%s_gt_database_level_%s.pkl' % ('train', '-'.join(classes))))
    #     print(len(db[0].keys()), "Keys Available: ", db[0].keys())
    #     # for keys in db: 
    #     #     print(keys, '=>', db[keys]) 
    #     #     # break

    #     # save the points as [h, w, l] to be plotted as 3d points later
    #     dimensions_np = np.zeros((len(db), 3))

    #     for obj_ind in tqdm.tqdm(range(len(db))):
    #         obj = db[obj_ind]
    #         dimensions_np[obj_ind, 0] = obj['gt_box3d'][3]
    #         dimensions_np[obj_ind, 1] = obj['gt_box3d'][4]
    #         dimensions_np[obj_ind, 2] = obj['gt_box3d'][5]

    #     # with open(os.path.join(currentdir, 'gt_box3d_sample.npy'), 'wb') as f:
    #     #     np.save(f, db[0]['gt_box3d'])

    #     # with open(os.path.join(currentdir, 'point_cloud_sample.npy'), 'wb') as f:
    #     #     np.save(f, db[0]['points'])

    #     # with open(os.path.join(currentdir, 'intensity_sample.npy'), 'wb') as f:
    #     #     np.save(f, db[0]['intensity'])

    #     # save to current directory
    #     # with open(os.path.join(currentdir, '%s_dimensions_np.npy' % ('-'.join(classes))), 'wb') as numpyfile:
    #     #     np.save(numpyfile, dimensions_np)
    #     exit()