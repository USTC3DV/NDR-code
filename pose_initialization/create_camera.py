import numpy as np
import os
import shutil
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--object_scale', type=float, default=1.05,
                        help='hyper-parameter for radius scaling.'
                        'Larger it is, smaller scaled object will be.')
args = parser.parse_args()

data_path = args.data_path
object_scale = args.object_scale
intrinsic_matrix = np.loadtxt(data_path+'intrinsics.txt')
intrinsic_paras = [intrinsic_matrix[0,0], intrinsic_matrix[1,1], intrinsic_matrix[0,2], intrinsic_matrix[1,2]] # fx, fy, cx, cy
intrinsic = np.diag([intrinsic_paras[0], intrinsic_paras[1], 1.0, 1.0]).astype(np.float32)
intrinsic[0, 2] = intrinsic_paras[2]
intrinsic[1, 2] = intrinsic_paras[3]

depths_lis = sorted(glob(os.path.join(data_path, 'depth/*.jpg')))
if len(depths_lis) == 0:
    depths_lis = sorted(glob(os.path.join(data_path, 'depth/*.png')))
total_num = len(depths_lis)

# save world matrix (world_mat_{i}), normalization matrix (scale_mat_{i}).
transformations = np.load(data_path + 'intermediate/transformations.npy').reshape(-1, 4, 4)
radius = np.loadtxt(data_path + 'intermediate/radius.txt')
if os.path.exists(data_path + 'intermediate/ref.xyz'):
    ref_pts = np.loadtxt(data_path + 'intermediate/ref.xyz')
    np.savetxt(data_path + 'intermediate/radius.txt.bk', radius.reshape(-1), fmt='%.8f')
    ref_radius = np.linalg.norm(ref_pts, ord=2, axis=-1, keepdims=False)
    radius *= ref_radius.max() * object_scale # hyper para: 1.0-1.2
    np.savetxt(data_path + 'intermediate/radius.txt', radius.reshape(-1), fmt='%.8f')

cam_dict = dict()
for i in range(total_num):
    w2c = transformations[i, ...]
    world_mat = intrinsic @ w2c
    world_mat = world_mat.astype(np.float32)

    cam_dict['world_mat_{}'.format(i)] = world_mat

scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
for i in range(total_num):
    cam_dict['scale_mat_{}'.format(i)] = scale_mat

np.savez(os.path.join(data_path, 'cameras_sphere.npz'), **cam_dict)
print('Process done!')
