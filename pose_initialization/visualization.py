import numpy as np
from PIL import Image
import os
import argparse
from glob import glob

"""read depth image
"""
def depth_read(filename, depth_scale=1000.):
    # loads depth map D from png file
    # and returns it as a numpy array

    depth_png = np.array(Image.open(filename), dtype=int) # (h, w)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    # assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / depth_scale # use the depth images in the meter unit
    depth[depth_png == 0] = -1.
    return depth

"""read mask image
"""
def mask_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array

    mask_png = np.array(Image.open(filename), dtype=int) / 255 # (h, w)
    # make sure we have a proper 16bit depth map here.. not 8bit!

    if mask_png.ndim == 3:
        mask = np.sum(mask_png[...,:3], 2) / 3.
    elif mask_png.ndim == 2:
        mask = mask_png
    else:
        mask = None
        print('The dim of mask is wrong!')
        exit
    return (mask > 0.5) + 0

"""Based on numpy, pixel2camera transform
"""
def mask2camera(mask, depth, intrinsic_paras):
    fx, fy, cx, cy = intrinsic_paras[0], intrinsic_paras[1], intrinsic_paras[2], intrinsic_paras[3]
    height, width = depth.shape[0], depth.shape[1]

    matrix_u = np.arange(width).repeat(height, 0).reshape(width, height).transpose().astype(np.float32)
    matrix_v = np.arange(height).repeat(width, 0).reshape(height, width).astype(np.float32)

    x = depth * (matrix_u - cx) / fx
    y = depth * (matrix_v - cy) / fy
    xyz = np.concatenate([np.expand_dims(x, 2), np.expand_dims(y, 2), np.expand_dims(depth, 2)], 2)
    xyz_mask = xyz[np.nonzero(mask)]

    return xyz_mask

"""Based on numpy & scipy, camera2world transform
"""
def camera2world(xyz, rot, trans):
    return np.dot(xyz-np.expand_dims(trans, 0), rot)

"""Write to obj file
"""
def obj_write(save_path, points):
    file = open(save_path, 'w')
    for i in range(points.shape[0]):
        file.write('v ')
        for j in range(points.shape[1]):
            file.write(str(points[i, j]) + ' ')
        file.write('\n')
    file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--depth_scale', type=float, default=1000.)
    args = parser.parse_args()

    data_path = args.data_path
    depth_scale = args.depth_scale
    intrinsic_matrix = np.loadtxt(data_path+'intrinsics.txt')
    intrinsic_paras = [intrinsic_matrix[0,0], intrinsic_matrix[1,1], intrinsic_matrix[0,2], intrinsic_matrix[1,2]] # fx, fy, cx, cy

    if not os.path.exists(data_path + 'intermediate/colored_pointclouds/'):
        os.makedirs(data_path + 'intermediate/colored_pointclouds/', exist_ok=True)
    depths_lis = sorted(glob(os.path.join(data_path, 'depth/*.jpg')))
    if len(depths_lis) == 0:
        depths_lis = sorted(glob(os.path.join(data_path, 'depth/*.png')))
    masks_lis = sorted(glob(os.path.join(data_path, 'mask/*.jpg')))
    if len(masks_lis) == 0:
        masks_lis = sorted(glob(os.path.join(data_path, 'mask/*.png')))
    images_lis = sorted(glob(os.path.join(data_path, 'rgb/*.jpg')))
    if len(images_lis) == 0:
        images_lis = sorted(glob(os.path.join(data_path, 'rgb/*.png')))
    total_num = len(depths_lis)
    if total_num != len(masks_lis) or total_num != len(images_lis):
        print('The number of depth is not equal to the number of mask\n'
            'or the number of depth is not equal to the number of rgb')
        exit()

    transformations = np.load(os.path.join(data_path, 'intermediate/transformations.npy')).astype(np.float32).reshape(-1, 4, 4)
    radius = np.loadtxt(os.path.join(data_path, 'intermediate/radius.txt')).astype(np.float32)
    for i in range(total_num):
        depth_path = depths_lis[i]
        mask_path = masks_lis[i]
        rgb_path = images_lis[i]
        colored_pointcloud_path = data_path + 'intermediate/colored_pointclouds/' + str(i).zfill(8) + '.obj'

        depth = depth_read(depth_path, depth_scale)
        mask = mask_read(mask_path) #  * (depth!=-1.)

        xyz = mask2camera(mask, depth, intrinsic_paras)
        rot = transformations[i, :3, :3]
        center = transformations[i, :3, -1]
        xyz_world = camera2world(xyz, rot, center)
        xyz_canon = xyz_world / radius

        image = np.array(Image.open(rgb_path), dtype=int)
        color = image[np.nonzero(mask)] / 255.

        # relax_inside_sphere, need sdf_loss
        xyz_canon_norm = np.linalg.norm(xyz_canon, ord=2, axis=-1, keepdims=False)
        xyz_color = np.concatenate((xyz_canon, color), 1)
        obj_write(colored_pointcloud_path, xyz_color)
        print(rgb_path)
