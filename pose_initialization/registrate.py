import numpy as np
from PIL import Image
import os
import argparse
from glob import glob

"""read depth image
"""
def depth_read(filename, depth_scale=1000.):
    # loads depth map D from png file
    # and returns it as a numpy array.

    depth_png = np.array(Image.open(filename), dtype=int) # (h, w)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    # assert(np.max(depth_png) > 255)

    # Unit (Important)
    depth = depth_png.astype(np.float32) / depth_scale # use the depth images in the meter unit
    depth[depth_png == 0] = -1.
    return depth

"""read mask image
"""
def mask_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array.

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

"""Based on numpy & scipy, camera2coarse transform
"""
def camera2coarse(xyz, trans):
    return xyz - np.expand_dims(trans, 0)

def coarse2fine(xyz, rot, trans):
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
    parser.add_argument('--registration_alg_path', type=str, default='./Fast-Robust-ICP/build/FRICP')
    parser.add_argument('--rigid_registrate', type=int, default=1)
    args = parser.parse_args()

    data_path = args.data_path
    depth_scale = args.depth_scale
    rigid_registrate = args.rigid_registrate
    registration_alg_path = args.registration_alg_path
    
    if os.path.exists(data_path+'intrinsics.txt'):
        print('Do not find intrinsic.txt')
        exit()
    intrinsic_matrix = np.loadtxt(data_path+'intrinsics.txt')
    # fx, fy, cx, cy
    intrinsic_paras = [intrinsic_matrix[0,0], intrinsic_matrix[1,1], intrinsic_matrix[0,2], intrinsic_matrix[1,2]]
    
    if not os.path.exists(data_path + 'intermediate/pointclouds/'):
        os.makedirs(data_path + 'intermediate/pointclouds/', exist_ok=True)
    depths_lis = sorted(glob(os.path.join(data_path, 'depth/*.jpg')))
    if len(depths_lis) == 0:
        depths_lis = sorted(glob(os.path.join(data_path, 'depth/*.png')))
    masks_lis = sorted(glob(os.path.join(data_path, 'mask/*.jpg')))
    if len(masks_lis) == 0:
        masks_lis = sorted(glob(os.path.join(data_path, 'mask/*.png')))
    total_num = len(depths_lis)
    if total_num != len(masks_lis):
        print('The number of depth is not equal to the number of mask!')
        exit()
    
    points_list = []
    transformations = np.repeat(np.eye(4)[None, ...], total_num, axis=0)

    for i in range(total_num):
        depth_path = depths_lis[i]
        mask_path = masks_lis[i]

        depth = depth_read(depth_path, depth_scale)
        mask = mask_read(mask_path) * (depth!=-1.) # mask with depth

        xyz = mask2camera(mask, depth, intrinsic_paras) # (n_points, 3)

        if rigid_registrate:
            current_pointcloud_path = data_path + 'intermediate/pointclouds/' + str(i).zfill(8) + '.obj'
            
            if rigid_registrate == 1 and i == 0:
                first_pointcloud_path = current_pointcloud_path
            trans_coarse = xyz.mean(0)
            transformation_coarse = np.eye(4)
            transformation_coarse[:3, -1] = trans_coarse
            xyz_coarse = camera2coarse(xyz, trans_coarse) # remove coarse
            obj_write(current_pointcloud_path, xyz_coarse)

            if i != 0:
                # where is rot and trans_fine
                current_record_path = data_path + 'intermediate/pointclouds/' + str(i).zfill(8) + '_'
                # command path: https://github.com/yaoyx689/Fast-Robust-ICP
                # {method_path}+{target_path}+{source_path}+{record_path}+{method_index}
                # target_path: current, source_path: first
                if rigid_registrate == 1:
                    os.system(registration_alg_path + ' ' + current_pointcloud_path + ' ' + \
                        first_pointcloud_path + ' ' + current_record_path + ' 3') # 3: Robust ICP
                elif rigid_registrate == 2:
                    os.system(registration_alg_path + ' ' + current_pointcloud_path + ' ' + \
                        previous_pointcloud_path + ' ' + current_record_path + ' 3') # 3: Robust ICP
                transformation_fine = np.loadtxt(current_record_path + 'm3trans.txt')
                if rigid_registrate == 1:
                    transformations[i, ...] = transformation_coarse @ transformation_fine # First -> Current
                    rot = transformation_fine[:3, :3]
                    trans_fine = transformation_fine[:3, -1]
                elif rigid_registrate == 2:
                    transformations[i:, ...] = transformation_fine[None, ...] @ transformations[i:, ...] # Previous -> Current
                    transformations[i, ...] = transformation_coarse @ transformations[i, ...] # Previous -> Current
                    rot = transformations[i, :3, :3]
                    trans_fine = transformations[i, :3, -1]
                xyz_world = coarse2fine(xyz_coarse, rot, trans_fine) # remove fine
            else:
                transformations[i, ...] = transformation_coarse
                xyz_world = xyz_coarse
            if rigid_registrate == 2:
                previous_pointcloud_path = current_pointcloud_path
        else:
            trans_coarse = xyz.mean(0)
            transformation_coarse = np.eye(4)
            transformation_coarse[:3, -1] = trans_coarse
            transformations[i, ...] = transformation_coarse
            xyz_world = camera2coarse(xyz, trans_coarse)

        points_list += xyz_world.tolist()
    
    whole_points = np.array(points_list)
    print('Center point in all at:', whole_points.mean(0))
    whole_radius = np.linalg.norm(whole_points, ord=2, axis=-1)
    # Very important!!! Denoise Module
    radius_denoise = whole_radius[whole_radius<=np.percentile(whole_radius, 95)] # 95: hyper para for denosing (Lower, Stricter).
    radius = radius_denoise.max() * 1.2 # hyper para for range
    
    np.savetxt(data_path + 'intermediate/radius.txt', radius.reshape(-1), fmt='%.8f')
    np.save(data_path + 'intermediate/transformations.npy', transformations.reshape(-1, 16)) # transformations