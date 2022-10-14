import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import logging

from .LieAlgebra import se3



# This function is based upon IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        logging.info('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.dtype = torch.get_default_dtype()

        # Camera
        self.is_monocular = conf.get_bool('is_monocular')
        self.camera_trainable = conf.get_bool('camera_trainable')

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.jpg')))
        if len(self.images_lis) == 0:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
        if len(self.masks_lis) == 0:
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # Depth
        self.use_depth = conf.get_bool('use_depth')
        if self.use_depth:
            self.depth_scale = conf.get_float('depth_scale', default=1000.)
            self.depths_lis = sorted(glob(os.path.join(self.data_dir, 'depth/*.jpg')))
            if len(self.depths_lis) == 0:
                self.depths_lis = sorted(glob(os.path.join(self.data_dir, 'depth/*.png')))
            self.depths_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in self.depths_lis]) / self.depth_scale
            self.depths_np[self.depths_np == 0] = -1. # avoid nan values
            self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).to(self.dtype).cpu()

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrinsics_all = []
        poses_all = []
        # Depth, needs x,y,z have equal scale
        self.scales_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            # Depth
            self.scales_all.append(np.array([1.0/scale_mat[0,0]]))
            intrinsics_all.append(torch.from_numpy(intrinsics).to(self.dtype))
            poses_all.append(torch.from_numpy(pose).to(self.dtype)) # the inverse of extrinsic matrix

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 3]
        # Depth
        self.scales_all = torch.from_numpy(np.stack(self.scales_all)).to(self.dtype).to(self.device) # [n_images, 3]
        intrinsics_all = torch.stack(intrinsics_all).to(self.device) # [n_images, 4, 4]
        poses_all = torch.stack(poses_all).to(self.device) # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]

        # Camera
        if self.is_monocular:
            self.intrinsics_paras = torch.stack((intrinsics_all[:1,0,0], intrinsics_all[:1,1,1], \
                                                intrinsics_all[:1,0,2], intrinsics_all[:1,1,2]),
                                                    dim=1) # [1, 4]: (fx, fy, cx, cy)
        else:
            self.intrinsics_paras = torch.stack((intrinsics_all[:,0,0], intrinsics_all[:,1,1], \
                                                intrinsics_all[:,0,2], intrinsics_all[:,1,2]),
                                                    dim=1) # [n_images, 4]: (fx, fy, cx, cy)
        # Depth
        if self.use_depth:
            self.depth_intrinsics_paras = self.intrinsics_paras.clone().detach()
        self.poses_paras = se3.log(poses_all) # [n_images, 6]
        if self.camera_trainable:
            self.intrinsics_paras.requires_grad_()
            if self.use_depth:
                self.depth_intrinsics_paras.requires_grad_()
            self.poses_paras.requires_grad_()
        else:
            self.static_paras_to_mat()

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        logging.info('Load data: End')


    # Camera
    def static_paras_to_mat(self):
        fx, fy, cx, cy = self.intrinsics_paras[:,0], self.intrinsics_paras[:,1],\
                            self.intrinsics_paras[:,2], self.intrinsics_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_all_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        self.intrinsics_all_inv = torch.cat((torch.cat(
                                (intrinsics_all_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        if self.use_depth:
            fx_d, fy_d, cx_d, cy_d = self.depth_intrinsics_paras[:,0], self.depth_intrinsics_paras[:,1],\
                            self.depth_intrinsics_paras[:,2], self.depth_intrinsics_paras[:,3]
            zeros = torch.zeros_like(fx_d)
            ones = torch.ones_like(fx_d)
            depth_intrinsics_all_inv_mat = torch.stack((torch.stack(
                                    (1/fx_d, zeros, -cx_d/fx_d), dim=1), torch.stack(
                                    (zeros, 1/fy_d, -cy_d/fy_d), dim=1), torch.stack(
                                    (zeros, zeros, ones), dim=1)),
                                        dim=1)
            self.depth_intrinsics_all_inv = torch.cat((torch.cat(
                                    (depth_intrinsics_all_inv_mat, torch.stack(
                                    (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                    (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                        dim=1)
        self.poses_all = se3.exp(self.poses_paras)


    def dynamic_paras_to_mat(self, img_idx, add_depth=False):
        if self.is_monocular:
            intrinsic_paras = self.intrinsics_paras[:1, :]
        else:
            intrinsic_paras = self.intrinsics_paras[img_idx:(img_idx+1), :]
        fx, fy, cx, cy = intrinsic_paras[:,0], intrinsic_paras[:,1], intrinsic_paras[:,2], intrinsic_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        intrinsic_inv = torch.cat((torch.cat(
                                (intrinsics_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        pose_paras = self.poses_paras[img_idx:(img_idx+1), :]
        pose = se3.exp(pose_paras)
        if add_depth:
            if self.is_monocular:
                depth_intrinsic_paras = self.depth_intrinsics_paras[:1, :]
            else:
                depth_intrinsic_paras = self.depth_intrinsics_paras[img_idx:(img_idx+1), :]
            fx_d, fy_d, cx_d, cy_d = depth_intrinsic_paras[:,0], depth_intrinsic_paras[:,1],\
                                        depth_intrinsic_paras[:,2], depth_intrinsic_paras[:,3]
            zeros = torch.zeros_like(fx_d)
            ones = torch.ones_like(fx_d)
            depth_intrinsics_inv_mat = torch.stack((torch.stack(
                                    (1/fx_d, zeros, -cx_d/fx_d), dim=1), torch.stack(
                                    (zeros, 1/fy_d, -cy_d/fy_d), dim=1), torch.stack(
                                    (zeros, zeros, ones), dim=1)),
                                        dim=1)
            depth_intrinsic_inv = torch.cat((torch.cat(
                                    (depth_intrinsics_inv_mat, torch.stack(
                                    (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                    (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                        dim=1)
            return intrinsic_inv.squeeze(), pose.squeeze(), depth_intrinsic_inv.squeeze()
        return intrinsic_inv.squeeze(), pose.squeeze()


    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    # Depth
    def gen_rays_at_depth(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # Normalize
        npixels_x = (pixels_x - (self.W-1)/2) / ((self.W-1)/2)
        npixels_y = (pixels_y - (self.H-1)/2) / ((self.H-1)/2)
        mask = self.masks[img_idx].permute(2,0,1)[None,...].to(self.device) # 1, 3, H, W
        mask = mask[:,:1,...] # 1, 1, H, W
        depth = self.depths[img_idx][None,None,...].to(self.device) # 1, 1, H, W
        npixels = torch.cat([npixels_x.unsqueeze(-1).unsqueeze(0),npixels_y.unsqueeze(-1).unsqueeze(0)], dim=-1) # 1, W, H, 2
        # grid_sample: sample image on (x_i, y_i)
        mask = F.grid_sample(mask, npixels, mode='nearest', padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        depth = F.grid_sample(depth, npixels, padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p_d = p.clone().detach()
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, None, :3, :3], p_d[:, :, :, None]).squeeze() * self.scales_all[img_idx, :] # W, H, 3
        rays_s = torch.matmul(pose[None, None, :3, :3], p_d[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0,1), rays_v.transpose(0,1), rays_s.transpose(0,1), mask.transpose(0,1)


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        mask = self.masks[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1) # batch_size, 10


    # Depth
    def gen_random_rays_at_depth(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)
        mask = self.masks[img_idx].to(self.device)
        depth = self.depths[img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)] # batch_size, 3
        mask = mask[(pixels_y, pixels_x)] # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, :3, :3], p_d[:, :, None]).squeeze() * self.scales_all[img_idx, :] # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True) # batch_size, 1
        rays_s = torch.matmul(pose[None, :3, :3], p_d[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1) # batch_size, 14


    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far


    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


    def get_image_size(self):
        return self.H, self.W

    def depth_at(self, idx, resolution_level):
        depth_img = cv.resize(self.depths_np[idx],
                                (self.W // resolution_level, self.H // resolution_level))
        depth_img = 255. - np.clip(depth_img/depth_img.max(), a_max=1, a_min=0) * 255.
        return cv.applyColorMap(np.uint8(depth_img), cv.COLORMAP_JET)
