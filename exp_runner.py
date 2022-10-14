import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory

from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, DeformNetwork, AppearanceNetwork, TopoNetwork
from models.renderer import NeuSRenderer, DeformNeuSRenderer



class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')
        self.gpu = torch.cuda.current_device()
        self.dtype = torch.get_default_dtype()

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Deform
        self.use_deform = self.conf.get_bool('train.use_deform')
        if self.use_deform:
            self.deform_dim = self.conf.get_int('model.deform_network.d_feature')
            self.deform_codes = torch.randn(self.dataset.n_images, self.deform_dim, requires_grad=True).to(self.device)
            self.appearance_dim = self.conf.get_int('model.appearance_rendering_network.d_global_feature')
            self.appearance_codes = torch.randn(self.dataset.n_images, self.appearance_dim, requires_grad=True).to(self.device)

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.important_begin_iter = self.conf.get_int('model.neus_renderer.important_begin_iter')
        # Anneal
        self.max_pe_iter = self.conf.get_int('train.max_pe_iter')

        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.validate_idx = self.conf.get_int('train.validate_idx', default=-1)
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.test_batch_size = self.conf.get_int('test.test_batch_size')

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Depth
        self.use_depth = self.conf.get_bool('dataset.use_depth')
        if self.use_depth:
            self.geo_weight = self.conf.get_float('train.geo_weight')
            self.angle_weight = self.conf.get_float('train.angle_weight')

        # Deform
        if self.use_deform:
            self.deform_network = DeformNetwork(**self.conf['model.deform_network']).to(self.device)
            self.topo_network = TopoNetwork(**self.conf['model.topo_network']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        # Deform
        if self.use_deform:
            self.color_network = AppearanceNetwork(**self.conf['model.appearance_rendering_network']).to(self.device)
        else:
            self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        # Deform
        if self.use_deform:
            self.renderer = DeformNeuSRenderer(self.report_freq,
                                     self.deform_network,
                                     self.topo_network,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])
        else:
            self.renderer = NeuSRenderer(self.sdf_network,
                                        self.deviation_network,
                                        self.color_network,
                                        **self.conf['model.neus_renderer'])

        # Load Optimizer
        params_to_train = []
        if self.use_deform:
            params_to_train += [{'name':'deform_network', 'params':self.deform_network.parameters(), 'lr':self.learning_rate}]
            params_to_train += [{'name':'topo_network', 'params':self.topo_network.parameters(), 'lr':self.learning_rate}]
            params_to_train += [{'name':'deform_codes', 'params':self.deform_codes, 'lr':self.learning_rate}]
            params_to_train += [{'name':'appearance_codes', 'params':self.appearance_codes, 'lr':self.learning_rate}]
        params_to_train += [{'name':'sdf_network', 'params':self.sdf_network.parameters(), 'lr':self.learning_rate}]
        params_to_train += [{'name':'deviation_network', 'params':self.deviation_network.parameters(), 'lr':self.learning_rate}]
        params_to_train += [{'name':'color_network', 'params':self.color_network.parameters(), 'lr':self.learning_rate}]

        # Camera
        if self.dataset.camera_trainable:
            params_to_train += [{'name':'intrinsics_paras', 'params':self.dataset.intrinsics_paras, 'lr':self.learning_rate}]
            params_to_train += [{'name':'poses_paras', 'params':self.dataset.poses_paras, 'lr':self.learning_rate}]
            # Depth
            if self.use_depth:
                params_to_train += [{'name':'depth_intrinsics_paras', 'params':self.dataset.depth_intrinsics_paras, 'lr':self.learning_rate}]

        self.optimizer = torch.optim.Adam(params_to_train)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if self.mode == 'validate_pretrained':
                latest_model_name = 'pretrained.pth'
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs
        if self.mode[:5] == 'train':
            self.file_backup()


    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            # Deform
            if self.use_deform:
                image_idx = image_perm[self.iter_step % len(image_perm)]
                # Deform
                deform_code = self.deform_codes[image_idx][None, ...]
                if iter_i == 0:
                    print('The files will be saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    self.validate_observation_mesh(self.validate_idx)
                # Depth
                if self.use_depth:
                    data = self.dataset.gen_random_rays_at_depth(image_idx, self.batch_size)
                    rays_o, rays_d, rays_s, rays_l, true_rgb, mask = \
                        data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 13], data[:, 13: 14]
                else:
                    data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)
                    rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

                # Deform
                appearance_code = self.appearance_codes[image_idx][None, ...]
                # Anneal
                alpha_ratio = max(min(self.iter_step/self.max_pe_iter, 1.), 0.)

                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).to(self.dtype)
                else:
                    mask = torch.ones_like(mask)
                mask_sum = mask.sum() + 1e-5
                
                render_out = self.renderer.render(deform_code, appearance_code, rays_o, rays_d, near, far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                alpha_ratio=alpha_ratio, iter_step=self.iter_step)
                # Depth
                if self.use_depth:
                    sdf_loss, angle_loss, valid_depth_region =\
                        self.renderer.errorondepth(deform_code, rays_o, rays_d, rays_s, mask,
                                            alpha_ratio, iter_step=self.iter_step)
                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_o_error = render_out['gradient_o_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                depth_map = render_out['depth_map']

                # Loss
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_o_error

                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-5, 1.0 - 1e-5), mask)
                # Depth
                if self.use_depth:
                    depth_minus = (depth_map - rays_l) * valid_depth_region
                    depth_loss = F.l1_loss(depth_minus, torch.zeros_like(depth_minus), reduction='sum') \
                                    / (valid_depth_region.sum() + 1e-5)
                    if self.iter_step < self.important_begin_iter:
                        rgb_scale = 0.1
                        geo_scale = 10.0
                        regular_scale = 10.0
                        geo_loss = sdf_loss
                    elif self.iter_step < self.max_pe_iter:
                        rgb_scale = 1.0
                        geo_scale = 1.0
                        regular_scale = 10.0
                        geo_loss = 0.5 * (depth_loss + sdf_loss)
                    else:
                        rgb_scale = 1.0
                        geo_scale = 0.1
                        regular_scale = 1.0
                        geo_loss = 0.5 * (depth_loss + sdf_loss)
                else:
                    if self.iter_step < self.max_pe_iter:
                        regular_scale = 10.0
                    else:
                        regular_scale = 1.0
                
                if self.use_depth:
                    loss = color_fine_loss * rgb_scale +\
                        (geo_loss * self.geo_weight + angle_loss * self.angle_weight) * geo_scale +\
                        (eikonal_loss * self.igr_weight + mask_loss * self.mask_weight) * regular_scale
                else:
                    loss = color_fine_loss +\
                        (eikonal_loss * self.igr_weight + mask_loss * self.mask_weight) * regular_scale

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                del color_fine_loss
                # Depth
                if self.use_depth:
                    self.writer.add_scalar('Loss/sdf_loss', sdf_loss, self.iter_step)
                    self.writer.add_scalar('Loss/depth_loss', depth_loss, self.iter_step)
                    self.writer.add_scalar('Loss/angle_loss', angle_loss, self.iter_step)
                    del sdf_loss
                    del depth_loss
                    del angle_loss
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                del eikonal_loss
                del mask_loss

                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print('The files have been saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    print('iter:{:8>d} loss={} idx={} alpha_ratio={} lr={}'.format(self.iter_step, loss, image_idx,
                            alpha_ratio, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image(self.validate_idx)
                    # Depth
                    if self.use_depth:
                        self.validate_image_with_depth(self.validate_idx)

                if self.iter_step % self.val_mesh_freq == 0:
                    self.validate_observation_mesh(self.validate_idx)

                self.update_learning_rate()
                
                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

            else:
                if self.iter_step == 0:
                    self.validate_mesh()
                data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

                rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).to(self.dtype)
                else:
                    mask = torch.ones_like(mask)

                mask_sum = mask.sum() + 1e-5
                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio())

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']

                # Loss
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_error

                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

                loss = color_fine_loss +\
                    eikonal_loss * self.igr_weight +\
                    mask_loss * self.mask_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                del color_fine_loss
                del eikonal_loss
                if self.mask_weight > 0.0:
                    self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                    del mask_loss
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print('The file have been saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image()

                if self.iter_step % self.val_mesh_freq == 0:
                    self.validate_mesh()

                self.update_learning_rate()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

        
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)


    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])


    def update_learning_rate(self, scale_factor=1):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        learning_factor *= scale_factor

        current_learning_rate = self.learning_rate * learning_factor
        for g in self.optimizer.param_groups:
            if g['name'] in ['intrinsics_paras', 'poses_paras', 'depth_intrinsics_paras']:
                g['lr'] = current_learning_rate * 1e-1
            elif self.iter_step >= self.max_pe_iter and g['name'] == 'deviation_network':
                g['lr'] = current_learning_rate * 1.5
            else:
                g['lr'] = current_learning_rate


    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        logging.info('File Saved')


    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        # Deform
        if self.use_deform:
            self.deform_network.load_state_dict(checkpoint['deform_network'])
            self.topo_network.load_state_dict(checkpoint['topo_network'])
            self.deform_codes = torch.from_numpy(checkpoint['deform_codes']).to(self.device).requires_grad_()
            self.appearance_codes = torch.from_numpy(checkpoint['appearance_codes']).to(self.device).requires_grad_()
            logging.info('Use_deform True')
        self.dataset.intrinsics_paras = torch.from_numpy(checkpoint['intrinsics_paras']).to(self.device)
        self.dataset.poses_paras = torch.from_numpy(checkpoint['poses_paras']).to(self.device)
        # Depth
        if self.use_depth:
            self.dataset.depth_intrinsics_paras = torch.from_numpy(checkpoint['depth_intrinsics_paras']).to(self.device)
        # Camera
        if self.dataset.camera_trainable:
            self.dataset.intrinsics_paras.requires_grad_()
            self.dataset.poses_paras.requires_grad_()
            # Depth
            if self.use_depth:
                self.dataset.depth_intrinsics_paras.requires_grad_()
        else:
            self.dataset.static_paras_to_mat()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')


    def save_checkpoint(self):
        # Depth
        if self.use_depth:
            depth_intrinsics_paras = self.dataset.depth_intrinsics_paras.data.cpu().numpy()
        else:
            depth_intrinsics_paras = self.dataset.intrinsics_paras.data.cpu().numpy()
        # Deform
        if self.use_deform:
            checkpoint = {
                'deform_network': self.deform_network.state_dict(),
                'topo_network': self.topo_network.state_dict(),
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
                'deform_codes': self.deform_codes.data.cpu().numpy(),
                'appearance_codes': self.appearance_codes.data.cpu().numpy(),
                'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
                'depth_intrinsics_paras': depth_intrinsics_paras,
            }
        else:
            checkpoint = {
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
                'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
                'depth_intrinsics_paras': depth_intrinsics_paras,
            }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>7d}.pth'.format(self.iter_step)))


    def validate_image(self, idx=-1, resolution_level=-1, mode='train', normal_filename='normals', rgb_filename='rgbs', depth_filename='depths'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        # Deform
        if self.use_deform:
            deform_code = self.deform_codes[idx][None, ...]
            appearance_code = self.appearance_codes[idx][None, ...]
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if mode == 'train':
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            if self.use_deform:
                render_out = self.renderer.render(deform_code,
                                                appearance_code,
                                                rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.),
                                                iter_step=self.iter_step)
                render_out['gradients'] = render_out['gradients_o']
            else:
                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio())
            
            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                if self.iter_step >= self.important_begin_iter:
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                else:
                    n_samples = self.renderer.n_samples
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out['depth_map'] # Annotate it if you want to visualize estimated depth map!
            if feasible('depth_map'):
                out_depth_fine.append(render_out['depth_map'].detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # Camera
            if self.dataset.camera_trainable:
                _, pose = self.dataset.dynamic_paras_to_mat(idx)
            else:
                pose = self.dataset.poses_all[idx]
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        depth_img = None
        if len(out_depth_fine) > 0:
            depth_img = np.concatenate(out_depth_fine, axis=0)
            depth_img = depth_img.reshape([H, W, 1, -1])
            depth_img = 255. - np.clip(depth_img/depth_img.max(), a_max=1, a_min=0) * 255.
            depth_img = np.uint8(depth_img)
        os.makedirs(os.path.join(self.base_exp_dir, rgb_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, normal_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, depth_filename), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        rgb_filename,
                                        '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        normal_filename,
                                        '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                           normal_img[..., i])
            
            if len(out_depth_fine) > 0:
                if self.use_depth:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            depth_filename,
                                            '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                            np.concatenate([cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET),
                                                self.dataset.depth_at(idx, resolution_level=resolution_level)]))
                else:
                    cv.imwrite(os.path.join(self.base_exp_dir, depth_filename,
                                            '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                                            cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET))


    def validate_image_with_depth(self, idx=-1, resolution_level=-1, mode='train'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        # Deform
        if self.use_deform:
            deform_code = self.deform_codes[idx][None, ...]
            appearance_code = self.appearance_codes[idx][None, ...]
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if mode == 'train':
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, rays_s, mask = self.dataset.gen_rays_at_depth(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)
        rays_s = rays_s.reshape(-1, 3).split(batch_size)
        mask = (mask > 0.5).to(self.dtype).detach().cpu().numpy()[..., None]

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch, rays_s_batch in zip(rays_o, rays_d, rays_s):
            color_batch, gradients_batch = self.renderer.renderondepth(deform_code,
                                                    appearance_code,
                                                    rays_o_batch,
                                                    rays_d_batch,
                                                    rays_s_batch,
                                                    alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.))

            out_rgb_fine.append(color_batch.detach().cpu().numpy())
            out_normal_fine.append(gradients_batch.detach().cpu().numpy())
            del color_batch, gradients_batch

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            img_fine = img_fine * mask

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # w/ pose -> w/o pose. similar: world -> camera
            # Camera
            if self.dataset.camera_trainable:
                _, pose = self.dataset.dynamic_paras_to_mat(idx)
            else:
                pose = self.dataset.poses_all[idx]
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            normal_img = normal_img * mask

        os.makedirs(os.path.join(self.base_exp_dir, 'rgbsondepth'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normalsondepth'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'rgbsondepth',
                                        '{:0>8d}_depth_{}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normalsondepth',
                                        '{:0>8d}_depth_{}.png'.format(self.iter_step, idx)),
                           normal_img[..., i])


    def validate_all_image(self, resolution_level=-1):
        for image_idx in range(self.dataset.n_images):
            self.validate_image(image_idx, resolution_level, 'test', 'validations_normals', 'validations_rgbs', 'validations_depths')
            print('Used GPU:', self.gpu)


    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


    # Deform
    def validate_canonical_mesh(self, world_space=False, resolution=64, threshold=0.0, filename='meshes_canonical'):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_canonical_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                        alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.))
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_canonical.ply'.format(self.iter_step)))

        logging.info('End')

    
    # Deform
    def validate_observation_mesh(self, idx=-1, world_space=False, resolution=64, threshold=0.0, filename='meshes'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        # Deform
        deform_code = self.deform_codes[idx][None, ...]
        
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_observation_geometry(deform_code, bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                        alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.))
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_{}.ply'.format(self.iter_step, idx)))

        logging.info('End')


    # Deform
    def validate_all_mesh(self, world_space=False, resolution=64, threshold=0.0):
        for image_idx in range(self.dataset.n_images):
            self.validate_observation_mesh(image_idx, world_space, resolution, threshold, 'validations_meshes')
            print('Used GPU:', self.gpu)



# This implementation is built upon NeuS: https://github.com/Totoro97/NeuS
if __name__ == '__main__':
    print('Welcome to NDR')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(args.gpu)

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode[:8] == 'validate':
        if runner.use_deform:
            runner.validate_all_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
            runner.validate_all_image(resolution_level=1)
        else:
            runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
            runner.validate_all_image(resolution_level=1)
