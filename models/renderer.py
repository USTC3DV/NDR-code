import torch
import torch.nn.functional as F
import numpy as np
import mcubes



def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 128 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('Threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is built upon NeRF
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1) # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# Deform
class DeformNeuSRenderer:
    def __init__(self,
                 report_freq,
                 deform_network,
                 ambient_network,
                 sdf_network,
                 deviation_network,
                 color_network,
                 begin_n_samples,
                 end_n_samples,
                 important_begin_iter,
                 n_importance,
                 up_sample_steps,
                 perturb):
        self.dtype = torch.get_default_dtype()
        # Deform
        self.deform_network = deform_network
        self.ambient_network = ambient_network
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.begin_n_samples = begin_n_samples
        self.end_n_samples = end_n_samples
        self.n_samples = self.begin_n_samples
        self.important_begin_iter = important_begin_iter
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.report_freq = report_freq


    def update_samples_num(self, iter_step, alpha_ratio=0.):
        if iter_step >= self.important_begin_iter:
            self.n_samples = int(self.begin_n_samples*(1-alpha_ratio)+self.end_n_samples*alpha_ratio)


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples


    def cat_z_vals(self, deform_code, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False,
                alpha_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            # Deform
            pts_canonical = self.deform_network(deform_code, pts, alpha_ratio)
            ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio)
            new_sdf = self.sdf_network.sdf(pts_canonical, ambient_coord, alpha_ratio).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf


    def render_core(self,
                    deform_code,
                    appearance_code,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    deform_network,
                    ambient_network,
                    sdf_network,
                    deviation_network,
                    color_network,
                    cos_anneal_ratio=0.0,
                    alpha_ratio=0.):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs_o = rays_d[:, None, :].expand(pts.shape) # view in observation space

        pts = pts.reshape(-1, 3)
        dirs_o = dirs_o.reshape(-1, 3)

        # Deform
        # observation space -> canonical space
        pts_canonical = deform_network(deform_code, pts, alpha_ratio)
        ambient_coord = ambient_network(deform_code, pts, alpha_ratio)
        sdf_nn_output = sdf_network(pts_canonical, ambient_coord, alpha_ratio)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        # Deform, gradients in observation space
        def gradient(deform_network=None, ambient_network=None, sdf_network=None, deform_code=None, x=None, alpha_ratio=None):
            x.requires_grad_(True)
            x_c = deform_network(deform_code, x, alpha_ratio)
            amb_coord = ambient_network(deform_code, x, alpha_ratio)
            y = sdf_network.sdf(x_c, amb_coord, alpha_ratio)
            
            # gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o =  torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            # Jacobian on pts
            y_0 = x_c[:, 0]
            y_1 = x_c[:, 1]
            y_2 = x_c[:, 2]
            d_output = torch.ones_like(y_0, requires_grad=False, device=y_0.device)
            grad_0 = torch.autograd.grad(
                outputs=y_0,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_1 = torch.autograd.grad(
                outputs=y_1,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_2 = torch.autograd.grad(
                outputs=y_2,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            gradient_pts =  torch.cat([grad_0, grad_1, grad_2], dim=1) # (batch_size, dim_out, dim_in)

            return gradient_o, gradient_pts

        # Deform
        # observation space -> canonical space
        gradients_o, pts_jacobian = gradient(deform_network, ambient_network, sdf_network, deform_code, pts, alpha_ratio)
        dirs_c = torch.bmm(pts_jacobian, dirs_o.unsqueeze(-1)).squeeze(-1) # view in observation space
        dirs_c = dirs_c / torch.linalg.norm(dirs_c, ord=2, dim=-1, keepdim=True)
        
        sampled_color = color_network(appearance_code, pts_canonical, gradients_o, \
            dirs_c, feature_vector, alpha_ratio).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs_o * gradients_o).sum(-1, keepdim=True) # observation

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).to(self.dtype).detach()
        relax_inside_sphere = (pts_norm < 1.2).to(self.dtype).detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        # depth map
        depth_map = torch.sum(weights * mid_z_vals, -1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        # Eikonal loss, observation + canonical
        gradient_o_error = (torch.linalg.norm(gradients_o.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        relax_inside_sphere_sum = relax_inside_sphere.sum() + 1e-5
        gradient_o_error = (relax_inside_sphere * gradient_o_error).sum() / relax_inside_sphere_sum

        return {
            'pts': pts.reshape(batch_size, n_samples, 3),
            'pts_canonical': pts_canonical.reshape(batch_size, n_samples, 3),
            'relax_inside_sphere': relax_inside_sphere,
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients_o': gradients_o.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'weights_sum': weights_sum,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_o_error': gradient_o_error,
            'inside_sphere': inside_sphere,
            'depth_map': depth_map
        }


    def render(self, deform_code, appearance_code, rays_o, rays_d, near, far, perturb_overwrite=-1,
            cos_anneal_ratio=0.0, alpha_ratio=0., iter_step=0):
        self.update_samples_num(iter_step, alpha_ratio)
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if iter_step >= self.important_begin_iter and self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts = pts.reshape(-1, 3)
                # Deform
                pts_canonical = self.deform_network(deform_code, pts, alpha_ratio)
                ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio)
                sdf = self.sdf_network.sdf(pts_canonical, ambient_coord, alpha_ratio).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(deform_code,
                                                  rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps),
                                                  alpha_ratio=alpha_ratio)

            n_samples = self.n_samples + self.n_importance

        # Render core
        ret_fine = self.render_core(deform_code,
                                    appearance_code,
                                    rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.deform_network,
                                    self.ambient_network,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    alpha_ratio=alpha_ratio)

        weights = ret_fine['weights']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'pts': ret_fine['pts'],
            'pts_canonical': ret_fine['pts_canonical'],
            'relax_inside_sphere': ret_fine['relax_inside_sphere'],
            'color_fine': ret_fine['color'],
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': ret_fine['weights_sum'],
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients_o': ret_fine['gradients_o'],
            'weights': weights,
            'gradient_o_error': ret_fine['gradient_o_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'depth_map': ret_fine['depth_map']
        }


    # Depth
    def renderondepth(self,
                    deform_code,
                    appearance_code,
                    rays_o,
                    rays_d,
                    rays_s,
                    alpha_ratio=0.):
        pts = rays_o + rays_s # n_rays, 3

        pts_canonical = self.deform_network(deform_code, pts, alpha_ratio)
        ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio)
        feature_vector = self.sdf_network(pts_canonical, ambient_coord, alpha_ratio)[:, 1:]

        # Deform, gradients in observation space
        def gradient(deform_network=None, ambient_network=None, sdf_network=None, deform_code=None, x=None, alpha_ratio=None):
            x.requires_grad_(True)
            x_c = deform_network(deform_code, x, alpha_ratio)
            amb_coord = ambient_network(deform_code, x, alpha_ratio)
            y = sdf_network.sdf(x_c, amb_coord, alpha_ratio)
            
            # gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o =  torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            # Jacobian on pts
            y_0 = x_c[:, 0]
            y_1 = x_c[:, 1]
            y_2 = x_c[:, 2]
            d_output = torch.ones_like(y_0, requires_grad=False, device=y_0.device)
            grad_0 = torch.autograd.grad(
                outputs=y_0,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_1 = torch.autograd.grad(
                outputs=y_1,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_2 = torch.autograd.grad(
                outputs=y_2,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            gradient_pts =  torch.cat([grad_0, grad_1, grad_2], dim=1) # (batch_size, dim_out, dim_in)

            return gradient_o, gradient_pts

        gradients_o, pts_jacobian = gradient(self.deform_network, self.ambient_network, self.sdf_network, deform_code, pts, alpha_ratio)
        dirs_c = torch.bmm(pts_jacobian, rays_d.unsqueeze(-1)).squeeze(-1) # view in observation space
        dirs_c = dirs_c / torch.linalg.norm(dirs_c, ord=2, dim=-1, keepdim=True)
        color = self.color_network(appearance_code, pts_canonical, gradients_o, \
            dirs_c, feature_vector, alpha_ratio)

        return color, gradients_o


    # Depth
    def errorondepth(self, deform_code, rays_o, rays_d, rays_s, mask, alpha_ratio=0., iter_step=0):
        pts = rays_o + rays_s
        pts_canonical = self.deform_network(deform_code, pts, alpha_ratio)
        ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio)
        if iter_step % self.report_freq == 0:
            pts_back = self.deform_network.inverse(deform_code, pts_canonical, alpha_ratio)
        sdf = self.sdf_network(pts_canonical, ambient_coord, alpha_ratio)[:, :1]
        # Deform, gradients in observation space
        def gradient_obs(deform_network=None, ambient_network=None, sdf_network=None, deform_code=None, x=None, alpha_ratio=None):
            x.requires_grad_(True)
            x_c = deform_network(deform_code, x, alpha_ratio)
            amb_coord = ambient_network(deform_code, x, alpha_ratio)
            y = sdf_network.sdf(x_c, amb_coord, alpha_ratio)
            
            # gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o =  torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            return gradient_o

        gradient_o = gradient_obs(self.deform_network, self.ambient_network, self.sdf_network,
                                    deform_code, pts, alpha_ratio)
        true_cos = (rays_d * gradient_o).sum(-1, keepdim=True)
        relu_cos = F.relu(true_cos)
        pts = pts.detach()
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True)
        # Denoise. not use: out of mask or sphere
        inside_masksphere = (pts_norm < 1.0).to(self.dtype) * mask # inside_sphere * mask
        sdf = inside_masksphere * sdf
        inside_masksphere_sum = inside_masksphere.sum() + 1e-5
        sdf_error = F.l1_loss(sdf, torch.zeros_like(sdf), reduction='sum') / inside_masksphere_sum
        angle_error = F.l1_loss(relu_cos, torch.zeros_like(relu_cos), reduction='sum') / inside_masksphere_sum

        if iter_step % self.report_freq == 0:
            print('Invertibility evaluation: ', torch.abs((pts_back-pts)*inside_masksphere).max().data.item())

        return sdf_error, angle_error, inside_masksphere


    def extract_canonical_geometry(self, bound_min, bound_max, resolution, threshold=0.0, alpha_ratio=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts, alpha_ratio))

    
    def extract_observation_geometry(self, deform_code, bound_min, bound_max, resolution, threshold=0.0, alpha_ratio=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(self.deform_network(deform_code, pts,
                                                            alpha_ratio), self.ambient_network(deform_code, pts,
                                                            alpha_ratio), alpha_ratio))


class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples


    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf


    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None] # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts)
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio) # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).to(self.dtype).detach()
        relax_inside_sphere = (pts_norm < 1.2).to(self.dtype).detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }


    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }


    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
