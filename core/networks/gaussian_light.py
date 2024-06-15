import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

from core.utils.skeleton_utils import rot6d_to_rotmat, rot_to_rot6d

from typing import Any, Union, List, Optional
from einops import rearrange

# TODO: rename to something more appropriate

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-7

RANGE = np.array([2*np.pi,-np.pi,np.pi,0])


def erf(x):
    # save the sign of x
    sign = (x > 0)*2-1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*torch.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)


def gaussian3d_pdf_loop(gaussian, x):
    mean = gaussian[:3][None,...]
    D = torch.diag(1/(gaussian[3:6]**2))
    #R = rc.rotation_6d_to_matrix(gaussian[6:12])
    R = rot6d_to_rotmat(gaussian[6:12])
    density = torch.abs(gaussian[12])

    output = torch.clip(-0.5*((x-mean) @ (R.T @ D @ R) @ (x-mean).T), -100, 0)
    output_exp = density*torch.exp(output)
    return torch.diagonal(output_exp)

def gaussian3d_pdf(G_mu, G_std, G_Rvec, G_c, x):
    N_joints, N_g = G_mu.shape[:2]
    R = rot6d_to_rotmat(G_Rvec)
    G_density = G_c
    
    #diff_sqr = torch.einsum('jgkl,bjgl->bjgk', R, x[:, :, None] - G_mu[None]).pow(2.)
    #D = 1 / G_std.pow(2.)

    # TODO: debugging
    diff = x[:, :, None] - G_mu[None]
    eye = rearrange(torch.eye(3), 'i j -> 1 1 i j').expand(N_joints, N_g, -1, -1)
    D = eye * (1 / G_std.pow(2.))[..., None]
    R_T = rearrange(R, 'j g k l -> j g l k')
    S =  R_T @ D @ R
    nom = diff[..., None, :] @ S @ diff[..., None]
    nom = rearrange(nom, 'b j g 1 1 -> b j g')

    #density = G_density[None] * torch.exp((-0.5 * (diff_sqr * D).sum(dim=-1)).clamp(min=-100, max=0))
    density = G_density[None] * torch.exp((-0.5 * nom).clamp(min=-100, max=0))
    return density


def gaussian1d_pdf(mean, sigma, x):
    return torch.exp(-(0.5*(x-mean)**2)/sigma**2)


def gaussian1d_cdf(mean, sigma, density, x):
    sqrt_two = 1.41421356237 
    return density*0.5*(1+erf((x-mean)/(sigma*sqrt_two)))


def gaussian1d_quantile(mean, sigma, density, p):
    sqrt_pi = 1.7724538509055159
    return mean + sigma*sqrt_pi*torch.atanh(2*density*p-1)


def equirectangular_range_to_direction(u, v, range = np.array([2*np.pi,-np.pi,np.pi,0])):
    phi = range[0] * u + range[1]
    theta = range[2] * v + range[3]
    sin_theta = np.sin(theta)
    return np.array([sin_theta * np.cos(phi), sin_theta*np.sin(phi), np.cos(theta)])


def direction_to_equirectangular_range(dir, range):
    u = (np.arctan2(dir[1], dir[0]) - range[1]) / range[0]
    v = (np.arccos(dir[2]) - range[3]) / range[2]
    return u, v


class GaussianVisibility(nn.Module):
    def __init__(
        self, 
        num_bones: int, 
        num_gaussians_per_bone: int,
        init_means: Optional[np.ndarray] = None, # num_bones x 3 | mean to initialize gaussians to in case we want to initialize them at the heads, tails, or anywhere in between
        bone_align_T: Optional[np.ndarray] = None,
        init_Gs: Optional[Union[float, np.ndarray]] = None,
        base_scale: float = 0.001,
        filter_density: str = 'soft',
        stop_opt: Optional[int] = None,
    ):

        super().__init__()

        self.num_bones = num_bones
        self.num_gaussians_per_bone = num_gaussians_per_bone
        self.filter_density = filter_density
        self.stop_opt = stop_opt

        if init_Gs is None:
            init_Gs = torch.Tensor([[[
                0.00, 0.00, 0.00, # mean
                0.06, 0.06, 0.06, # axis-aligned variance D
                #1/0.06, 1/0.06, 1/0.06, # axis-aligned variance D
                1.00, 0.00, 0.00, 1.00, 0.00, 0.00, # 6d rotation (identity initialization)
                10,
            ]]]).repeat(num_bones, num_gaussians_per_bone, 1)
            if init_means is not None:
                init_Gs[:,:,0:3] = torch.Tensor(init_means[:, None, :])
            init_Gs[:, :, 0:3] += torch.randn_like(init_Gs[:, :, 0:3]) * 0.015
        else:
            init_Gs = torch.Tensor(init_Gs)

        if bone_align_T is not None:
            Rt = torch.eye(4)[None, None].expand(num_bones, num_gaussians_per_bone, -1, -1).clone()
            Rt[:, :, :3, -1] = init_Gs[:, :, :3]
            Rt[:, :, :3, :3] = rot6d_to_rotmat(init_Gs[:, :, 6:12])

            unalign_T = bone_align_T.inverse()[:, None]
            Rt = unalign_T @ Rt 
            Rvec = rot_to_rot6d(Rt[:, :, :3, :3])
            t = Rt[:, :, :3, -1]

            init_Gs[..., :3] = t
            init_Gs[..., 6:12] = Rvec
            init_Gs = init_Gs.contiguous()
 

        self.register_parameter(
            'Gs', 
            nn.Parameter(init_Gs, requires_grad=True)
        )
        self.register_buffer(
            'base_scale',
            torch.tensor(base_scale, dtype=torch.float32)
        )
    
    def get_Gs(self, global_iter: Optional[int] = None):
        G_mu = self.Gs[..., 0:3]
        G_std = self.Gs[..., 3:6].abs() + self.base_scale
        G_Rvec = self.Gs[..., 6:12]
        G_c = self.Gs[..., 12].abs()

        if (global_iter is not None) and (self.stop_opt is not None) and (global_iter >= self.stop_opt):
            G_mu = G_mu.detach()
            G_std = G_std.detach()
            G_Rvec = G_Rvec.detach()
            G_c = G_c.detach()

        return G_mu, G_std, G_Rvec, G_c

    def forward(
        self,
        w2ls: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        N_unique: int = 1,
    ):


        means, sigmas, densities = self.get_1D_gaussians(
            w2ls,
            rays_o,
            rays_d,
            N_unique,
        )
        """ 
        means_, sigmas_, densities_ = self.get_1D_gaussians_(
            w2ls,
            rays_o,
            rays_d,
            N_unique,
        )

        import pdb; pdb.set_trace()
        print
        """

        cdf = gaussian1d_cdf(means, sigmas, densities, 10.0) 
        shadow_map = torch.exp(-torch.sum(cdf, dim=0))
        if torch.isnan(shadow_map).any():
            print('shadow map computation resulted in nan!')
            import pdb; pdb.set_trace()
            print()
        return shadow_map

    def sample_gaussian_preds(
        self,
        density: torch.Tensor,
        pts: torch.Tensor,
        w2ls: torch.Tensor,
        N_samples: int = 256,
        valid_idxs: Optional[torch.Tensor] = None,
        global_iter: Optional[int] = None,
    ):
        B, S, C = pts.shape
        pts = rearrange(pts, 'b s c -> (b s) c')
        density = rearrange(density, 'b s c -> (b s) c')
        #if valid_idxs is not None:
        #    pts = pts[valid_idxs]
        #    density = density[valid_idxs]
        #else:
        valid_idxs = torch.arange(pts.shape[0])
        # Gaussians
        rand_idxs = np.random.choice(pts.shape[0], N_samples, False)

        pts_sample = pts[rand_idxs]
        sample_idxs = valid_idxs[rand_idxs]
        pose_idxs = sample_idxs // S
        w2ls_sample = w2ls[pose_idxs]

        w2ls_sample_ = w2ls[:, None].expand(-1, S, -1, -1, -1).reshape(-1, self.num_bones, 4, 4)[valid_idxs][rand_idxs]
        assert (w2ls_sample == w2ls_sample_).all()

        gaussian_pred = self.sample_pdf(
            w2ls_sample, 
            pts_sample,
            global_iter=global_iter,
        )
        # tanh: press the density value to [-1, 1]
        # NOTE: in reality the negative part has 0 gradient
        # and using tanh means that the Gaussian densities does not need to match
        # huge density values
        gaussian_pred = torch.tanh(rearrange(gaussian_pred, 'p b g -> p (b g)').sum(dim=-1, keepdim=True))
        density_target = torch.tanh(density[rand_idxs].detach())

        G_mu, G_std, _, _ = self.get_Gs(global_iter)
        if (global_iter is not None) and (self.stop_opt is not None) and (global_iter >= self.stop_opt):
            gassian_pred = gaussian_pred.detach()

        return {
            'gaussian_density_diff': (gaussian_pred - density_target),
            'gaussian_sigma': G_std,
            'gaussian_mean': G_mu,
        }
 
    def get_1D_gaussians(
        self, 
        w2ls: torch.Tensor, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor,
        N_unique: int = 1,
        no_grad: bool = True,
    ): 
        # batch_size x num_bones x 4 x 4 | N x 3 | N x 3

        #skip = w2ls.shape[0] // N_unique
        # only keep the unique transformation since some rays may come from the same image

        ray_origins = torch.cat((rays_o, torch.ones((rays_o.shape[0], 1))), 1)
        ray_directions = torch.cat((rays_d, torch.zeros((rays_d.shape[0], 1))), 1)

        # TODO: FIXME: n is actually r!
        # (w2ls @ ray_origins[:, None, :, None])[..., :3, 0]
        o = torch.einsum('rbij,rj->rbi', w2ls, ray_origins)[..., :3]
        o = rearrange(o, 'r b c -> r b () c')

        # (w2ls @ ray_directions[:, None, :, None])[..., :3, 0]
        n = torch.einsum('rbij,rj->rbi', w2ls, ray_directions)[..., :3]
        n = rearrange(n, 'r b c -> r b () c ()')
        nT = rearrange(n, 'r b g c 1 -> r b g 1 c')

        G_mu, G_std, G_Rvec, G_c = self.get_Gs()
        G_mu = G_mu[..., 0:3, None].detach()
        S = (1 / G_std.detach())[..., None] * torch.eye(3)[None, None]
        G_R = rot6d_to_rotmat(G_Rvec).detach()
        G_c = G_c[None].detach()

        SR = S @ G_R
        Sn1 = rearrange(SR, 'b g i j -> b g j i') @ SR

        mu_d_o = (G_mu[None] - o[..., None])
        nTSn1 = nT @ Sn1 
        sigma_bar_sqr = 1 / (nTSn1 @ n)
        mu_bar = (nTSn1 @ mu_d_o) * sigma_bar_sqr 

        mu_d_o_T_Sn1_mu_d_o = rearrange(mu_d_o, 'r b g i j -> r b g j i') @ Sn1[None] @ mu_d_o
        densities = G_c * torch.exp(
            -0.5 * (
                mu_d_o_T_Sn1_mu_d_o - mu_bar.pow(2.) / sigma_bar_sqr
            )[..., 0, 0]
        )
        means = rearrange(mu_bar, 'r b g 1 1 -> (b g) r')
        sigmas = rearrange(sigma_bar_sqr.pow(0.5), 'r b g 1 1 -> (b g) r')
        densities = rearrange(densities, 'r b g -> (b g) r')

        # Filtering out density behind the 2nd ray origin
        if self.filter_density == 'hard':
            densities = torch.where(means >= 0, densities, torch.zeros_like(densities))
        elif self.filter_density == 'soft':
            weights = 1/(1+torch.exp(-(100*means.detach()-1))) # detached otherwise cuda kernel error :/
            densities = densities * weights

        return means, sigmas, densities

    def get_1D_gaussians_(
        self, 
        w2ls: torch.Tensor, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor,
        N_unique: int = 1,
        no_grad: bool = True,
    ): 
        # batch_size x num_bones x 4 x 4 | N x 3 | N x 3

        #skip = w2ls.shape[0] // N_unique
        # only keep the unique transformation since some rays may come from the same image

        ray_origins = torch.cat((rays_o, torch.ones((rays_o.shape[0], 1))), 1)
        ray_directions = torch.cat((rays_d, torch.zeros((rays_d.shape[0], 1))), 1)

        # TODO: FIXME: n is actually r!
        # (w2ls @ ray_origins[:, None, :, None])[..., :3, 0]
        o = torch.einsum('rbij,rj->rbi', w2ls, ray_origins)[..., :3]
        o = rearrange(o, 'r b c -> r b () c')

        # (w2ls @ ray_directions[:, None, :, None])[..., :3, 0]
        n = torch.einsum('rbij,rj->rbi', w2ls, ray_directions)[..., :3]

        G_mu, G_std, G_Rvec, G_c = self.get_Gs()
        G_mu = G_mu[None].detach()
        G_std = G_std[None].abs().detach()
        G_R = rot6d_to_rotmat(G_Rvec).detach()
        G_c = G_c[None].abs().detach()

        # (n[:, :, None, None, :] @ G_R[None])[..., 0, :]
        #n_aligned_sphere = F.normalize(torch.einsum('rbi,bgij->rbgj', n, G_R) / G_std, dim=-1)
        n_aligned_sphere = F.normalize(torch.einsum('bgij,rbj->rbgi', G_R, n) / G_std, dim=-1)

        # (((o - G_mu)[..., None, :] @ G_R))[..., 0, :] / G_std + G_mu
        #o_aligned_sphere = torch.einsum('rbgi,bgij->rbgj', o - G_mu, G_R) / G_std + G_mu
        o_aligned_sphere = torch.einsum('bgij,rbgj->rbgi', G_R, o - G_mu) / G_std + G_mu

        # diagonal of A @ B.T is the same as sum(A * B, dim=-1)
        # -> at least that's what I wrote down on paper
        means = torch.norm(
            G_std * n_aligned_sphere * \
            torch.clamp_min(
                ((G_mu - o_aligned_sphere) * n_aligned_sphere).sum(dim=-1, keepdim=True),
                0.0
            ),
            dim=-1
        )

        sigmas = torch.norm(G_std * n_aligned_sphere, dim=-1)

        densities = G_c * torch.exp(
            -0.5 * (
                (G_mu - o_aligned_sphere).pow(2.).sum(dim=-1) - means.pow(2.) / sigmas.pow(2.)
            )
        )

        means = rearrange(means, 'r b g -> (b g) r')
        sigmas = rearrange(sigmas, 'r b g -> (b g) r')
        densities = rearrange(densities, 'r b g -> (b g) r')

        return means, sigmas, densities
    

    def sample_pdf(self, w2ls, query, global_iter: Optional[int] = None): # w2ls: num_bones,4,4 | query: batch, 3
        # queries the mixture of gaussians by first transforming each query to a bone's local space using the world to local transformation w2ls
        # then sampling the associated gaussian
        # and summing across all gaussians

        # w2ls (b, j, 4, 4)
        local_query = torch.einsum('sjkl,sl->sjk', w2ls[..., :3, :3], query) + w2ls[..., :3, -1]
        G_mu, G_std, G_Rvec, G_c = self.get_Gs(global_iter)
        sample_density = gaussian3d_pdf(
            G_mu, 
            G_std,
            G_Rvec,
            G_c,
            local_query
        )

        return sample_density 

