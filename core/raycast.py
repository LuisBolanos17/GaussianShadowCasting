from core.utils.ray_utils import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from core.utils.ray_utils import *
from core.utils.skeleton_utils import get_bone_align_transforms, Skeleton
from typing import Optional, Union, List, Mapping, Any, Callable


class RayCast(nn.Module):
    '''
    Base raycasting module
    '''

    def __init__(
        self,
        near: float = 0.,
        far: float = 100.,
        N_samples: int = 48,
        N_importance: int = 32,
        g_axes: Optional[Union[List, np.ndarray]]=None,
        nf_enlarge: float = 0.0,
        nf_enlarge_ratio: float = 0.25, 
        **kwargs,
    ):
        '''
        Parameters
        ----------
        near: float, near plane for the ray, use only as default value when ray is not in cylinder
        far: float, far plane for the ray, also used only when the ray is not within the cynlider
        '''
        super(RayCast, self).__init__()
        self.near = near
        self.far = far
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.g_axes = g_axes # ground-plane orientation
        self.nf_enlarge = nf_enlarge
        self.nf_enlarge_ratio = nf_enlarge_ratio
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(
        self, 
        batch: Mapping[str, Any], 
        N_samples: Optional[int] = None, 
        pts: Optional[torch.Tensor] = None, 
        weights: Optional[torch.Tensor] = None, 
        z_vals: Optional[torch.Tensor] = None, 
        importance: bool = False, 
        **kwargs
    ):

        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        cyls = batch['cyls']
        skts = batch['skts']

        N_rays = len(rays_o)

        near, far = self.get_near_far(
            rays_o,
            rays_d,
            cyls,
            near=self.near,
            far=self.far,
            skts=skts,
            **kwargs,
        )

        sample_info = {}
        if pts is None:
            if N_samples is None:
                N_samples = self.N_samples
            pts, z_vals = self.sample_pts(rays_o, rays_d, near, far, N_rays, N_samples)
        elif importance:
            if N_samples is None:
                N_samples = self.N_importance
            pts, z_vals, z_samples, sorted_idxs = self.sample_pts_is(
                rays_o, 
                rays_d, 
                z_vals,
                weights,
                N_samples,
                is_only=True,
            )
            # noise_std = 0.01
            # noise = torch.randn_like(pts) * noise_std
            # pts = pts + noise
            sample_info['sorted_idxs'] = sorted_idxs
            sample_info['z_vals_is'] = z_samples
        else:
            NotImplementedError(f'pts input is not None, but not in importance sampling mode!')
        sample_info['pts'] = pts
        sample_info['z_vals'] = z_vals
        sample_info['near'] = near
        sample_info['far'] = far

        return sample_info

    def get_near_far(self, rays_o: torch.Tensor, rays_d: torch.Tensor, cyls: torch.Tensor, near: float = 0., far: float = 100., **kwargs):
        new_near, new_far = get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far, g_axes=self.g_axes)

        if self.nf_enlarge > 0.0 and self.training:
            enlarge_idxs = torch.randperm(len(rays_d))[:int(len(new_near) * self.nf_enlarge)]
            dist = (new_far - new_near) * self.nf_enlarge_ratio
            new_near[enlarge_idxs] = (new_near[enlarge_idxs] - dist[enlarge_idxs]).clamp(min=0.0)
            new_far[enlarge_idxs] = (new_far[enlarge_idxs] + dist[enlarge_idxs]).clamp(min=0.0)

        return new_near, new_far
    
    def sample_pts(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        near: float, 
        far: float, 
        N_rays: int, 
        N_samples: int,
        perturb: float = 0.0, 
        lindisp: bool = False, 
        pytest: bool = False, 
        ray_noise_std: float = 0.0
    ):

        z_vals = sample_from_lineseg(
            near, 
            far, 
            N_rays, 
            N_samples,
            perturb, 
            lindisp, 
            pytest=pytest
        )

        rays_d_noise = rays_d[...,None,:].expand(-1,z_vals.shape[1],-1)
        noise = (torch.rand_like(rays_d_noise) - 0.5) * 0.00002
        rays_d_noise = rays_d_noise + noise
        F.normalize(rays_d_noise, dim=2)

        # range of points should be bounded within 2pi.
        # so the lowest frequency component (sin(2^0 * p)) don't wrap around within the bound
        pts = rays_o[...,None,:] + rays_d_noise * z_vals[...,:,None] # [N_rays, N_samples, 3]

        if ray_noise_std > 0.:
            pts = pts + torch.randn_like(pts) * ray_noise_std

        return pts, z_vals

    def sample_pts_is(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        z_vals: torch.Tensor, 
        weights: torch.Tensor, 
        N_importance: int,
        det: bool = True, 
        pytest: bool = False, 
        is_only: bool = False, 
        ray_noise_std: float = 0.0
    ):

        z_vals, z_samples, sorted_idxs = isample_from_lineseg(z_vals, weights, N_importance, det=det,
                                                              pytest=pytest, is_only=is_only)

        rays_d_noise = rays_d[...,None,:].expand(-1,z_samples.shape[1],-1)
        noise = (torch.rand_like(rays_d_noise) - 0.5) * 0.00002
        rays_d_noise = rays_d_noise + noise
        F.normalize(rays_d_noise, dim=2)
        pts_is = rays_o[...,None,:] + rays_d_noise * z_samples[...,:,None] # [N_rays, N_samples + N_importance, 3]

        if ray_noise_std> 0.:
            pts_is = pts_is + torch.randn_like(pts_is) * ray_noise_std


        return pts_is, z_vals, z_samples, sorted_idxs

class VolRayCast(RayCast):
    """ Only sample within volumes
    """

    def __init__(
        self, 
        *args, 
        vol_scale_fn: Optional[Callable] = None,
        rest_pose: Optional[torch.Tensor] = None,
        rest_heads: Optional[np.ndarray] = None,
        skel_type: Optional[Skeleton] = None,
        rigid_idxs: Optional[torch.Tensor] = None,
        bound: float = 1.1,
        eps: float = 1e-5,
        **kwargs
    ):
        '''
        vol_scale_fn: function, to inquire the current volume size
        '''

        super(VolRayCast, self).__init__(*args, **kwargs)
        assert vol_scale_fn is not None
        assert rest_pose is not None
        assert skel_type is not None
        self.vol_scale_fn = vol_scale_fn
        self.rest_pose = rest_pose
        self.skel_type = skel_type
        self.rigid_idxs = rigid_idxs
        self.bound = bound
        self.eps = eps


        transforms, _ = get_bone_align_transforms(rest_pose.cpu().numpy(), skel_type,
                                                  rest_heads=rest_heads)
        self.register_buffer('transforms', transforms)

    @torch.no_grad()
    def get_near_far(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        cyls: torch.Tensor, 
        skts: torch.Tensor, 
        near: float = 0.0, 
        far: float = 100.0, 
        **kwargs
    ):
        near, far = get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far, g_axes=self.g_axes)

        B, J = skts.shape[:2]
        align_transforms = self.transforms.to(rays_o.device)
        vol_scale = self.vol_scale_fn().reshape(1, -1, 3).detach().to(rays_o.device)
        if self.rigid_idxs is not None:
            skts = skts[:, self.rigid_idxs]
            align_transforms = align_transforms[self.rigid_idxs]
            vol_scale = vol_scale[:, self.rigid_idxs]

            J = len(self.rigid_idxs)

        # transform both rays origin and direction to the per-joint coordinate
        rays_ot = (skts[..., :3, :3] @ rays_o.reshape(B, 1, 3, 1) + skts[..., :3, -1:]).reshape(B, J, 3)
        rays_dt = (skts[..., :3, :3] @ rays_d.reshape(B, 1, 3, 1)).reshape(B, J, 3)

        rays_ot = ((align_transforms[..., :3, :3] @ rays_ot[..., None]) + \
                    align_transforms[..., :3, -1:]).reshape(B, J, 3)
        rays_dt = (align_transforms[..., :3, :3] @ rays_dt[..., None]).reshape(B, J, 3)
        
        # scale the rays by the learned volume scale and find the intersections with the volumes
        bound_range = self.bound

        p_valid, v_valid, p_intervals = get_ray_box_intersections(
            rays_ot / vol_scale, 
            F.normalize(rays_dt / vol_scale, dim=-1), 
            bound_range=bound_range, # intentionally makes the bound a bit larger
            eps=self.eps,
        )

        # now undo the scale so we can calculate the near far in the original space
        vol_scale = vol_scale.expand(B, J, 3)
        p_intervals = p_intervals * vol_scale[v_valid][..., None, :]

        norm_rays = rays_dt[v_valid].norm(dim=-1)
        # extremely tiny negative values

        # find the step size (near / far)
        # t * norm_ray + ray_o = p -> t =  (p - ray_o) / norm_rays
        # -> distance is the norm 
        steps = (p_intervals - rays_ot[v_valid][..., None, :]).norm(dim=-1) / norm_rays[..., None]

        # extract near/far for each volume
        v_near = 100000 * torch.ones(B, J)
        v_far = -100000 * torch.ones(B, J)

        # find the near/far
        v_near[v_valid] = steps.min(dim=-1).values
        v_far[v_valid] = steps.max(dim=-1).values

        # pick the closest/farthest points as the near/far planes
        v_near = v_near.min(dim=-1).values
        v_far = v_far.max(dim=-1).values

        # merge the values back to the cylinder near far
        ray_valid = (v_valid.sum(-1) > 0)
        new_near = near.clone()
        new_far = far.clone()

        new_near[ray_valid, 0] = v_near[ray_valid]
        new_far[ray_valid, 0] = v_far[ray_valid]

        if self.nf_enlarge > 0.0 and self.training:
            enlarge_idxs = torch.randperm(len(rays_d))[:int(len(new_near) * self.nf_enlarge)]
            dist = (new_far - new_near) * self.nf_enlarge_ratio
            new_near[enlarge_idxs] = (new_near[enlarge_idxs] - dist[enlarge_idxs]).clamp(min=0.0)
            new_far[enlarge_idxs] = (new_far[enlarge_idxs] + dist[enlarge_idxs]).clamp(min=0.0)

        return new_near, new_far
