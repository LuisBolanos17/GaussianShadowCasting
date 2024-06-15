import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from core.utils.skeleton_utils import *
from typing import Optional, Any


class CamCal(nn.Module):

    def __init__(
        self,
        n_cams: int = 4,
        identity_cam: int = 0,
        load_path: Optional[str] = None,
        stop_opt: bool = False,
        opt_T: bool = False,
    ):
        super().__init__()
        self.n_cams = n_cams
        self.identity_cam = identity_cam
        self.load_path = load_path
        self.stop_opt = stop_opt
        self.opt_T = opt_T

        R = torch.eye(3)[None] 
        Rvec = rot_to_rot6d(R).expand(n_cams, -1)

        if self.load_path is not None:
            device = Rvec.device
            Rvec = torch.load(load_path, map_location=device)

        self.register_parameter('Rvec', nn.Parameter(Rvec.clone(), requires_grad=not self.stop_opt))

        if self.opt_T:
            T = torch.zeros(3)[None]
            T = T.expand(n_cams, -1)
            self.register_parameter('T', nn.Parameter(T.clone(), requires_grad=not self.stop_opt))
    
    def forward(
        self,
        batch: dict,
        z_vals: torch.Tensor,
        **kwargs,
    ):
        if 'real_cam_idx' not in batch:
            cam_idxs = torch.zeros(z_vals.shape[0]).long() + self.identity_cam
        else:
            cam_idxs = batch['real_cam_idx']
        

        Rvec = self.Rvec
        if self.stop_opt:
            Rvec = Rvec.detach()
 
        R = rot6d_to_rotmat(Rvec)
        masks = (cam_idxs == self.identity_cam).float()
        masks = masks.reshape(-1, 1, 1)
        identity = torch.eye(3)[None]

        R = R[cam_idxs] * (1 - masks) + identity * masks

        rays_o = batch['rays_o']
        rays_d = batch['rays_d']

        # adjust the ray
        rays_d_cal = (rays_d[:, None] @ R)
        if self.opt_T:
            T = self.T
            masks = masks.reshape(-1, 1) 
            identity = torch.zeros(3)[None]
            T = T[cam_idxs] * (1 - masks) + identity * masks
            if self.stop_opt:
                T = T.detach()
            rays_o_cal = rays_o[:, None] + T[:, None]
        else:
            rays_o_cal = rays_o[:, None]
        pts_cal = rays_d_cal * z_vals[..., None] + rays_o_cal

        batch.update(
            pts=pts_cal,
            rays_d=rays_d_cal[:, 0],
        )

        return batch

class ColorCal(nn.Module):

    def __init__(
        self,
        n_cams: int = 4,
        identity_cam: int = 0,
        load_path: Optional[str] = None,
        stop_opt: bool = False,
    ):
        super().__init__()
        self.n_cams = n_cams
        self.identity_cam = identity_cam
        self.load_path = load_path
        self.stop_opt = stop_opt

        cal = torch.tensor([[1., 1., 1., 0., 0., 0.]]).expand(n_cams, -1)

        if self.load_path is not None:
            device = cal.device
            cal = torch.load(load_path, map_location=device)

        self.register_parameter('cal', nn.Parameter(cal.clone(), requires_grad=not self.stop_opt))
    
    def forward(
        self,
        batch: dict,
        rgb_map: torch.Tensor,
        **kwargs,
    ):
        if batch is None or 'real_cam_idx' not in batch:
            cam_idxs = torch.zeros(rgb_map.shape[0]).long() + self.identity_cam
        else:
            cam_idxs = batch['real_cam_idx']

        cal = self.cal
        if self.load_path is not None:
            cal = cal.detach()
 
        masks = (cam_idxs == self.identity_cam).float()
        masks = masks.reshape(-1, 1)
        identity = torch.tensor([[1., 1., 1., 0., 0., 0.]])

        cal = cal[cam_idxs] * (1 - masks) + identity * masks
        rgb_cal = rgb_map * cal[:, :3] + cal[:, 3:] 

        return rgb_cal
