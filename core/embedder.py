import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from core.utils.skeleton_utils import (
    axisang_to_rot6d,
    get_bone_align_transforms,
)

from typing import Optional


def transform_batch_pts(pts: torch.Tensor, skt: torch.Tensor):
    '''
    Transform points/vectors from world space to local space

    Parameters
    ----------
    pts: Tensor (..., 3) in world space
    skt: Tensor (..., N_joints, 4, 4) world-to-local transformation
    '''

    N_rays, N_samples = pts.shape[:2]
    NJ = skt.shape[-3]

    if skt.shape[0] < pts.shape[0]:
        skt = skt.expand(pts.shape[0], *skt.shape[1:])

    # make it from (N_rays, N_samples, 4) to (N_rays, NJ, 4, N_samples)
    pts = torch.cat([pts, torch.ones(*pts.shape[:-1], 1)], dim=-1)
    pts = pts.view(N_rays, -1, N_samples, 4).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    # MM: (N_rays, NJ, 4, 4) x (N_rays, NJ, 4, N_samples) -> (N_rays, NJ, 4, N_samples)
    # permute back to (N_rays, N_samples, NJ, 4)
    mm = (skt @ pts).permute(0, 3, 1, 2).contiguous()

    return mm[..., :-1] # don't need the homogeneous part


def transform_batch_rays(rays_d: torch.Tensor, skt: torch.Tensor):
    '''
    Transform direction vectors from world space to local space

    Parameters
    ----------
    rays_d: Tensor (N_rays, 3) direction in world space
    skt: Tensor (..., N_joints, 4, 4) world-to-local transformation
    '''

    # apply only the rotational part
    assert rays_d.dim() == 2
    N_rays = len(rays_d)
    N_samples = 1
    NJ = skt.shape[-3]
    rot = skt[..., :3, :3]

    if rot.shape[0] < rays_d.shape[0]:
        rot = rot.expand(rays_d.shape[0], *rot.shape[1:])
    rays_d = rays_d.view(N_rays, -1, N_samples, 3).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    mm = (rot @ rays_d).permute(0, 3, 1, 2).contiguous()

    return mm


class BaseEmbedder(nn.Module):

    def __init__(self, N_joints=24, N_dims=3, skel_type=None, **kwargs):
        super().__init__()
        self.N_joints = N_joints
        self.N_dims = N_dims
        self.skel_type = skel_type

    @property
    def dims(self):
        return self.N_joints * self.N_dims

    @property
    def encoder_name(self):
        return self.__class__.__name__

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.encoder_name}(N_joints={self.N_joints}, N_dims={self.N_dims})'


class WorldToLocalEmbedder(BaseEmbedder):

    def forward(self, pts: torch.Tensor, skts: torch.Tensor, **kwargs):
        pts_t = transform_batch_pts(pts, skts)
        return {'pts_t': pts_t}


class BoneAlignEmbedder(WorldToLocalEmbedder):

    def __init__(self, rest_pose: np.ndarray, rest_heads: Optional[np.ndarray] = None, *args, **kwargs):
        super(BoneAlignEmbedder, self).__init__(*args, **kwargs)
        self.rest_pose = rest_pose
        self.rest_heads = rest_heads

        transforms, child_idxs = get_bone_align_transforms(
            rest_pose, 
            self.skel_type, 
            rest_heads=rest_heads,
        )
        self.child_idxs = np.array(child_idxs)
        self.register_buffer('transforms', transforms)

    def forward(self, pts: torch.Tensor, skts: torch.Tensor, rigid_idxs: Optional[torch.Tensor] = None, **kwargs):
        if rigid_idxs is not None:
            skts = skts[..., rigid_idxs, :, :]
        pts_jt = transform_batch_pts(pts, skts)
        pts_t = self.align_pts(pts_jt, rigid_idxs=rigid_idxs)
        return {'pts_t': pts_t, 'pts_jt': pts_jt}
    
    def align_pts(
        self, 
        pts: torch.Tensor, 
        align_transforms: Optional[torch.Tensor] = None, 
        rigid_idxs: Optional[torch.Tensor] = None, 
        is_dir: bool = False
    ):
        if align_transforms is None:
            align_transforms = self.get_bone_align_T(rigid_idxs)
        elif rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]
        
        pts_t = (align_transforms[..., :3, :3] @ pts[..., None]).squeeze(-1)
        if not is_dir:
            pts_t = pts_t + align_transforms[..., :3, -1]
        return pts_t
    
    def unalign_pts(
        self, 
        pts_t: torch.Tensor, 
        align_transforms: Optional[torch.Tensor] = None, 
        rigid_idxs: Optional[torch.Tensor] = None, 
        is_dir: bool = False
    ):
        if align_transforms is None:
            align_transforms = self.get_bone_align_T(rigid_idxs)
        elif rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]

        if not is_dir:
            pts_t = pts_t - align_transforms[..., :3, -1]
        pts = align_transforms[..., :3, :3].transpose(-1, -2) @ pts_t[..., None]

        return pts.squeeze(-1)
    
    def get_bone_align_T(self, rigid_idxs: Optional[torch.Tensor] = None):
        if rigid_idxs is not None:
            return self.transforms[rigid_idxs]
        return self.transforms
    

class SkeletonRelativeEmbedder(WorldToLocalEmbedder):
    '''
    Skeleton-relative embedding used in A-NeRF
    '''
    
    @property
    def dims(self):
        return (self.N_joints, self.N_joints * self.N_dims)
    
    def forward(self, pts: torch.Tensor, skts: torch.Tensor, **kwargs):

        ret = super(SkeletonRelativeEmbedder, self).forward(pts, skts, **kwargs)
        pts_t = ret['pts_t']
        v = torch.norm(pts_t, dim=-1)
        r = ret['pts_t'] / v[..., None]
        return {'v': v, 'r': r, **ret}


class Pose6DEmbedder(WorldToLocalEmbedder):

    @property
    def dims(self):
        return 6

    def forward(
        self, 
        pts: Optional[torch.Tensor] = None, 
        skts: Optional[torch.Tensor] = None, 
        bones: Optional[torch.Tensor] = None, 
        N_unique: int = 1, 
        **kwargs
    ):
        assert bones is not None
        skip = bones.shape[0] // N_unique
        unique_bones = bones[::skip]
        return {
            'pose': axisang_to_rot6d(unique_bones),
        }


class WorldToRootViewEmbedder(BaseEmbedder):

    @property
    def dims(self):
        return self.N_dims

    def forward(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        skts: torch.Tensor, 
        refs: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        root = self.skel_type.root_id
        # Assume root index is at 0
        rays_dt = transform_batch_rays(rays_d, skts[:, root:root+1])
        if refs is not None:
            # expand so that the ray embedding has sample dimension
            N_expand = refs.shape[1]
            rays_dt = rays_dt.expand(-1, N_expand, -1, -1)
        return {'d': rays_dt}


class WorldViewEmbedder(BaseEmbedder):
    @property
    def dims(self):
        return self.N_dims

    def forward(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        skts: torch.Tensor, 
        refs: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        # Assume root index is at 0
        rays_dt = rays_d
        if refs is not None:
            # expand so that the ray embedding has sample dimension
            rays_dt = rays_dt.reshape(-1, 1, 1, 3)
            N_expand = refs.shape[1]
            rays_dt = rays_dt.expand(-1, N_expand, -1, -1)
        return {'d': rays_dt}


class SkeletonRelativeViewEmbedder(BaseEmbedder):
    '''
    Skeleton-relative embedding used in A-NeRF
    '''
    def __init__(self, N_joints, N_dims, **kwargs):
        super(SkeletonRelativeViewEmbedder, self).__init__()
        self.N_joints = N_joints
        self.N_dims = N_dims
    
    @property
    def dims(self):
        return self.N_joints * self.N_dims

    def forward(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        skts: torch.Tensor, 
        refs: Optional[torch.Tensor] = None, 
        **kwargs
    ):

        rays_dt =  transform_batch_rays(rays_d, skts)

        if refs is not None:
            # expand so that the ray embedding has sample dimension
            N_expand = refs.shape[1]
            rays_dt = rays_dt.expand(-1, N_expand, -1, -1)
        return {'d': rays_dt}
    

class BoneAlignViewEmbedder(SkeletonRelativeViewEmbedder):

    def __init__(self, rest_pose, skel_type, rest_heads=None, *args, **kwargs):
        super(BoneAlignViewEmbedder, self).__init__(*args, **kwargs)
        self.skel_type = skel_type
        self.rest_pose = rest_pose

        transforms, child_idxs = get_bone_align_transforms(rest_pose, self.skel_type, 
                                                           rest_heads=rest_heads)
        self.child_idxs = np.array(child_idxs)
        self.register_buffer('transforms', transforms)

    @property
    def dims(self):
        return self.N_dims

    def forward(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        skts: torch.Tensor, 
        refs: torch.Tensor, 
        valid_idxs: Optional[torch.Tensor] = None, 
        rigid_idxs: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        if rigid_idxs is not None:
            skts = skts[..., rigid_idxs, :, :]
        N_pose, N_joints = skts.shape[:2]
        
        rays_dt = transform_batch_rays(rays_d, skts)
        rays_dt = (skts[..., :3, :3] @ rays_d[:, None, :, None])[..., 0]
        rays_dt = self.align_pts(rays_dt, self.transforms, rigid_idxs=rigid_idxs)
        if refs is not None:
            # expand so that the ray embedding has sample dimension
            N_expand = refs.shape[1]
            rays_dt = rays_dt[:, None]
            rays_dt = rays_dt.expand(-1, N_expand, -1, -1)
        rays_dt = rays_dt.reshape(-1, N_joints, self.dims)
        if valid_idxs is not None:
            rays_dt = rays_dt[valid_idxs]
        return {'d': rays_dt}
    
    def align_pts(
        self, 
        rays_d: torch.Tensor, 
        align_transforms: Optional[torch.Tensor] = None, 
        rigid_idxs: Optional[torch.Tensor] = None,
    ):
        if align_transforms is None:
            align_transforms = self.transforms
        if rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]
        
        pts_t = (align_transforms[..., :3, :3] @ rays_d[..., None]).squeeze(-1)
        return pts_t

    def unalign_pts(
        self, 
        pts_t: torch.Tensor, 
        align_transforms: Optional[torch.Tensor] = None, 
        rigid_idxs: Optional[torch.Tensor] = None, 
        is_dir: bool = False
    ):
        if align_transforms is None:
            align_transforms = self.transforms
        if rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]
        if not is_dir:
            pts_t = pts_t - align_transforms[..., :3, -1]
        pts = align_transforms[..., :3, :3].transpose(-1, -2) @ pts_t[..., None]
        return pts.squeeze(-1)
