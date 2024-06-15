import torch
import torch.nn as nn
import torch.nn.functional as F

from core.networks.pts import NPCPointClouds
from core.utils.skeleton_utils import rot6d_to_rotmat
from typing import Mapping, Optional
from einops import rearrange

class AnisotropicPointClouds(NPCPointClouds):

    def __init__(
        self,
        *args,
        **wargs,
    ):
        super().__init__(*args, **wargs)

    def init_pts_info(self, *args, **kwargs):
        super().init_pts_info(*args, **kwargs)

        pts_per_volume = self.pts_per_volume
        N_joints = self.N_joints
        init_pts_beta = self.init_pts_beta

        init_beta = (torch.tensor(init_pts_beta).exp() - 1.).log()
        pts_beta = init_beta * torch.ones(N_joints, pts_per_volume, 3)
        # update the already initialized parameters
        self.pts_beta.data = pts_beta
        self.pts_beta.requires_grad = True

        rot = torch.tensor([[[1, 0, 0, 1, 0, 0]]]).float().repeat(N_joints, pts_per_volume, 1)
        self.register_parameter(
            'pts_rot',
            nn.Parameter(rot, requires_grad=True)
        )

    def knn_search(
        self,
        p_w: torch.Tensor,
        p_bs: torch.Tensor,
        q_w: torch.Tensor,
        q_bs: torch.Tensor,
        pose_idxs: torch.Tensor,
        pc_info: Mapping[str, torch.Tensor],
        **kwargs,
    ):
        N_pts = q_w.shape[0]
        N_graphs, N_joints, N_pts_v = p_w.shape[:3]
        knn_vols = self.knn_vols

        # hierachical: first find k-closest volume
        d_vols = q_bs.detach().norm(dim=-1)
        if self.block_irrel:
            # find the closest volume
            closest_vol_idxs = d_vols.argmin(dim=-1)
            vol_hop_masks = self.hop_masks[closest_vol_idxs]

            # set distance to irrelevant volumes to be large
            d_vols = d_vols * vol_hop_masks + (1 - vol_hop_masks) * 1e6
        k_vol_idxs = d_vols.topk(dim=-1, k=knn_vols, largest=False)[1]

        # get the corresponding points from the closest volumes
        cand_idxs = pose_idxs[..., None] * N_joints + k_vol_idxs
        p_cand = rearrange(p_w, 'g j p d -> (g j) p d')[cand_idxs]
        p_cand = rearrange(p_cand, 'b k p d -> b (k p) d')

        # now look for the closet points from the candidcate
        closest_idx = (p_cand - q_w[:, None]).pow(2.).sum(dim=-1).argmin(dim=-1)
            

        # now, turn the closest index back to the actual index to the point clouds
        # Step 1. turn these indices back to corresponding volume indices 
        vol_idxs = k_vol_idxs.flatten()[closest_idx.div(N_pts_v, rounding_mode='trunc') + knn_vols * torch.arange(N_pts)]

        # Step 2. going from volume indices to actual point indices
        closest_pts_idxs = (vol_idxs * N_pts_v + closest_idx % N_pts_v)

        # get the cached neighbors
        knn_idxs = rearrange(self.nb_idxs, 'j p k -> (j p) k')[closest_pts_idxs]

        # now, find the mapping to each of the "posed points"
        # note: each pose has (N_joints * N_pts_v) entry, 
        # and knn_idxs is in [0, N_joints * N_pts_v - 1]
        posed_knn_idxs = pose_idxs[:, None] * N_joints * N_pts_v + knn_idxs

        p_nb = rearrange(p_w, 'g j p d -> (g j p) d')[posed_knn_idxs]
        knn_d_vec = (p_nb - q_w[:, None])

        # this is for query points to find the right knn volume
        # -> each point has N_joints entry
        knn_vol_idxs = torch.arange(N_pts)[:, None] * N_joints + knn_idxs.div(N_pts_v, rounding_mode='trunc')
        return {
            'knn_idxs': knn_idxs,
            'knn_d_vec': knn_d_vec,
            'posed_knn_idxs': posed_knn_idxs,
            'knn_vol_idxs': knn_vol_idxs,
        }

    def compute_feature(
        self,
        p_b: torch.Tensor,
        p_bs: torch.Tensor,
        q_b: torch.Tensor,
        q_bs: torch.Tensor,
        vd: torch.Tensor,
        knn_info: Mapping[str, torch.Tensor],
        pose_idxs: torch.Tensor,
        pc_info: Mapping[str, torch.Tensor],
        vw: Optional[torch.Tensor] = None,
        is_pc: bool = False,
        need_hessian: bool = False,
        **kwargs,
    ):
        N_pts = q_bs.shape[0]
        N_graphs, N_joints, N_pts_v = p_bs.shape[:3]

        r = pc_info['r']
        knn_idxs = knn_info['knn_idxs']
        posed_knn_idxs = knn_info['posed_knn_idxs']
        knn_vol_idxs = knn_info['knn_vol_idxs']
        d_vec = knn_info['knn_d_vec']

        # need feature -> 
        # (f_p, f_s), f_theta, f_d, f_v

        # (f_p, f_s)
        f_p_s = self.pts_feat(self.p_j, need_hessian=need_hessian)
        f_p_s = rearrange(f_p_s, 'j p d -> (j p) d')[knn_idxs]

        # f_theta
        f_theta = pc_info['f_theta']
        f_theta = rearrange(f_theta, 'g j p d -> (g j p) d')[posed_knn_idxs]

        # f_d
        c_q = self.bone_feat(rearrange(q_bs, 'q j d -> j q d'), need_hessian=need_hessian)
        c_q = rearrange(c_q, 'j q d -> (q j) d')[knn_vol_idxs]

        c_p = self.bone_feat(rearrange(p_bs, 'g j p d -> j (g p) d'))
        c_p = rearrange(c_p, 'j (g p) d -> (g j p) d', g=N_graphs)[posed_knn_idxs]
        f_d = c_p - c_q


        # get f_v
        if self.use_global_view:
            assert vw is not None
            vw_idxs = knn_vol_idxs // N_joints
            vd = vw[vw_idxs]
            r = pc_info['r_w']
        else:
            vd = rearrange(vd, 'q j d -> (q j) d')[knn_vol_idxs]
        r = rearrange(r, 'g j p d -> (g j p) d')[posed_knn_idxs]
        f_v = (vd * r).sum(dim=-1, keepdim=True)

        q_b = rearrange(q_b, 'q j d -> (q j) d')[knn_vol_idxs]
        p_b = rearrange(p_b, 'g j p d -> (g j p) d')[posed_knn_idxs]
        f_r = ((q_b - p_b) * r).sum(dim=-1, keepdim=True)

        # get beta for RBF
        T_lbs = pc_info['T_lbs'][..., :3, :3]#.expand(-1, -1, N_pts_v, -1, -1)

        # inverse map to the points' coordinate
        T_lbs_T = rearrange(T_lbs, 'g j p a b -> g j p b a')
        R = self.get_R()[None] @ T_lbs_T
        beta_inv = self.get_pts_inv_beta()[None]
        R_T = rearrange(R, 'g j p a b -> g j p b a')

        mid_term = R_T @ beta_inv @ R
        mid_term = rearrange(mid_term, 'g j p a b -> (g j p) a b')[posed_knn_idxs]
        d_sqr = (d_vec[..., None, :] @ mid_term @ d_vec[..., None])[..., 0]

        a = torch.exp(-d_sqr)
        if is_pc and self.dropout > 0:
            mask = torch.ones_like(a)
            flip = torch.rand_like(a[:, :1]) > self.dropout
            mask[:, :1, :] = flip.float() # masked out the closest point
            a = a * mask


        a_sum = a.sum(dim=1)
        a_norm = a / a_sum[..., None].clamp(min=1e-6)

        # separate them for now as PE is only on f_p_s
        f_p_s = (a * f_p_s).sum(dim=1)
        f_theta = (a * f_theta).sum(dim=1)
        f_d = (a_norm * f_d).sum(dim=1) 
        f_v = (a_norm * f_v).sum(dim=1)
        f_r = (a_norm * f_r).sum(dim=1)


        if is_pc:
            # detach because 2nd order grid sampling backward is too slow 
            f_d = f_d.detach()

        return {
            'f_p_s': f_p_s,
            'f_theta': f_theta,
            'f_d': f_d,
            'f_v': f_v,
            'f_r': f_r,
            'a': a,
            'a_sum': a_sum,
        }
    
    def get_pts_beta(self):
        return self.pts_beta.abs()
    
    def get_pts_inv_beta(self):
        beta = 1. / (super().get_pts_beta() + 1e-8)
        return torch.eye(3)[None, None] * beta[..., None]
    
    def get_R(self):
        return rot6d_to_rotmat(self.pts_rot)