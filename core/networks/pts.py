import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import instantiate
from core.networks.misc import ParallelLinear
from core.utils.skeleton_utils import SMPLSkeleton, HARESkeleton, WOLFSkeleton, MixamoSkeleton
from core.utils.skeleton_utils import (
    find_n_hops_joint_neighbors,
    farthest_point_sampling,
    rot6d_to_rotmat,
)
from copy import deepcopy
from einops import rearrange
from typing import Callable, Optional, Mapping, Any, List
#import pytorch3d.ops as p3do

from omegaconf import DictConfig


class NPCPointClouds(nn.Module):

    def __init__(
        self,
        deform_net: Callable,
        bone_centers: torch.Tensor,
        pts_embedder: Callable,
        n_hops: int = 2,
        knn_vols: int = 3,
        knn_pts: int = 8,
        pts_per_volume: int = 200,
        pts_file: Optional[str] = None,
        init_pts_beta: float = 0.0005,
        init_pts_k: int = 12,
        feat_config: Optional[DictConfig] = None,
        bone_config: Optional[DictConfig] = None,
        skel_profile: Optional[dict] = None,
        tar_init: bool = False,
        dropout: float = 0.8,
        use_global_view: bool = False,
        block_irrel: bool = False,
        opt_beta: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert pts_file is not None
        assert feat_config is not None
        assert bone_config is not None
        assert skel_profile is not None

        self.skel_profile = deepcopy(skel_profile)

        self.deform_net = deform_net
        self.pts_embedder = pts_embedder
        self.rest_pose = torch.tensor(self.skel_profile['rest_pose'])
        self.bone_centers = bone_centers
        self.rigid_idxs = self.skel_profile['rigid_idxs'].copy()
        self.n_hops = n_hops
        self.knn_vols = knn_vols
        self.knn_pts = knn_pts
        self.pts_per_volume = pts_per_volume
        self.init_pts_k = init_pts_k
        self.init_pts_beta = init_pts_beta
        self.N_joints = len(self.rigid_idxs)
        self.skel_type = self.skel_profile['skel_type']
        self.dropout = dropout
        self.tar_init = tar_init
        self.use_global_view = use_global_view
        self.block_irrel = block_irrel
        self.opt_beta = opt_beta

        self.init_pts_info(pts_file, feat_config, bone_config)
    
    def init_pts_info(self, pts_file: str, feat_config: DictConfig, bone_config: DictConfig):
        """ 
        """
        init_pts_beta = self.init_pts_beta
        init_pts_k = self.init_pts_k
        pts_per_volume = self.pts_per_volume
        knn_pts = self.knn_pts
        N_joints = self.N_joints

        device = torch.randn(1).device # hack to get the device

        # read pre-extracted point data
        pts_data = torch.load(pts_file, map_location=device)
        
        p_j = pts_data['anchor_pts']
        # point in canonical space (rest pose)
        p_c = pts_data['canon_pts']
        axis_scale = pts_data['axis_scale']

        # set the deform net's axis scale
        self.deform_net.axis_scale.data.copy_(axis_scale)

        # init points
        selected_dict = self.select_init_pts(p_j, p_c, pts_per_volume)
        p_j = selected_dict['p_j']
        p_c = selected_dict['p_c']

        # init LBS weights
        hop_masks, nb_joints = find_n_hops_joint_neighbors(
            self.skel_profile,
            n_hops=self.n_hops,
        )
        lbs_weights, lbs_masks = self.get_initial_lbs_weights(
            p_c, 
            self.bone_centers, 
            nb_joints,
            pts_data=pts_data,
            pts_idxs=selected_dict['pts_idxs'],
        )
        assert (hop_masks == lbs_masks[:, 0, :]).all()

        # init per-point features
        self.pts_feat = instantiate(
            feat_config,
            n_vols=N_joints,
        )
        
        # init bone-relative features
        self.bone_feat = instantiate(
            bone_config,
            n_vols=N_joints,
        )

        # set per-point neighbors
        nb_idxs, nb_diffs, _ = self.get_pts_neighbors(p_c, knn_pts)

        # init per-point vars
        # initilize it so that after softplus it gives us the desired values
        init_beta = (torch.tensor(init_pts_beta).exp() - 1.).log()
        pts_beta = init_beta * torch.ones(N_joints, pts_per_volume, 1)

        # register parameters and buffers
        self.register_parameter(
            'p_j',
            nn.Parameter(p_j.clone(), requires_grad=True),
        )

        self.register_parameter(
            'lbs_weights',
            nn.Parameter(lbs_weights.clone(), requires_grad=True),
        )

        # TODO: hop mask and lbs mask are the same
        self.register_buffer(
            'lbs_masks',
            lbs_masks,
        )

        self.register_buffer(
            'hop_masks',
            hop_masks,
        )

        # per-joint point clouds in the canonical space
        self.register_buffer(
            'init_p_c',
            p_c,
        )

        self.register_buffer(
            'nb_idxs',
            nb_idxs,
        )

        self.register_buffer(
            'nb_diffs',
            nb_diffs,
        )

        if self.tar_init:
            print('tar_init')
            target = 0.00005

            #init_beta = (torch.tensor(init_pts_beta).exp() - 1.).log()
            #pts_beta = init_beta * torch.ones(N_joints, pts_per_volume, 1)

            ln_tar = torch.tensor(target).log()
            init_beta_ = -nb_diffs.pow(2.).sum(-1)[..., init_pts_k-1:init_pts_k] / ln_tar
            init_beta_ = (torch.tensor(init_beta_).exp() - 1.).log()

            pts_beta = init_beta_ #_.clamp(min=init_beta)

        self.register_parameter(
            'pts_beta',
            nn.Parameter(pts_beta, requires_grad=self.opt_beta),
        )

    def select_init_pts(self, p_j: torch.Tensor, p_c: torch.Tensor,  pts_per_volume: int = 200):

        selected_p_j, selected_p_c = [], []
        pts_idxs = []
        for j, (p, p_c_) in enumerate(zip(p_j, p_c)):
            assert len(p) == len(p_c_)
            selected_idxs = farthest_point_sampling(
                p,
                pts_per_volume
            )#.sort().values
            selected_p_j.append(p[selected_idxs])
            selected_p_c.append(p_c_[selected_idxs])
            pts_idxs.append(selected_idxs)
        
        return {
            'p_j': torch.stack(selected_p_j), 
            'p_c': torch.stack(selected_p_c),
            'pts_idxs': torch.stack(pts_idxs),
        }
    
    def get_pts_neighbors(self, p_c: torch.Tensor, knn_pts: int):
        
        N_joints = self.N_joints

        # calculate the initial distance from anchor to all rest pose joints
        dist_to_rp = (p_c[..., None, :] - self.rest_pose[None, None]).norm(dim=-1)
        pc_flatten = p_c.reshape(-1, 3)

        nb_idxs, nb_diffs = [], []
        for j in range(N_joints):
            x = p_c[j]
            dist = torch.cdist(x, pc_flatten, compute_mode='donot_use_mm_for_euclid_dist')

            _, nb_idxs_ = dist.topk(k=knn_pts, dim=-1, largest=False)
            nb_pts_diff = x[:, None] - pc_flatten[nb_idxs_]

            nb_idxs.append(nb_idxs_)
            nb_diffs.append(nb_pts_diff)
        
        nb_idxs = torch.stack(nb_idxs)
        nb_diffs = torch.stack(nb_diffs)
        
        return nb_idxs, nb_diffs, dist_to_rp
    
    def get_initial_lbs_weights(
        self, 
        p_c: torch.Tensor, 
        bone_centers: torch.Tensor, 
        nb_joints: List, 
        pts_data: Optional[Mapping[str, Any]] = None, 
        pts_idxs: Optional[torch.Tensor] = None
    ):
        """ 
        p_c: (N_joint, N_pts, 3) point clouds in rest pose / canonical space
        bone_centers: (1, N_joints, 3) bone centers in rest pose / canonical space
        """

        assert len(p_c) == len(self.rigid_idxs)

        # N_joints 
        N_joints = len(p_c)
        ppv = self.pts_per_volume

        lbs_masks = torch.zeros(N_joints, 1, N_joints)
        lbs_weights = torch.zeros(N_joints, ppv, N_joints)

        # make it (N_joints, N_joints, N_pts, 3)
        # the 2nd dimension is the "possible "
        p_c = p_c[:, :, None]
        bone_centers = bone_centers[None]
        # the 3rd dimension is the distance to the j bone
        dists_to_bones = (p_c - bone_centers).norm(dim=-1)

        for i in range(len(self.rigid_idxs)):
            for nb_joint in nb_joints[i]:
                if nb_joint in self.rigid_idxs:
                    j = list(self.rigid_idxs).index(nb_joint)
                    lbs_masks[i, :, j] = 1.
            dist_to_bones = dists_to_bones[i, :, :]
            lbs_weights[i, :, :] = (1 / (dist_to_bones + 1e-10).pow(0.5))
        
        return lbs_weights, lbs_masks
    
    def query_feature(
        self, 
        inputs: Mapping[str, Any],
        pc_info: Optional[Mapping[str, torch.Tensor]] = None,
        is_pc: bool = False,
        need_hessian: bool = False,
    ):
        vol_scale = self.get_axis_scale(rigid_idxs=self.rigid_idxs)
        if pc_info is None or is_pc:
            pc_info = self.get_deformed_pc_info(inputs, is_pc=is_pc)
        
        vw = inputs['vw']
        vd = inputs['vd']
        q_w = inputs['q_w']
        q_b = inputs['q_b']
        q_bs = q_b / vol_scale

        p_w = pc_info['p_w']
        p_b = pc_info['p_b']
        p_bs = p_b / vol_scale[None, :, None]

        # hierarchical k-NN
        pose_idxs = inputs['valid_pose_idxs']
        """
        knn_info = self.ball_query(
            p_w,
            p_bs, 
            q_w,
            q_bs,
            pose_idxs=pose_idxs,
            pc_info=pc_info,
        )
        """ 
        knn_info = self.knn_search(
            p_w,
            p_bs, 
            q_w,
            q_bs,
            pose_idxs=pose_idxs,
            pc_info=pc_info,
        )


        encoded_q = self.compute_feature(
            p_b,
            p_bs,
            q_b,
            q_bs,
            vd,
            vw=vw,
            knn_info=knn_info,
            pose_idxs=pose_idxs,
            pc_info=pc_info,
            is_pc=is_pc,
            need_hessian=need_hessian,
        )
        
        encoded_q.update(pc_info=pc_info)
        return encoded_q
    
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
        d_sqr = knn_info['knn_d_sqr']
        valid_knn_idxs = knn_info.get('valid_knn_idxs', None)

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
        f_d = (c_p - c_q) 

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
        beta = rearrange(self.get_pts_beta(), 'j p d -> (j p) d')[knn_idxs]

        a = torch.exp(-d_sqr / beta)

        if valid_knn_idxs is not None:
            a = torch.where(valid_knn_idxs[..., None], a, 0)
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

    def get_deformed_pc_info(self, inputs: Mapping[str, Any], is_pc: bool = False):
        """ 
        apply pose-dependent deformation to the point clouds
        """

        rigid_idxs = self.rigid_idxs
        rest_pose = self.rest_pose[:, rigid_idxs]
        N_joints = len(rigid_idxs)

        # get skts
        N_unique = inputs.get('N_unique', 1)
        skip = inputs['skts'].shape[0] // N_unique
        skts = inputs['skts'][::skip, rigid_idxs]
        bones = inputs['bones'][::skip]
        t = inputs['real_kp_idx'][::skip] if 'real_kp_idx' in inputs else None

        # get local to world (joint to world)
        l2ws = skts.inverse()

        # get rest pose to world (rest pose to world)
        # -> include rest pose to local transformation
        T = l2ws[..., :3, :3] @ -rest_pose.reshape(1, N_joints, 3, 1)
        r2ws = l2ws.clone()
        r2ws[..., :3, -1:] += T

        p_j = self.p_j
        p_c = self.pts_to_canon(p_j)

        bone_align_T = self.pts_embedder.get_bone_align_T(rigid_idxs=rigid_idxs)

        # w2ajs: world to aligned-joint space
        w2ajs = bone_align_T[None] @ skts
        w2ajs = rearrange(w2ajs, 'g j a b -> g j 1 a b')
        aj2ws = w2ajs.inverse()

        deform_info = self.deform_net(
            p_j,
            p_c,
            r2ws=r2ws,
            aj2ws=aj2ws,
            pose=inputs['pose_pe'],
            lbs_weights=self.lbs_weights,
            lbs_masks=self.lbs_masks,
            bone_align_T=bone_align_T,
            t=t,
            bones=bones,
        )
        deform_info['w2aj'] = w2ajs

        # get deformed point clouds in bone-aligned space
        # -> for querying bone-relative features and compute bone-to-surface-point vector
        p_w = deform_info['p_w'] 
        p_b = (w2ajs[..., :3, :3] @ p_w[..., None] + w2ajs[..., :3, -1:])[..., 0]

        r = self.get_bone_to_surface_vector(p_b)
        r_w = (aj2ws[..., :3, :3] @ r[..., None])[..., 0]

        aj2w = rearrange(aj2ws, 'g j 1 a b -> g j a b')

        deform_info.update(p_b=p_b, r=r, r_w=r_w, aj2w=aj2w)

        return deform_info
    
    def knn_search(
        self,
        p_w: torch.Tensor,
        p_bs: torch.Tensor,
        q_w: torch.Tensor,
        q_bs: torch.Tensor,
        pose_idxs: torch.Tensor,
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
        knn_d_sqr = (p_nb - q_w[:, None]).pow(2.).sum(dim=-1, keepdim=True)

        # this is for query points to find the right knn volume
        # -> each point has N_joints entry
        knn_vol_idxs = torch.arange(N_pts)[:, None] * N_joints + knn_idxs.div(N_pts_v, rounding_mode='trunc')

        return {
            'knn_idxs': knn_idxs,
            'knn_d_sqr': knn_d_sqr,
            'posed_knn_idxs': posed_knn_idxs,
            'knn_vol_idxs': knn_vol_idxs,
        }
        
    def get_axis_scale(self, rigid_idxs: Optional[torch.Tensor] = None):
        axis_scale = self.deform_net.get_axis_scale()
        if rigid_idxs is not None:
            axis_scale = axis_scale[rigid_idxs]
        return axis_scale

    def pts_to_canon(self, p_j: torch.Tensor):
        """ Move points from (aligned) joint spaces to the canonical space
        """
        rigid_idxs = self.rigid_idxs
        rest_pose = self.rest_pose[:, rigid_idxs]
        p_j = rearrange(p_j, 'j p d -> p j d')
        vol_scale = self.get_axis_scale(rigid_idxs=rigid_idxs)[None]
        # remove axis-align transformation
        p_j = self.pts_embedder.unalign_pts(
            p_j * vol_scale,
            rigid_idxs=rigid_idxs,
        )
        p_c = p_j + rest_pose
        p_c = rearrange(p_c, 'p j d -> j p d')
        return p_c
    
    @torch.no_grad()
    def get_bone_to_surface_vector(self, p_b: torch.Tensor):
        """
        TODO: this should be defined elsewhere
        """
        if self.skel_type == SMPLSkeleton:
            ball_vols = [0, 9, 12]
            bone_vols = [i for i in range(len(self.rigid_idxs)) if i not in ball_vols]
        else:
            ball_vols, bone_vols = [], []

            for i, j in enumerate(self.rigid_idxs):
                parent_idx = self.skel_type.joint_trees[j]
                if j == parent_idx:
                    ball_vols.append(i)
                else:
                    bone_vols.append(i)

        r = torch.zeros_like(p_b)
        r[:, ball_vols] += p_b[:, ball_vols] 
        r[:, bone_vols, ..., :2] += p_b[:, bone_vols, ..., :2]
        r = F.normalize(r, dim=-1)

        return r
    
    def get_pts_beta(self):
        return F.softplus(self.pts_beta)
    
    def get_summaries(self, *args, **kwargs):
        return {}

    def ball_query(
        self,
        p_w: torch.Tensor,
        p_bs: torch.Tensor,
        q_w: torch.Tensor,
        q_bs: torch.Tensor,
        pose_idxs: torch.Tensor,
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

        _, idxs, nn = p3do.ball_query(q_w[:, None], p_cand, K=self.knn_pts, radius=0.1) 
        idxs = rearrange(idxs, 'b () k -> b k')
        #d_sqr = rearrange(d_sqr, 'b () k -> b k')

        # This is the expensive part
        vol_idxs = k_vol_idxs.flatten()[idxs.div(N_pts_v, rounding_mode='trunc') + knn_vols * torch.arange(N_pts)[..., None]]
        knn_idxs = (vol_idxs * N_pts_v + idxs % N_pts_v)

        # now, find the mapping to each of the "posed points"
        # note: each pose has (N_joints * N_pts_v) entry, 
        # and knn_idxs is in [0, N_joints * N_pts_v - 1]
        posed_knn_idxs = pose_idxs[:, None] * N_joints * N_pts_v + knn_idxs

        # this is for query points to find the right knn volume
        # -> each point has N_joints entry
        knn_vol_idxs = torch.arange(N_pts)[:, None] * N_joints + knn_idxs.div(N_pts_v, rounding_mode='trunc')
        p_nb = rearrange(p_w, 'g j p d -> (g j p) d')[posed_knn_idxs]
        knn_d_vec = (p_nb - q_w[:, None]).pow(2.)
        knn_d_sqr = knn_d_vec.pow(2.).sum(dim=-1, keepdim=True)

        return {
            'knn_idxs': knn_idxs,
            'knn_d_vec': knn_d_vec,
            'knn_d_sqr': knn_d_sqr,
            'posed_knn_idxs': posed_knn_idxs,
            'knn_vol_idxs': knn_vol_idxs,
            'valid_knn_idxs': (idxs >= 0),
        }

    def get_pts_inv_beta(self):
        beta = 1. / (super().get_pts_beta() + 1e-8)
        return torch.eye(3)[None, None] * beta[..., None]
    
    def get_R(self):
        return rot6d_to_rotmat(self.pts_rot)