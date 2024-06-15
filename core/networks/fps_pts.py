import torch
import torch.nn as nn
import torch.nn.functional as F
from core.networks.pts import NPCPointClouds

from hydra.utils import instantiate

from core.utils.skeleton_utils import SMPLSkeleton, HARESkeleton, WOLFSkeleton, MixamoSkeleton
from core.utils.skeleton_utils import (
    find_n_hops_joint_neighbors,
    farthest_point_sampling,
)
from copy import deepcopy
from einops import rearrange
from typing import Callable, Optional, Mapping, Any, List

from omegaconf import DictConfig


class UniformPointClouds(NPCPointClouds):

    def __init__(
        self,
        *args,
        deform_net: Callable,
        bone_centers: torch.Tensor,
        n_hops: int = 2,
        knn_pts: int = 8,
        total_pts: int = 4000,
        pts_file: Optional[str] = None,
        init_pts_beta: float = 0.0005,
        feat_config: Optional[DictConfig] = None,
        bone_config: Optional[DictConfig] = None,
        skel_profile: Optional[dict] = None,
        dropout: float = 0.8,
        use_global_view: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self)

        assert pts_file is not None
        assert feat_config is not None
        assert bone_config is not None
        assert skel_profile is not None

        self.skel_profile = deepcopy(skel_profile)

        self.deform_net = deform_net
        self.bone_centers = bone_centers
        self.rest_pose = torch.tensor(self.skel_profile['rest_pose'])
        self.rigid_idxs = self.skel_profile['rigid_idxs'].copy()
        self.n_hops = n_hops
        self.knn_pts = knn_pts
        self.total_pts = total_pts

        self.init_pts_beta = init_pts_beta
        self.skel_type = self.skel_profile['skel_type']
        self.dropout = dropout
        self.use_global_view = use_global_view

        self.init_pts_info(pts_file, feat_config, bone_config)

    def init_pts_info(self, pts_file: str, feat_config: DictConfig, bone_config: DictConfig):

        init_pts_beta = self.init_pts_beta
        knn_pts = self.knn_pts

        device = torch.randn(1).device # hack to get the device
        pts_data = torch.load(pts_file, map_location=device)

        """
        canon_pts = torch.stack(pts_data['canon_pts'])
        N_joints, N_pts, _ = canon_pts.shape
        canon_pts = rearrange(canon_pts, 'j p c -> (j p) c')
        """

        canon_pts = pts_data['canon_pts']
        labels = torch.cat([torch.ones(len(pts)) * i for i, pts in enumerate(canon_pts)])
        N_joints = len(canon_pts)
        canon_pts = torch.cat(canon_pts)

        fps_idxs = farthest_point_sampling(canon_pts, self.total_pts).sort().values
        # joint labels
        #labels = fps_idxs // N_pts

        labels = labels[fps_idxs].long()
        p_c = canon_pts[fps_idxs]

        hop_masks, nb_joints = find_n_hops_joint_neighbors(
            self.skel_profile,
            n_hops=self.n_hops,
        )

        lbs_weights, lbs_masks = self.get_initial_lbs_weights(
            p_c,
            self.bone_centers,
            nb_joints,
            labels,
        )

        self.pts_feat = instantiate(feat_config, n_pts=self.total_pts, n_vols=1)
        nb_idxs, nb_diffs = self.get_pts_neighbors(p_c, knn_pts)

        init_beta = (torch.tensor(init_pts_beta).exp() - 1.).log()
        pts_beta = init_beta * torch.ones(self.total_pts, 1)

        self.register_parameter(
            'p_c',
            nn.Parameter(p_c.clone(), requires_grad=True),
        )

        self.register_parameter(
            'lbs_weights',
            nn.Parameter(lbs_weights.clone(), requires_grad=True),
        )

        self.register_parameter(
            'pts_beta',
            nn.Parameter(pts_beta, requires_grad=True),
        )

        # TODO: hop mask and lbs mask are the same
        self.register_buffer(
            'lbs_masks',
            lbs_masks,
        )

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

    def get_pts_neighbors(self, p_c: torch.Tensor, knn_pts: int):
        
        dist = torch.cdist(p_c, p_c, compute_mode='donot_use_mm_for_euclid_dist')
        _, nb_idxs = dist.topk(k=knn_pts, dim=-1, largest=False)
        nb_diffs = p_c[:, None] - p_c[nb_idxs]
        
        return nb_idxs, nb_diffs

    def get_initial_lbs_weights(
        self,
        p_c: torch.Tensor,
        bone_centers: torch.Tensor,
        nb_joints: List[List],
        labels: torch.Tensor,
    ):
        N_joints = bone_centers.shape[1]

        lbs_masks = torch.zeros(N_joints, N_joints)
        pts_lbs_masks = torch.zeros(self.total_pts, N_joints)
        lbs_weights = torch.zeros(self.total_pts, N_joints)

        # the 3rd dimension is the distance to the j bone
        dists_to_bones = (p_c[:, None] - bone_centers).norm(dim=-1)

        for i in range(len(self.rigid_idxs)):
            # TODO: just block left/right for each points?
            for nb_joint in nb_joints[i]:
                if nb_joint in self.rigid_idxs:
                    j = list(self.rigid_idxs).index(nb_joint)
                    lbs_masks[i, j] = 1.

            pts_idxs = torch.where(labels == i)[0]
            dist_to_bones = dists_to_bones[pts_idxs]
            lbs_weights[pts_idxs] = (1 / (dist_to_bones + 1e-10).pow(0.5))
            lbs_weights[pts_idxs] = lbs_weights[pts_idxs] * lbs_masks[i:i+1]
            pts_lbs_masks[pts_idxs, :] = lbs_masks[i]

        return lbs_weights, pts_lbs_masks

    def pts_to_canon(self, p_j: torch.Tensor):
        raise NotImplementedError("This shouldn't be called because p_j does not exist!")
    
    def query_feature(
        self, 
        inputs: Mapping[str, Any],
        pc_info: Optional[Mapping[str, torch.Tensor]] = None,
        is_pc: bool = False,
        need_hessian: bool = False,
    ):

        if pc_info is None or is_pc:
            pc_info = self.get_deformed_pc_info(inputs, is_pc=is_pc)

        q_w = inputs['q_w']
        p_w = pc_info['p_w']

        N_unique = inputs['N_unique']
        N_rays_per_pose = q_w.shape[0] // N_unique

        # TODO: threshold hyperparameter
        knn_info = self.knn_search(p_w, q_w, filter=not is_pc, threshold=0.05)

        encoded_q = self.compute_feature(
            p_w=p_w,
            q_w=q_w,
            knn_info=knn_info,
            pc_info=pc_info,
            is_pc=is_pc,
            need_hessian=need_hessian,
        )

        encoded_q.update(
            pc_info=pc_info,
            valid_idxs=knn_info['valid_idxs']
        )

        return encoded_q
    
    def knn_search(
        self,
        p_w: torch.Tensor,
        q_w: torch.Tensor,
        filter: bool = True,
        threshold: float = 0.05,
    ):
        N_graphs, N_pts = p_w.shape[:2]
        N_rays, N_samples = q_w.shape[:2]

        # collapse all samples from a pose into a single dimension
        q_w = rearrange(q_w, '(g r) s c -> g (r s) c', g=N_graphs)
        # TODO: also removes too far away samples?
        # TODO: do we disable mm?

        # detach to save memory... we will get the distance later
        dists = torch.cdist(q_w.detach(), p_w.detach())
        closest_idxs = dists.argmin(dim=-1)

        # get the cached neightbors
        pose_idxs = torch.arange(N_graphs)
        knn_idxs = self.nb_idxs[closest_idxs]
        posed_knn_idxs = pose_idxs[:, None, None] * N_pts + knn_idxs

        # compute distance
        p_nb = rearrange(p_w, 'g p c -> (g p) c')[posed_knn_idxs]
        knn_d_sqr = (p_nb - q_w[:, :, None]).pow(2.).sum(dim=-1, keepdim=True)

        if filter:
            knn_d = knn_d_sqr.detach().sqrt()
            valid = ((knn_d < threshold).sum(dim=-2) > 0)
            valid = rearrange(valid, 'g (r s) 1 -> (g r s)', s=N_samples)
            valid_idxs = valid.nonzero()[..., 0]
        else:
            valid_idxs = torch.arange(N_rays * N_samples)
        
        # keep only valid points
        knn_idxs = rearrange(knn_idxs, 'g (r s) k -> (g r s) k', s=N_samples)[valid_idxs]
        posed_knn_idxs = rearrange(posed_knn_idxs, 'g (r s) k -> (g r s) k', s=N_samples)[valid_idxs]
        knn_d_sqr = rearrange(knn_d_sqr, 'g (r s) k 1 -> (g r s) k 1', s=N_samples)[valid_idxs]

        return {
            'knn_idxs': knn_idxs,
            'posed_knn_idxs': posed_knn_idxs,
            'knn_d_sqr': knn_d_sqr, 
            'valid_idxs': valid_idxs,
        }

    def compute_feature(
        self,
        p_w: torch.Tensor,
        q_w: torch.Tensor,
        knn_info: Mapping[str, torch.Tensor],
        pc_info: Mapping[str, torch.Tensor],
        is_pc: bool = False,
        need_hessian: bool = False,
    ):
        knn_idxs = knn_info['knn_idxs']
        posed_knn_idxs = knn_info['posed_knn_idxs']
        d_sqr = knn_info['knn_d_sqr']

        # f_p, f_theta, (TODO: do we want f_v, f_r?)

        f_p = self.pts_feat(p_w, need_hessian=need_hessian)[knn_idxs]
        f_theta = pc_info['f_theta']
        f_theta = rearrange(f_theta, 'g p f -> (g p) f')[posed_knn_idxs]

        beta = self.get_pts_beta()[knn_idxs]

        a = torch.exp(-d_sqr / beta)
        if is_pc and self.dropout > 0:
            mask = torch.ones_like(a)
            flip = torch.rand_like(a[:, :1]) > self.dropout
            mask[:, :1, :] = flip.float() # masked out the closest point
            a = a * mask

        a_sum = a.sum(dim=1)
        a_norm = a / a_sum[..., None].clamp(min=1e-6)

        f_p = (a * f_p).sum(dim=1)
        f_theta = (a * f_theta).sum(dim=1)

        return {
            'f_p': f_p,
            'f_theta': f_theta,
            'a': a,
            'a_sum': a_sum,
            'a_norm': a_norm,
        }

    def get_deformed_pc_info(
        self, 
        inputs: Mapping[str, Any], 
        is_pc: bool = False
    ):

        rigid_idxs = self.rigid_idxs
        rest_pose = self.rest_pose[:, rigid_idxs]
        N_joints = len(rigid_idxs)

        # get skts
        N_unique = inputs.get('N_unique', 1)
        skip = inputs['skts'].shape[0] // N_unique
        skts = inputs['skts'][::skip, rigid_idxs]

        # get local to world (joint to world)
        l2ws = skts.inverse()

        # get rest pose to world (rest pose to world)
        # -> include rest pose to local transformation
        T = l2ws[..., :3, :3] @ -rest_pose.reshape(1, N_joints, 3, 1)
        r2ws = l2ws.clone()
        r2ws[..., :3, -1:] += T

        deform_info = self.deform_net(
            p_c=self.p_c,
            r2ws=r2ws,
            pose=inputs['pose_pe'],
            lbs_weights=self.lbs_weights,
            lbs_masks=self.lbs_masks,
        )

        # TODO: figure out what else do you need

        deform_info.update(
            p_c=self.p_c,
        )

        return deform_info
