import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from hydra.utils import instantiate
from core.networks.npc import NPC
from core.networks.embedding import Optcodes
from core.networks.anerf import merge_encodings

from core.positional_enc import PositionalEncoding
from core.utils.skeleton_utils import Skeleton
from einops import rearrange

from omegaconf import DictConfig
from typing import Mapping, Any, List, Optional

class NPCFPS(NPC):

    def init_embedder(
        self,
        *args,
        view_posi_enc: DictConfig,
        voxel_posi_enc: DictConfig,
        **kwargs,
    ):
        # TODO: overwrite all of these, don't need voxel posi
        super(NPC, self).init_embedder(
            *args,
            view_posi_enc=view_posi_enc, 
            voxel_posi_enc=voxel_posi_enc, 
            **kwargs
        )
        pts_feat_dims = self.pts_config.feat_config.n_out
        pose_feat_dims = self.deform_config.n_pose_feat
        self.voxel_posi_enc = instantiate(voxel_posi_enc, input_dims=pts_feat_dims)
        self.input_ch = self.voxel_posi_enc.dims + pose_feat_dims

        if self.add_film:
            self.input_ch = self.input_ch + 32
        
        view_posi_enc_inputs = 3
        self.view_posi_enc = instantiate(view_posi_enc, input_dims=view_posi_enc_inputs, dist_inputs=True)
        self.input_ch_view = self.view_posi_enc.dims

        if self.use_global_view:
            self.global_view_posi_enc = PositionalEncoding(3, num_freqs=4)
            self.input_ch_view += self.global_view_posi_enc.dims
        
    def encode_pts(
        self, 
        inputs: Mapping[str, Any], 
        pc_info: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded_coarse: Optional[Mapping[str, torch.Tensor]] = None, 
        is_pc: bool = False
    ):

        q_w = inputs['pts']
        N_rays, N_samples = q_w.shape[:2]
        N_joints = len(self.rigid_idxs)
        N_unique = inputs['N_unique']
        rays_per_pose = N_rays // N_unique
        unique_idxs = torch.arange(N_rays) // rays_per_pose
        unique_idxs = unique_idxs.reshape(N_rays, 1).expand(-1, N_samples)
        unique_idxs = unique_idxs.reshape(-1)

        # get pts_t (3d points in local space)
        encoded_pts = self.pts_embedder(**inputs, rigid_idxs=self.rigid_idxs)
        encoded_pose = self.pose_embedder(**inputs)

        if self.training and self.constraint_pts > 0:
            encoded_pose['pose'].requires_grad = True
        pose_pe = self.pose_posi_enc(encoded_pose['pose'])[0]
        N_unique = inputs['N_unique']

        inputs.update(
            q_w=q_w,
            pose_pe=pose_pe,
        )

        encoded_q = self.pc.query_feature(
            inputs,
            pc_info,
            is_pc=is_pc,
            need_hessian=self.training and self.full_sdf,
        )

        f_p = encoded_q['f_p']
        f_theta = encoded_q['f_theta']

        f_p = self.voxel_posi_enc(f_p, weights=encoded_q['a_sum'])[0]
        density_inputs = torch.cat([f_p, f_theta], dim=-1)

        f_v = self.view_embedder(
            **inputs, 
            rigid_idxs=self.rigid_idxs,
            refs=encoded_pts['pts_t'],
        )['d']

        f_v = rearrange(f_v, 'r s () c -> (r s) c')[encoded_q['valid_idxs']]
        encoded_pts.update(**encoded_q, f_v=f_v)

        return density_inputs, encoded_pts

    def get_pc_constraints(
        self,
        inputs: Mapping[str, Any],
        pc_info: Mapping[str, torch.Tensor],
        eik_noise: float = 0.03,
    ):

        p_w = pc_info['p_w']
        # remove redundant info -> keep only the unique poses
        skip = len(inputs['skts']) // inputs['N_unique']
        skts = inputs['skts'][::skip]
        bones = inputs['bones'][::skip]
        kp3d = inputs['kp3d'][::skip]
        rays_o = inputs['rays_o'][::skip]
        rays_d = inputs['rays_d'][::skip]

        N_graphs, N_pts = p_w.shape[:2]
        constraint_idxs = np.stack([sorted(np.random.choice(
            N_pts,
            self.constraint_pts,
            replace=False,
        )) for _ in range(N_graphs)])
        constraint_idxs = constraint_idxs + np.arange(N_graphs)[:, None] * N_pts
        constraint_idxs = torch.tensor(constraint_idxs).long()

        # note: after indexing, the shape is (N_graphs, constraint_pts, 3)
        p_cts = rearrange(p_w, 'g p d -> (g p) d')[constraint_idxs]

        encode_inputs = {
            'pts': p_cts, 
            'rays_o': rays_o,
            'rays_d': rays_d,
            'skts': skts,
            'kp3d': kp3d,
            'bones': bones,
            'N_unique': N_graphs,
        }

        density_inputs, encoded = self.encode_pts(encode_inputs, pc_info=pc_info, is_pc=True)

        # note: sigma is sdf
        sigma, pc_feat = self.inference_sigma(density_inputs)
        sigma = rearrange(sigma, '(g p) d-> g p d', g=N_graphs)

        # eikonal / surface gradient
        noise = (torch.rand_like(p_cts) * 2. - 1.) * eik_noise
        p_ncts = p_cts + noise
        encode_inputs.update(pts=p_ncts)

        density_inputs, encoded = self.encode_pts(encode_inputs, pc_info=pc_info, is_pc=True)
        # note: sigma is sdf at perturbed locations
        sigma_n, pc_feat = self.inference_sigma(density_inputs)
        sigma_n = rearrange(sigma_n, '(g p) d-> g p d', g=N_graphs)

        pc_grad = torch.autograd.grad(
            outputs=sigma_n,
            inputs=[p_ncts],
            grad_outputs=torch.ones_like(sigma_n),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        ret = {
            'pc_sigma': sigma,
            'pc_grad': pc_grad,
        }

        return ret