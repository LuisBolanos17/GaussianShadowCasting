import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from hydra.utils import instantiate
from core.networks.anerf import ANeRF
from core.networks.misc import fill_valid_tensor
from core.utils.visualization import plot_boxes_mpl
from core.utils.skeleton_utils import boxes_to_pose, create_axis_boxes

from typing import Mapping, Any, List, Optional
from omegaconf import DictConfig


class DANBO(ANeRF):

    def __init__(
        self,
        *args,
        graph_net: Optional[DictConfig] = None,
        voxel_agg_net: Optional[DictConfig] = None,
        voxel_feat: int = 15,
        voxel_res: int = 16,
        agg_type: str = 'sigmoid',
        use_rigid: bool = True,
        rest_heads: Optional[np.ndarray] = None,
        **kwargs,
    ):

        # set DANBO-specific args
        self.voxel_feat = voxel_feat
        self.voxel_res = voxel_res
        self.agg_type = agg_type
        self.use_rigid = use_rigid
        self.rest_heads = rest_heads

        super(DANBO, self).__init__(*args, **kwargs)

        if graph_net is not None:
            # Note: this should only happen for NPC
            self.init_graph_net(graph_net, voxel_agg_net)

        self.rigid_idxs = None
        if self.use_rigid:
            self.N_joints = len(self.skel_type.joint_names)
            self.part_ignore = self.skel_profile['rigid_ignore']
            self.rigid_idxs = np.array([
                i for i in range(self.N_joints) 
                if i not in self.part_ignore
            ])
        
    def init_embedder(
        self, 
        pts_embedder: DictConfig, 
        pts_posi_enc: DictConfig,
        view_embedder: DictConfig, 
        view_posi_enc: DictConfig, 
        pose_embedder: DictConfig,
        pose_posi_enc: DictConfig,
        voxel_posi_enc: DictConfig,
        **kwargs
    ):
        N_joints = len(self.skel_type.joint_names)
        # initialize points transformation
        self.pts_embedder = instantiate(
            pts_embedder, 
            N_joints=N_joints, 
            N_dims=3, 
            skel_type=self.skel_type,
            rest_pose=self.rest_pose.cpu().numpy(),
            rest_heads=self.rest_heads,
        )

        # initialize pose transformation
        self.pose_embedder = instantiate(pose_embedder, N_joints=N_joints, N_dims=3)

        # initialize positional encoding for poses/bones
        self.pose_dims = self.pose_embedder.dims
        self.pose_posi_enc = instantiate(pose_posi_enc, input_dims=self.pose_dims)

        # initialize voxel posi_enc
        self.voxel_posi_enc = instantiate(voxel_posi_enc, input_dims=self.voxel_feat)

        # initialize view transformation
        self.view_embedder = instantiate(
            view_embedder, 
            N_joints=N_joints, 
            N_dims=3, 
            skel_type=self.skel_type,
            rest_pose=self.rest_pose.cpu().numpy(),
            rest_heads=self.rest_heads,
        )

        # initialize positional encoding for views (rays)
        self.view_dims = view_dims= self.view_embedder.dims
        self.view_posi_enc = instantiate(view_posi_enc, input_dims=view_dims, dist_inputs=True)

        self.input_ch = self.voxel_posi_enc.dims
        self.input_ch_graph = self.pose_posi_enc.dims
        self.input_ch_view = self.view_posi_enc.dims
        if self.cat_normal:
            self.input_ch_view += 3

    def init_raycast(
        self,
        raycast: DictConfig,
        **kwargs,
    ):
        self.raycast = instantiate(
            raycast, 
            **kwargs, 
            vol_scale_fn=self.get_axis_scale,
            rest_pose=self.rest_pose,
            rest_heads=self.rest_heads,
            skel_type=self.skel_type,
            _recursive_=False
        )
    
    def init_graph_net(self, graph_net: DictConfig, voxel_agg_net: DictConfig):

        self.graph_net = instantiate(
            graph_net, 
            skel_type=self.skel_type,
            per_node_input=self.input_ch_graph,
            output_ch=None, # determined by voxel_feat and voxel_res
            voxel_feat=self.voxel_feat, 
            voxel_res=self.voxel_res,
            skel_profile=self.skel_profile,
            keep_extra_joint=True,
        )
        
        self.voxel_agg_net = instantiate(
            voxel_agg_net,
            skel_type=self.skel_type,
            per_node_input=self.voxel_feat,
            output_ch=1,
            skel_profile=self.skel_profile,
            keep_extra_joint=True,
        )
    
    def fix_danbo_box_params(self):
        self.graph_net.fix_danbo_box_params()
    
    def encode_pts(self, inputs: Mapping[str, Any], no_skip: bool = False, no_pe: bool = False, no_agg: bool = False):

        N_rays, N_samples = inputs['pts'].shape[:2]
        N_joints = len(self.skel_type.joint_names)

        # get pts_t (3d points in local space)
        encoded_pts = self.pts_embedder(**inputs)

        # Note: this return encoding for "unique bones/poses"
        # (See inputs['N_unique'])
        encoded_pose = self.pose_embedder(**inputs)
        pose_pe = self.pose_posi_enc(encoded_pose['pose'])[0]

        # get graph feature
        h = self.graph_net(pose_pe)
        h_j, invalid, _ = self.graph_net.sample_from_volume(
            h, 
            encoded_pts['pts_t'], 
            need_hessian=self.pred_sdf
        )
        
        h_j = h_j.reshape(N_rays * N_samples, N_joints, self.voxel_feat)

        # we only need to compute valid points, i.e., points that's within volume
        valid = (1 - invalid).reshape(N_rays * N_samples, N_joints)
        valid_idxs = torch.where(valid.sum(-1) > 0)[0] 
        if no_skip:
            valid_idxs = torch.arange(len(valid))
        h_j = h_j[valid_idxs]
        if len(valid_idxs) == 0:
            return None, None
        
        if not no_agg:
            a = self.voxel_agg_net(h_j).reshape(len(valid_idxs), N_joints)
            # see which volumes are invalid
            vol_invalid = invalid.flatten(end_dim=1)[valid_idxs]
            p = self.get_agg_weights(a, vol_invalid)

            h_hat = (p[..., None] * h_j)
            if self.use_rigid:
                h_hat = h_hat[:, self.rigid_idxs]
            h_hat = h_hat.sum(dim=1)
        else:
            h_hat = h_j
            p = a = torch.zeros(len(valid_idxs), N_joints)

        if not no_pe and not no_agg:
            density_inputs = self.voxel_posi_enc(h_hat)[0]
        else:
            density_inputs = h_hat

        a_full = fill_valid_tensor(a, (N_rays, N_samples), valid_idxs=valid_idxs)
        encoded = {}

        encoded.update(
            agg_logit=a_full,
            p=p,
            valid_idxs=valid_idxs,
            vol_invalid=invalid,
            h_hat=h_hat,
            **encoded_pts,
        )

        return density_inputs, encoded
        
    def collect_encoded(self, encoded_pts: Mapping[str, torch.Tensor], encoded_views: Mapping[str, torch.Tensor]):
        ret = super(DANBO, self).collect_encoded(encoded_pts, encoded_views)
        if 'agg_logit' in encoded_pts:
            ret['agg_logit'] = encoded_pts['agg_logit']
        if 'vol_invalid' in encoded_pts:
            ret['vol_invalid'] = encoded_pts['vol_invalid']
        ret['valid_idxs'] = encoded_pts['valid_idxs']

        return ret

    def get_agg_weights(self, a: torch.Tensor, invalid: torch.Tensor, eps: float = 1e-7, mask_invalid: bool = True):
        if self.agg_type == 'softmax':
            return self.softmax(a, invalid, eps=eps)
        if self.agg_type == 'sigmoid':
            return self.sigmoid(a, invalid, eps=eps, mask_invalid=mask_invalid)
        else:
            raise NotImplementedError

    def sigmoid(
        self, 
        logit: torch.Tensor, 
        invalid: torch.Tensor,
        sigmoid_eps: float = 0.001, 
        mask_invalid: bool = True,
        **kwargs
    ):

        #invalid = invalid.flatten(end_dim=-2)
        valid = 1 - invalid
        p = torch.sigmoid(logit) * (1 + 2 * sigmoid_eps) - sigmoid_eps
        if mask_invalid:
            p = p * valid
        return p

    def softmax(self, logit: torch.Tensor, invalid: torch.Tensor, eps: float = 1e-7, temp: float = 1.0, **kwargs):

        logit = logit / temp
        # find the valid part
        valid = 1 - invalid
        # for stability: doesn't change the output as the term will be canceled out
        max_logit = logit.max(dim=-1, keepdim=True)[0]

        # only keep the valid part!
        nominator = torch.exp(logit - max_logit) * valid
        denominator = torch.sum(nominator + eps, dim=-1, keepdim=True)

        return nominator / denominator.clamp(min=eps)
    
    def collect_outputs(
        self, 
        ret: Mapping[str, torch.Tensor], 
        ret0: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded0: Optional[Mapping[str, torch.Tensor]] = None
    ):
        collected = super(DANBO, self).collect_outputs(ret, ret0, encoded, encoded0)

        if encoded is not None and 'agg_logit' in encoded and self.training:

            collected['agg_logit'] = encoded['agg_logit']
            if encoded0 is not None:
                collected['agg_logit0'] = encoded0['agg_logit']   

        if encoded is not None and 'vol_invalid' in encoded and self.training:
            collected['vol_invalid'] = encoded['vol_invalid']
        
        collected['vol_scale'] = self.get_axis_scale()
        return collected
    
    def get_axis_scale(self, rigid_idxs: Optional[torch.Tensor] = None):
        axis_scale = self.graph_net.get_axis_scale()
        if rigid_idxs is None:
            return axis_scale
        return axis_scale[rigid_idxs]

    def get_summaries(self, *args, **kwargs):
        # grab volume boxes
        # TODO: implement these on networks instead ..
        volume_scales = self.graph_net.get_axis_scale().detach().cpu().numpy()
        volume_boxes = create_axis_boxes(volume_scales)

        rest_pose = self.rest_pose.cpu().numpy()
        l2w = np.eye(4).reshape(-1, 4, 4).repeat(len(volume_scales), 0)
        l2w[:, :3, -1] = rest_pose

        posed_boxes = boxes_to_pose(l2w, volume_boxes, rest_pose, skel_type=self.skel_type, rest_heads=self.rest_heads)

        # use matplotlib (mpl) for this because it's faster
        box_image = plot_boxes_mpl(posed_boxes)

        return {'box_image': (box_image, 'png')}
    