from core.utils.skeleton_utils import torch
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from core.networks.misc import (
    ParallelLinear,
    factorize_grid_sample,
    init_volume_scale,
    softmax_invalid,
    spline_feature_interp,
)
from core.utils.skeleton_utils import *
from core.positional_enc import PositionalEncoding
from copy import deepcopy
from einops import rearrange
from core.networks.embedding import Optcodes

from typing import Optional, Mapping, List, Union, Callable, Any
from omegaconf import DictConfig


'''
Modified from Skeleton-aware Networks https://github.com/DeepMotionEditing/deep-motion-editing
'''
def skeleton_to_graph(skel: Optional[Skeleton] = None, edges: Optional[List[Union[List, np.ndarray]]] = None):
    ''' Turn skeleton definition to adjacency matrix and edge list
    '''

    if skel is not None:
        edges = []
        for i, j in enumerate(skel.joint_trees):
            if i == j:
                continue
            edges.append([j, i])
    else:
        assert edges is not None

    n_nodes = np.max(edges) + 1
    adj = np.eye(n_nodes, dtype=np.float32)

    for edge in edges:
        adj[edge[0], edge[1]] = 1.0
        adj[edge[1], edge[0]] = 1.0

    return adj, edges


def clamp_deform_to_max(x_d: torch.Tensor, max_deform: float = 0.04):
    x_d_scale = x_d.detach().norm(dim=-1)
    masks = (x_d_scale < max_deform)[..., None]
    return x_d * masks


class DenseWGCN(nn.Module):
    """ Basic GNN layer with learnable adj weights
    """
    def __init__(
        self, 
        adj: torch.Tensor, 
        in_channels: int, 
        out_channels: int, 
        init_adj_w: float = 0.05, 
        bias: bool = True, 
        **kwargs
    ):
        super(DenseWGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        adj = adj.clone()
        idx = torch.arange(adj.shape[-1])
        adj[:, idx, idx] = 1

        init_w = init_adj_w
        perturb = 0.1
        adj_w = (adj.clone() * (init_w + (torch.rand_like(adj) - 0.5 ) * perturb).clamp(min=0.01, max=1.0))
        adj_w[:, idx, idx] = 1.0


        self.lin = nn.Linear(in_channels, out_channels)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.register_buffer('adj', adj) # fixed, not learnable
        self.register_parameter('adj_w', nn.Parameter(adj_w, requires_grad=True)) # learnable

    def get_adjw(self):
        adj, adj_w = self.adj, self.adj_w

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        adj_w = adj_w.unsqueeze(0) if adj_w.dim() == 2 else adj_w
        adj_w = adj_w * adj # masked out not connected part

        return adj_w

    def forward(self, x: torch.Tensor):

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj_w = self.get_adjw().to(x.device)

        out = self.lin(x)
        out = torch.matmul(adj_w, out)

        if self.bias is not None:
            out = out + self.bias

        return out


class DensePNGCN(DenseWGCN):
    """ Basic GNN layer with learnable adj weights, and each node has its own linear layer
    """
    def __init__(
        self, 
        adj: torch.Tensor, 
        in_channel: int, 
        out_channel: int,
        *args, 
        **kwargs
    ):
        super(DensePNGCN, self).__init__(
            adj, 
            in_channel, 
            out_channel,
            *args, 
            **kwargs
        )
        self.lin = ParallelLinear(adj.shape[-1], in_channel, out_channel, bias=False)


class BasicGNN(nn.Module):
    """ A basic GNN with several graph layers
    """

    def __init__(
        self, 
        per_node_input: int, 
        output_ch: int,
        W: int = 64, 
        D: int = 4,
        gcn_module: Callable = DensePNGCN, 
        gcn_module_kwargs: dict = {}, 
        nl: Callable = F.relu, 
        mask_root: bool = True,
        skel_profile: Optional[dict] = None,
        keep_extra_joint: bool = False,
        **kwargs,
    ):
        """
        mask_root: Bool, to remove root input so everything is in relative coord
        """
        super(BasicGNN, self).__init__()
        assert skel_profile is not None


        self.skel_profile = deepcopy(skel_profile)
        self.skel_type = skel_profile['skel_type']
        self.per_node_input = per_node_input
        self.output_ch = output_ch
        self.keep_extra_joint = keep_extra_joint

        self.rigid_idxs = skel_profile['rigid_idxs']
        self.mask_root = mask_root
        self.W = W
        self.D = D

        adj_matrix, _ = skeleton_to_graph(self.skel_type)
        self.adj_matrix = adj_matrix
        self.gcn_module_kwargs = gcn_module_kwargs

        self.nl = nl
        if output_ch is None:
            self.output_ch = self.W + 1
        else:
            self.output_ch = output_ch
        self.init_network(gcn_module)

    def init_network(self, gcn_module: Callable):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D = self.W, self.D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, **self.gcn_module_kwargs)]
        for i in range(D-2):
            layers += [gcn_module(adj_matrix, W, W, **self.gcn_module_kwargs)]

        layers += [gcn_module(adj_matrix, W, self.output_ch, **self.gcn_module_kwargs)]
        self.layers = nn.ModuleList(layers)

        if self.mask_root:
            # mask root inputs, so that everything is in relative coordinate
            mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            mask[:, self.skel_type.root_id, :] = 0.
            self.register_buffer('mask', mask)

    def forward(self, inputs: torch.Tensor, **kwargs):

        n = inputs
        if self.mask_root:
            n = n * self.mask

        for i, l in enumerate(self.layers):
            n = l(n)
            if (i + 1) < len(self.layers) and self.nl is not None:
                n = self.nl(n, inplace=True)
            if (i + 2) == len(self.layers) and self.rigid_idxs is not None and not self.keep_extra_joint:
                n = n[:, self.rigid_idxs]
        return n 

    def get_adjw(self):
        adjw_list = []

        for m in self.modules():
            if hasattr(m, 'adj_w'):
                adjw_list.append(m.get_adjw())

        return adjw_list


class MixGNN(BasicGNN):

    def __init__(
        self, 
        *args, 
        voxel_res: int = 1, 
        voxel_feat: int = 1, 
        fc_D: int = 2, 
        legacy: bool = False,
        fc_module: Callable = ParallelLinear, 
        **kwargs,
    ):
        """ 
        Parameters
        ----------
        fc_D: int, start using fc_module at layer fc_D
        fc_module: nn.Module, module to use at layer fc_D
        """
        self.fc_module = fc_module
        self.voxel_res = voxel_res
        self.voxel_feat = voxel_feat
        self.fc_D = fc_D
        self.legacy = legacy

        super(MixGNN, self).__init__(*args, **kwargs)

    @property
    def output_size(self):
        if self.output_ch is not None:
            return self.output_ch
        return self.voxel_res**3 * self.voxel_feat

    @property
    def sample_feat_size(self):
        return self.voxel_feat
    
    def init_network(self, gcn_module: Callable):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, fc_D = self.W, self.D, self.fc_D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, **self.gcn_module_kwargs)]
        for i in range(D-2):
            if i + 1 < fc_D:
                layers += [gcn_module(adj_matrix, W, W, **self.gcn_module_kwargs)]
            else:
                layers += [self.fc_module(n_nodes, W, W)]

        if self.fc_module in [ParallelLinear]:
            n_nodes = len(self.rigid_idxs) if self.rigid_idxs is not None and not self.keep_extra_joint else n_nodes
            layers += [self.fc_module(n_nodes, W, self.output_size)]
        else:
            layers += [self.fc_module(adj_matrix, W, self.output_size, **self.gcn_module_kwargs)]

        if self.mask_root or self.legacy:
            mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            mask[:, self.skel_type.root_id, :] = 0.
            self.register_buffer('mask', mask)

        self.layers = nn.ModuleList(layers)
        self.volume_shape = [len(self.skel_type.joint_trees), self.voxel_feat] + \
                                3 * [self.voxel_res]

    def sample_from_volume(self, graph_feat: torch.Tensor, x: torch.Tensor):
        """ Sample per-part features from predicted factorized volume.

        Args:
            graph_feat (torch.Tensor): (N_graph, N_joints, voxel_res^3*voxel_feat) predicted factorized volume
            x (torch.Tensor): (N_rays, N_samples, N_joints, 3) 3D coordinates in local joint spaces
        
        Returns:
            part_feat (torch.Tensor): (N_rays, N_samples, N_joints, voxel_feat) sampled per-part features
            invalid (torch.Tensor): (N_rays, N_samples, N_joints) invalid mask indicating whether the sampled points are out of the volume
        """
        align_corners = self.align_corners
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_trees) - offset

        # turns graph_feat into (N, F, H, W, D) format for more efficient grid_sample
        # -> (N_graphs * N_joints, F, H, W, D)
        graph_feat = graph_feat.reshape(-1, *self.volume_shape[1:])
        # reshape and permute x similarly
        x = x.reshape(N_graphs, N_expand, N_samples, N_joints, 3)
        # turns x into (N, H, W, D, 3) format for efficient_grid_sample
        # -> (N_graphs * N_joints, N_expands, N_samples, 1, 3)
        x = x.permute(0, 3, 1, 2, 4).reshape(N_graphs * N_joints, N_expand, N_samples, 1, 3)
        # (N_rays * N_samples * N_joints, 1, 1, 1, 3)
        # mode='bilinear' is actually trilinear for 5D input
        part_feat = F.grid_sample(
            graph_feat, 
            x, 
            mode='bilinear',
            padding_mode='zeros', 
            align_corners=align_corners
        )
        # turn it back to (N_rays, N_samples, N_joints, voxel_feat)
        part_feat = part_feat.reshape(N_graphs, N_joints, self.voxel_feat, N_expand, N_samples)
        part_feat = part_feat.permute(0, 3, 4, 1, 2).reshape(N_rays, N_samples, N_joints, -1)

        return part_feat.reshape(N_rays, N_samples, N_joints, -1), torch.ones(N_rays, N_samples, N_joints)
    

class DANBOGNN(MixGNN):

    def __init__(
        self, 
        *args, 
        opt_scale: bool = False, 
        base_scale: float = 0.5, 
        alpha: float = 2.,
        beta: float = 6.,
        legacy: bool = False,
        post_conv_layers: int = 0,
        spline: bool = False,
        **kwargs
    ):

        self.opt_scale  = opt_scale
        super(DANBOGNN, self).__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.legacy = legacy
        self.voxel_feat_per_axis = self.voxel_feat // 3
        self.post_conv_layers = post_conv_layers
        self.spline = spline
        self.init_scale(base_scale)

        self.init_post_conv()

    def init_post_conv(self):
        self.post_conv = None
        if self.post_conv_layers < 1:
            return
        n_nodes = len(self.rigid_idxs) if self.rigid_idxs is not None and not self.keep_extra_joint else len(self.skel_type.joint_names)

        # number of joints * axis_per_joint
        g = n_nodes * 3
        ch = g * self.voxel_feat_per_axis
        layers = [nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1, groups=g)]
        for i in range(self.post_conv_layers - 1):
            layers += [
                nn.ReLU(inplace=True), 
                nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1, groups=g),
            ]
        self.post_conv = nn.Sequential(*layers)

    @property
    def output_size(self):
        return self.voxel_res * self.voxel_feat
    
    def forward(self, *args, **kwargs):
        n = super().forward(*args, **kwargs)

        # n: (N_graph, N_joints, N_voxel_res * voxel_feat * 3)
        if self.post_conv is not None:
            # run post conv
            n = rearrange(n, 'g j (f r c) -> g (j c f) r', r=self.voxel_res, c=3)
            n = rearrange(self.post_conv(n), 'g (j c f) r -> g j (f r c)', f=self.voxel_feat_per_axis, c=3)
        return n
    
    def fix_danbo_box_params(self):
        self.axis_scale.requires_grad = False
    
    def init_scale(self, base_scale: float):
        N_joints = len(self.skel_type.joint_names) 

        scale = torch.ones(N_joints, 3) * base_scale
        if self.skel_profile is not None:
            scale = init_volume_scale(base_scale, self.skel_profile, self.skel_type)
        self.register_buffer('base_scale', scale.clone())
        self.register_parameter('axis_scale', nn.Parameter(scale, requires_grad=self.opt_scale))

    def sample_from_volume(self, graph_feat: torch.Tensor, x: torch.Tensor, *args, **kwargs):
        '''
        graph_feat: predicted factorized volume (N_graph, N_joints, voxel_feat*voxel_res*3)
        x: points in local joint coordinates for sampling
        '''
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        
        # reshape scale to (1, 1, N_joints, 3)
        alpha, beta = self.alpha, self.beta
        x_v = x / self.get_axis_scale().reshape(1, 1, -1, 3).abs()
        coord_window =  torch.exp(-alpha * ((x_v**beta).sum(-1))).detach()

        # "hard" invalid
        invalid = ((x_v.abs() > 1).sum(-1) > 0).float()

        # (N_graphs * N_joints, C, voxel_res, 3)
        graph_feat = rearrange(graph_feat, 'g j (f r c) -> (g j) f r c', r=self.voxel_res, c=3)

        # permute to make it looks like (N, H, W) for faster, cheaper grid_sampling
        # (N_graphs, N_expand * N_samples, N_joints, 3) -> (N_graphs * N_joints, N_expand * N_samples, 3)
        x_sampled = rearrange(x_v, '(g e) s j c -> (g j) (e s) c', e=N_expand)
        
        if self.spline:
            pad = torch.zeros_like(graph_feat[:, :, :1])
            rf = torch.cat([graph_feat[:, :, 1:], pad], dim=2)
            lf = torch.cat([pad, graph_feat[:, :, :-1]], dim=2)
            m = ((graph_feat - lf) + (rf - graph_feat)) * 0.5
            graph_feat = spline_feature_interp(
                graph_feat,
                m,
                x_sampled,
                True,
            )
        else:
            graph_feat = factorize_grid_sample(
                graph_feat, 
                x_sampled, 
                training=self.training, 
                need_hessian=kwargs['need_hessian']
            )

        # turn it back to (N_rays, N_samples, N_joints, voxel_feat, 3)
        graph_feat = rearrange(graph_feat, '(g j) f (e s) c -> (g e) s j (f c)', g=N_graphs, e=N_expand)
        graph_feat = graph_feat * coord_window[..., None]

        return graph_feat, invalid, x_v
    
    def get_axis_scale(self):
        axis_scale = self.axis_scale.abs()
        if self.legacy:
            return axis_scale
        diff = axis_scale.detach() - self.base_scale * 0.95
        return torch.maximum(axis_scale, axis_scale - diff)
    
    def check_invalid(self, x: torch.Tensor):
        """ Assume points are in volume space already
        
        Args:
            x (torch.Tensor): (N_rays, N_samples, N_joints, 3)

        Returns:
            x_v (torch.Tensor): (N_rays, N_samples, N_joints, 3) points scaled by volume scales
            invalid (torch.Tensor): (N_rays, N_samples, N_joints) invalid mask indicating whether the sampled points are out of the volume
        """
        x_v = x / self.get_axis_scale().reshape(1, 1, -1, 3).abs()
        invalid = ((x_v.abs() > 1).sum(-1) > 0).float()
        return x_v, invalid
    
class DANBOGNNSpline(DANBOGNN):

    @property
    def output_size(self):
        # one extra dimension for tangent parameter
        return super().output_size + self.voxel_res * 3

    def sample_from_volume(self, graph_feat: torch.Tensor, x: torch.Tensor, *args, **kwargs):
        '''
        graph_feat: predicted factorized volume (N_graph, N_joints, voxel_feat*voxel_res*3)
        x: points in local joint coordinates for sampling
        '''
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        # reshape scale to (1, 1, N_joints, 3)
        alpha, beta = self.alpha, self.beta
        x_v = x / self.get_axis_scale().reshape(1, 1, -1, 3).abs()
        coord_window =  torch.exp(-alpha * ((x_v**beta).sum(-1))).detach()

        # "hard" invalid
        invalid = ((x_v.abs() > 1).sum(-1) > 0).float()

        # (N_graphs * N_joints, C, voxel_res, 3)
        graph_feat = rearrange(graph_feat, 'g j (f r c) -> (g j) f r c', r=self.voxel_res, c=3)

        # permute to make it looks like (N, H, W) for faster, cheaper grid_sampling
        # (N_graphs, N_expand * N_samples, N_joints, 3) -> (N_graphs * N_joints, N_expand * N_samples, 3)
        x_sampled = rearrange(x_v, '(g e) s j c -> (g j) (e s) c', e=N_expand)
        graph_feat = spline_feature_interp(
            graph_feat[:, 1:, :], 
            graph_feat[:, :1, :], 
            x_sampled, 
            align_corners=True
        ) 

        # turn it back to (N_rays, N_samples, N_joints, voxel_feat, 3)
        graph_feat = rearrange(graph_feat, '(g j) f (e s) c -> (g e) s j (f c)', g=N_graphs, e=N_expand)
        graph_feat = graph_feat * coord_window[..., None]

        return graph_feat, invalid, x_v

class DANBOPoseFree(DANBOGNN):

    def init_network(self, *args, **kwargs):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, fc_D = self.W, self.D, self.fc_D

        n_nodes = adj_matrix.shape[-1]
        per_joint_feature = torch.randn(1, n_nodes, per_node_input)
        self.register_parameter(
            'per_joint_feature', 
            nn.Parameter(per_joint_feature, requires_grad=True)
        )
        layers = [ParallelLinear(n_nodes, per_node_input, W), nn.ReLU(inplace=True)]
        for i in range(D-2):
            layers += [ParallelLinear(n_nodes, W, W)]
            layers += [nn.ReLU(inplace=True)]
        layers += [ParallelLinear(n_nodes, W, self.output_size)]

        self.mask_root = False # forcefully set to false as this doesn't matter
        self.layers = nn.Sequential(*layers)
        self.volume_shape = [len(self.skel_type.joint_trees), self.voxel_feat] + \
                                3 * [self.voxel_res]

    def forward(self, inputs: torch.Tensor, *args, **kwargs):

        n = self.per_joint_feature
        n = self.layers(n)
        n = n.repeat(inputs.shape[0], 1, 1)
        return n


class NPCGNNFiLM(MixGNN):
    """ 
    Note: forward function turns input body pose input per-part FiLM condition
    """

    def __init__(
        self,
        *args,
        RW: int = 128,
        RD: int = 3,
        n_hops: int = 2,
        n_pose_feat: int = 8,
        num_freqs: int = 0,
        opt_scale: bool = False,
        base_scale: float = 0.5,
        max_deform: float = 0.04,
        pts_per_volume: int = 200,
        deform_scale: float = 0.01,
        clamp_deform: bool = True,
        **kwargs,
    ):
        """
        n_hops: number of hops for initializing LBS weights
        """
        self.RW = RW
        self.RD = RD 
        self.n_hops = n_hops
        self.n_pose_feat = n_pose_feat
        self.num_freqs = num_freqs
        self.opt_scale = opt_scale
        self.max_deform = max_deform
        self.pts_per_volume = pts_per_volume
        self.deform_scale = deform_scale
        self.clamp_deform = clamp_deform


        super(NPCGNNFiLM, self).__init__(*args, **kwargs)

        self.init_scale(base_scale)
        self.init_deform_regressor()
        self.init_blend_network()
        self.reset_parameters()

    @property
    def output_size(self):
        return self.RW * 2

    def reset_parameters(self):

        nn.init.normal_(self.deform_layers[-1].weight.data, 0, 0.01)
        nn.init.constant_(self.deform_layers[-1].bias.data, 0)
        print('GNN FiLM parameters reset')

    def init_network(self, gcn_module: Callable):

        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, fc_D = self.W, self.D, self.fc_D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, **self.gcn_module_kwargs)]
        for i in range(D-2):
            if i + 1 < fc_D:
                layers += [gcn_module(adj_matrix, W, W, **self.gcn_module_kwargs)]
            else:
                layers += [self.fc_module(n_nodes, W, W)]

        if self.fc_module in [ParallelLinear]:
            n_nodes = len(self.rigid_idxs) if self.rigid_idxs is not None and not self.keep_extra_joint else n_nodes
            layers += [self.fc_module(n_nodes, W, self.output_size)]
        else:
            layers += [self.fc_module(adj_matrix, W, self.output_size, **self.gcn_module_kwargs)]

        if self.mask_root or self.legacy:
            mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            mask[:, self.skel_type.root_id, :] = 0.
            self.register_buffer('mask', mask)

        self.layers = nn.ModuleList(layers)
        self.volume_shape = [len(self.skel_type.joint_trees), self.voxel_feat] + 3 * [self.voxel_res]

    def init_deform_regressor(self):
        W, adj_matrix = self.W, self.adj_matrix
        n_nodes = len(self.rigid_idxs) if self.rigid_idxs is not None else adj_matrix.shape[-1]
        self.posi_enc = PositionalEncoding(3, num_freqs=self.num_freqs)

        deform_input = self.posi_enc.dims
        self.deform_input = deform_input

        self.deform_layers = nn.ModuleList([
            ParallelLinear(n_nodes, deform_input, self.RW),
            ParallelLinear(n_nodes, self.RW, self.RW),
            ParallelLinear(n_nodes, self.RW, self.RW),
            ParallelLinear(n_nodes, self.RW, self.n_pose_feat + 3),
        ])
        self.init_blend_network()
    
    def init_blend_network(self):
        assert self.rigid_idxs is not None
        assert self.skel_profile is not None
        N_joints = len(self.rigid_idxs)

        self.blend_network = nn.Sequential(
            ParallelLinear(N_joints, 3, self.W//4),
            nn.ReLU(inplace=True),
            ParallelLinear(N_joints, self.W//4, self.W//4),
            nn.ReLU(inplace=True),
            ParallelLinear(N_joints, self.W//4, N_joints),
        )

    def init_scale(self, base_scale: float):
        N_joints = len(self.skel_type.joint_names) 

        scale = torch.ones(N_joints, 3) * base_scale
        if self.skel_profile is not None:
            scale = init_volume_scale(base_scale, self.skel_profile, self.skel_type)

        self.register_buffer('base_scale', scale.clone())
        self.register_parameter('axis_scale', nn.Parameter(scale, requires_grad=self.opt_scale))

    def get_axis_scale(self, rigid_idxs: Optional[torch.Tensor] = None):
        axis_scale = self.axis_scale.abs()
        diff = axis_scale.detach() - self.base_scale * 0.95
        axis_scale = torch.maximum(axis_scale, axis_scale - diff)
        if rigid_idxs is not None:
            return axis_scale[rigid_idxs]
        return axis_scale
    
    def forward(
        self,
        p_j: torch.Tensor,
        p_c: torch.Tensor,
        r2ws: torch.Tensor,
        aj2ws: torch.Tensor,
        pose: torch.Tensor,
        lbs_weights: torch.Tensor,
        lbs_masks: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        bones: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """ 
        p_j: points in joint space
        pc_j: ponits in canonical space
        """
        # get FiLM condition from poses
        # dp: space is in joint space
        film = super().forward(pose)
        f_theta, dp = self.predict_nl_deform(p_j, film, aj2ws)

        # deformation before clamping
        dp_uc = dp.clone()
        if self.clamp_deform:
            dp = clamp_deform_to_max(dp, self.max_deform)

        p_lbs, T_lbs = self.lbs(
            p_j, 
            p_c, 
            r2ws,
            lbs_weights, 
            lbs_masks,
        )

        p_w = dp + p_lbs
        # un-clampped version for loss computation
        p_w_uc = dp_uc + p_lbs

        return {
            'p_w': p_w,
            'p_w_uc': p_w_uc,
            'dp': dp,
            'dp_uc': dp_uc,
            'p_lbs': p_lbs,
            'f_theta': f_theta,
            'T_lbs': T_lbs,
        }
    
    def predict_nl_deform(
        self,
        p_j: torch.Tensor,
        film: torch.Tensor,
        aj2ws: torch.Tensor,
        **kwargs
    ):
        """ 
        predict non-linear deformation in scaled, bone-aligned per-joint space
        
        returns: per-joint space in world scale (undone per-joint scale/alignment)

        """
        # p_j is in (N_joints, N_pts, 3)
        # -> reshape to match N_graphs and ParallelLinear API
        # -> reshape p_j to (N_graphs * N_pts, N_joints, 3)
        N_graphs, N_joints = film.shape[:2]
        N_pts = p_j.shape[1]

        p_j = rearrange(p_j, 'j p d -> 1 p j d')
        z = rearrange(
            p_j.expand(N_graphs, -1, -1, -1),
            'g p j d -> (g p) j d'
        )

        film = rearrange(
            film[:, None].expand(-1, N_pts, -1, -1), 
            'g p j d -> (g p) j d'
        )
        alpha, gamma = torch.chunk(film, dim=-1, chunks=2)

        for l, layer in enumerate(self.deform_layers):
            z = layer(z)
            if l < len(self.deform_layers) - 1:
                z = F.relu(z)
            if l == 0:
                z = z * alpha + gamma
        
        z = rearrange(
            z,
            '(g p) j d -> g j p d', 
            g=N_graphs, 
            j=N_joints,
        )
        
        f_theta, dp = torch.split(z, [self.n_pose_feat, 3], dim=-1)
        # apply deformation scale
        # directly predict in unscaled space
        dp = dp * self.deform_scale 

        # apply unalignment matrix and transformation to world
        # note that dp is direction, so we only need to apply the rotation
        R = aj2ws[..., :3, :3]
        dp = (R @ dp[..., None])[..., 0]

        return f_theta, dp
    
    def lbs(
        self,
        p_j: torch.Tensor,
        p_c: torch.Tensor,
        r2ws: torch.Tensor,
        lbs_weights: torch.Tensor,
        lbs_masks: torch.Tensor,
        **kwargs
    ):
        # TODO: 'predict' may not be the right term
        # p_j is in (N_joints, N_pts, 3)
        # -> reshape p_j to (N_pts, N_joints, 3) for api

        # Step 1: get final lbs weights
        p_j = rearrange(p_j, 'j p d -> p j d')
        lbs_residual = rearrange(
            self.blend_network(p_j),
            'p j d -> j p d'
        )
        lbs_logits = lbs_weights + lbs_residual
        invalid = 1 - lbs_masks
        lbs_weights = softmax_invalid(lbs_logits, invalid)

        # Step 2: do the actual lbs
        p_lbs, T_lbs = self._lbs(p_c, lbs_weights, r2ws)
        
        return p_lbs, T_lbs
    
    def _lbs(self, p_c: torch.Tensor, w: torch.Tensor, T: torch.Tensor):
        """ 
        p_c: (N_joints, N_pts, 3) canonical points
        w: (N_joints, N_pts, N_joints) lbs weights
        T: (N_graphs, N_joints, 4, 4) per-joint transformation matrix from *canonical space* to world space
        """

        p_c = rearrange(p_c, 'j p d -> 1 j p d 1')

        # expand to have (N_graphs, N_joints, N_pts, N_joints, 4, 4)
        T = rearrange(T, 'g j a b -> g 1 1 j a b', a=4, b=4)
        # expand to have (N_graphs, N_joints, N_pts, N_joints, 1, 1)
        w = rearrange(w, 'j p k -> 1 j p k 1 1')

        T_lbs = (w * T).sum(dim=3)
        p_lbs =  (T_lbs[..., :3, :3] @ p_c + T_lbs[..., :3, -1:])[..., 0]  

        return p_lbs, T_lbs


class NPCFPSGNNFiLM(NPCGNNFiLM):

    def init_deform_regressor(self):
        W, adj_matrix = self.W, self.adj_matrix
        n_nodes = len(self.rigid_idxs) if self.rigid_idxs is not None else adj_matrix.shape[-1]
        self.posi_enc = PositionalEncoding(3, num_freqs=self.num_freqs)

        deform_input = self.posi_enc.dims
        self.deform_input = deform_input

        self.deform_layers = nn.ModuleList([
            nn.Linear(deform_input, self.RW),
            nn.Linear(self.RW, self.RW),
            nn.Linear(self.RW, self.RW),
            nn.Linear(self.RW, self.n_pose_feat + 3),
        ])

        # TODO: elsewhere
        self.film_mapper = nn.Linear(self.RW * 2, self.RW * 2)
        self.init_blend_network()

    def init_blend_network(self):
        assert self.rigid_idxs is not None
        assert self.skel_profile is not None
        N_joints = len(self.rigid_idxs)

        self.blend_network = nn.Sequential(
            nn.Linear(3, self.W),
            nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W),
            nn.ReLU(inplace=True),
            nn.Linear(self.W, N_joints),
        )

    def forward(
        self,
        p_c: torch.Tensor,
        r2ws: torch.Tensor,
        pose: torch.Tensor,
        lbs_weights: torch.Tensor,
        lbs_masks: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        bones: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """ 
        p_j: points in joint space
        pc_j: ponits in canonical space
        """
        # get FiLM condition from poses
        # dp: space is in joint space
        film = MixGNN.forward(self, pose)

        # get lbs and transformation matrix first
        p_lbs, lbs_weights, T_lbs = self.lbs(
            p_c, 
            r2ws,
            lbs_weights, 
            lbs_masks,
        )

        f_theta, dp = self.predict_nl_deform(
            p_c, 
            film, 
            T_lbs,
            lbs_weights,
        )

        # deformation before clamping
        dp_uc = dp.clone()
        if self.clamp_deform:
            dp = clamp_deform_to_max(dp, self.max_deform)

        p_w = dp + p_lbs
        # un-clampped version for loss computation
        p_w_uc = dp_uc + p_lbs

        return {
            'p_w': p_w,
            'p_w_uc': p_w_uc,
            'dp': dp,
            'dp_uc': dp_uc,
            'p_lbs': p_lbs,
            'f_theta': f_theta,
            'r2ws': T_lbs,
        }

    def predict_nl_deform(
        self, 
        p_c: torch.Tensor, 
        film: torch.Tensor, 
        r2ws: torch.Tensor, 
        lbs_weights: torch.Tensor,
        **kwargs
    ):
        N_graphs, N_joints = film.shape[:2]
        z = rearrange(p_c, 'p d -> 1 p d')

        film = rearrange(film, 'g j f -> g 1 j f')
        lbs_weights = rearrange(lbs_weights, 'p j -> 1 p j 1')
        film = self.film_mapper((film * lbs_weights).sum(dim=2))

        alpha, gamma = torch.chunk(film, dim=-1, chunks=2)

        for l, layer in enumerate(self.deform_layers):
            z = layer(z)

            if l < len(self.deform_layers) - 1:
                z = F.relu(z)

            if l == 0:
                z = z * (1 + alpha) + gamma

        f_theta, dp = torch.split(z, [self.n_pose_feat, 3], dim=-1)

        # rotate the deformation from canonical space to world space
        R = r2ws[..., :3, :3]
        dp = (R @ dp[..., None])[..., 0]

        return f_theta, dp
    
    def lbs(
        self,
        p_c: torch.Tensor,
        r2ws: torch.Tensor,
        lbs_weights: torch.Tensor,
        lbs_masks: torch.Tensor,
        **kwargs,
    ):
        lbs_residual = self.blend_network(p_c)
        lbs_logits = lbs_weights + lbs_residual
        invalid = 1 - lbs_masks
        lbs_weights = softmax_invalid(lbs_logits, invalid)

        p_lbs, T_lbs = self._lbs(p_c, lbs_weights, r2ws)

        return p_lbs, lbs_weights, T_lbs 

    def _lbs(self, p_c: torch.Tensor, w: torch.Tensor, T: torch.Tensor):
        """ 
        p_c: (N_pts, 3) canonical points
        w: (N_pts, N_joints) lbs weights
        T: (N_graphs, N_joints, 4, 4) per-joint transformation matrix from *canonical space* to world space
        """
        # TODO: can we merge this with the one above?

        p_c = rearrange(p_c, 'p d -> 1 p d 1')

        # expand to have (N_graphs, N_joints, N_pts, N_joints, 4, 4)
        T = rearrange(T, 'g j a b -> g 1 j a b', a=4, b=4)
        # expand to have (N_graphs, N_joints, N_pts, N_joints, 1, 1)
        w = rearrange(w, 'p j -> 1 p j 1 1')

        T_lbs = (w * T).sum(dim=2)
        p_lbs =  (T_lbs[..., :3, :3] @ p_c + T_lbs[..., :3, -1:])[..., 0]  

        return p_lbs, T_lbs