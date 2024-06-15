import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

import mcubes
from core.utils.skeleton_utils import *
from core.utils.visualization import *

import numpy as np
import math
import time


def label_surface_points(points, model, n_iters=5):
    skel_profile = model.skel_profile
    rigid_idxs = np.array([
        i for i in range(24) if i not in skel_profile['rigid_ignore']
    ])

    box_centers = torch.zeros(1, len(rigid_idxs), 3)
    box_centers = model.pts_embedder.unalign_pts(box_centers,rigid_idxs=rigid_idxs)
    box_centers = (box_centers + model.rest_pose[None, rigid_idxs])
    dist_to_boxes = ((points[:, None] - box_centers)**2).sum(dim=-1)
    axis_scale = model.get_axis_scale().detach()
    rest_pose = model.rest_pose

    # initialize the labels to the closest box
    # Note: this does not guarantee the points to be valid in the box!
    dist_sorted = dist_to_boxes.sort().indices
    label_ptrs = torch.zeros(len(points)).long()

    # now, go through the points to check if they are valid.
    # if not, assign them to the next closest boxes    
    all_valid = False
    iter_cnt = 0
    while not all_valid and iter_cnt < n_iters:
        iter_cnt += 1 
        labels = dist_sorted[torch.arange(len(points)), label_ptrs]

        invalid_idxs = []
        for i, ri in enumerate(rigid_idxs):
            label_idxs = torch.where(labels==i)[0].clone()
            label_pts = points[label_idxs]
            r_loc = rest_pose[ri:ri+1]
            label_pts_j = (label_pts - r_loc)
            T = model.pts_embedder.transforms[ri:ri+1]
            Rpts = (T[..., :3, :3] @ label_pts_j[..., None])[..., 0]
            apts = Rpts + T[..., :3, -1]
            apts =  apts / axis_scale[ri:ri+1]
            
            invalid = ((apts.abs() > 1.).sum(-1) > 0).float()
            invalid_idxs.append(label_idxs[invalid==1])
        invalid_idxs = torch.cat(invalid_idxs)
        if len(invalid_idxs) > 0:
            label_ptrs[invalid_idxs] += 1
        else:
            all_valid = True
    labels = dist_sorted[torch.arange(len(points)), label_ptrs]
    return labels


def fill_valid_tensor(vals, index_shape, valid_idxs, filled_value=0):
    """ Create a tensor and fill in values to tensor[valid_idxs] = vals

    Parameter
    ---------
    vals: tensor, (len(valid_idxs), ....), values to fill in the full tensor
    index_shape: (...) shape that correspond to the indexing of valid_idxs
    """
    tensor = torch.zeros(np.prod(index_shape), *vals.shape[1:], dtype=vals.dtype) + filled_value
    tensor[valid_idxs] = vals
    return tensor.reshape(*index_shape, *vals.shape[1:])


def softmax_invalid(logit, invalid, eps=1e-7, temp=1.0, **kwargs):
    """ Softmax with invalid part handling
    """

    logit = logit / temp
    # find the valid part
    valid = 1 - invalid
    # for stability: doesn't change the output as the term will be canceled out
    max_logit = logit.max(dim=-1, keepdim=True)[0]

    # only keep the valid part!
    nominator = torch.exp(logit - max_logit) * valid
    denominator = torch.sum(nominator + eps, dim=-1, keepdim=True)

    return nominator / denominator.clamp(min=eps)


def extract_mcubes(model, res=512, radius=1.3, threshold=20., n_pts=1000):
    # obtain all the surface points in each of the volume box
    skel_type = model.skel_type
    skel_profile = model.skel_profile
    rest_pose = model.rest_pose
    rigid_idxs = np.array([
        i for i in range(24) if i not in skel_profile['rigid_ignore']
    ])

    axis_scale = model.get_axis_scale().detach()

    t = np.linspace(-radius, radius, res)
    grid_pts = torch.tensor(np.stack(np.meshgrid(t, t, t), axis=-1).astype(np.float32))
    geom_inputs = {
        'pts': grid_pts.reshape(-1, 3),
        'bones': torch.zeros(1, 24, 3)
    }

    model.eval()
    preds = model(geom_inputs, forward_type='geometry', chunk=1024*64)
    density = preds['density'].reshape(res, res, res).cpu().numpy()
    density = np.maximum(density, 0)

    vertices, triangles = mcubes.marching_cubes(
        density.reshape(res, res, res), 
        20.
    )
    # scale the vertices back to the original size
    # Note: in mcubes, the range is [0, res-1]
    vertices = radius * 2 * (vertices / (res-1) - 0.5)

    # differ by a rotate along z axis
    rot = np.array([
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., 0., 1.],]
    ).astype(np.float32)
    surface_pts = torch.tensor(vertices @ rot).float()

    # label each vertex with the closest box center
    labels = label_surface_points(surface_pts, model)

    # extract points
    val_range = [-1., 1.]
    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'lightblue']
    N_fps = n_pts
    fig = None
    extracted = {
        'anchor_pts': [],
        'canon_pts': [],
        'axis_scale': axis_scale.cpu(),
    }
    for i, ri in enumerate(rigid_idxs):
        cnt = (labels == i).sum()
        label_pts = surface_pts[labels==i].clone()
        
        r_loc = rest_pose[ri:ri+1]
        label_pts_j = (label_pts - r_loc)
        T = model.pts_embedder.transforms[ri:ri+1]
        Rpts = (T[..., :3, :3] @ label_pts_j[..., None])[..., 0]
        apts = Rpts + T[..., :3, -1]
        apts =  apts / axis_scale[ri:ri+1]
        
        invalid = ((apts.abs() > 1.).sum(-1) > 0).float()
        valid = 1 - invalid
        valid_pts = label_pts[valid > 0]

        if len(valid_pts) > N_fps:
            fps_idx = farthest_point_sampling(valid_pts, n=N_fps).sort().values
        else:
            fps_idx = torch.arange(len(valid_pts))
        color = colors[i % len(colors)]

        # for visualization
        fig = plot_points3d(valid_pts[fps_idx].reshape(-1, 3).cpu().numpy(), color=color, fig=fig, x_range=val_range, y_range=val_range, z_range=val_range)
        print(f'joint {skel_type.joint_names[ri]}: {cnt-invalid.sum()}/{cnt} (invalid count: {invalid.sum()}) ')

        valid_anchors = apts[valid > 0][fps_idx]
        extracted['anchor_pts'].append(valid_anchors)
        extracted['canon_pts'].append(valid_pts[fps_idx])

    img = byte2array(fig.to_image(format='png'))
    return extracted, img


class ParallelLinear(nn.Module):

    def __init__(self, n_parallel, in_feat, out_feat, share=False, bias=True):
        super().__init__()
        self.n_parallel = n_parallel
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.share = share

        if not self.share:
            self.register_parameter('weight',
                                    nn.Parameter(torch.randn(n_parallel, in_feat, out_feat),
                                                 requires_grad=True)
                                   )
            if bias:
                self.register_parameter('bias',
                                        nn.Parameter(torch.randn(1, n_parallel, out_feat),
                                                     requires_grad=True)
                                       )
        else:
            self.register_parameter('weight', nn.Parameter(torch.randn(1, in_feat, out_feat),
                                                           requires_grad=True))
            if bias:
                self.register_parameter('bias', nn.Parameter(torch.randn(1, 1, out_feat), requires_grad=True))
        if not hasattr(self, 'bias'):
            self.bias = None
        #self.bias = nn.Parameter(torch.Tensor(n_parallel, 1, out_feat))
        self.reset_parameters()
        """
        self.conv = nn.Conv1d(in_feat * n_parallel, out_feat * n_parallel,
                              kernel_size=1, groups=n_parallel, bias=bias)
        """

    def reset_parameters(self):

        for n in range(self.n_parallel):
            # transpose because the weight order is different from nn.Linear
            nn.init.kaiming_uniform_(self.weight[n].T.data, a=math.sqrt(5))

        if self.bias is not None:
            #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0].T)
            #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            #nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.bias.data, 0.)

    def forward(self, x):
        weight, bias = self.weight, self.bias
        if self.share:
            weight = weight.expand(self.n_parallel, -1, -1)
            if bias is not None:
                bias = bias.expand(-1, self.n_parallel, -1)
        out = torch.einsum("bkl,klj->bkj", x, weight.to(x.device))
        if bias is not None:
            out = out + bias.to(x.device)
        return out

    def extra_repr(self):
        return "n_parallel={}, in_features={}, out_features={}, bias={}".format(
            self.n_parallel, self.in_feat, self.out_feat, self.bias is not None
        )


def factorize_grid_sample(features, coords, align_corners=False, mode='bilinear', padding_mode='zeros', 
                          training=False, need_hessian=False):
    '''
    Factorized grid sample: only gives the same outcomes as the original one under certain circumstances.
    '''
    bnd = 1.0 if align_corners else 2.0 / 3.0
    # cols are meant to make grid_samples axis-independent
    cols = torch.linspace(-bnd, bnd, 3)
    coords = coords[..., None]
    cols = cols.reshape(1, 1, 3, 1).expand(*coords.shape[:2], -1, -1)
    coords = torch.cat([cols, coords], dim=-1)
    # the default grid_sample in pytorch does not have 2nd order derivative
    if training and need_hessian:
        sample_feature = grid_sample_diff(features, coords, padding_mode=padding_mode, align_corners=align_corners)
        #sample_feature = F.grid_sample(features, coords, mode=mode,
        #                               align_corners=align_corners, padding_mode=padding_mode)
    else:
        sample_feature = F.grid_sample(features, coords, mode=mode,
                                       align_corners=align_corners, padding_mode=padding_mode)
    #return sample_feature
    return sample_feature

def factorize_triplane_grid_sample(features, coords, align_corners=False, mode='bilinear', padding_mode='zeros',
                                   training=False, need_hessian=False):

    xy = coords[..., :2] # 1,2
    yz = coords[..., 1:3] # 2,3
    xz = coords[..., [0, 2]] # 1,3

    # shape in (N_graphs * N_joints, 3, N_samples, 2)
    B, N_samples = xy.shape[:2]

    # want 2,3 - 1,3 - 1,2
    #triplane_coords = torch.stack([xy, yz, xz], dim=1)
    triplane_coords = torch.stack([yz, xz, xy], dim=1)
    triplane_coords = triplane_coords.reshape(B * 3, N_samples, 1, 2)
    features = features.flatten(end_dim=1)
        
    if training and need_hessian:
        sample_feature = grid_sample_diff(features, triplane_coords, 
                                          padding_mode=padding_mode, 
                                          align_corners=align_corners)
    else:
        sample_feature = F.grid_sample(features, triplane_coords, mode=mode,
                                       align_corners=align_corners, 
                                       padding_mode=padding_mode)
    return sample_feature


def grid_sample_diff(image, optical, padding_mode='zero', align_corners=False, eps=1e-7, clamp_x=True):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    if align_corners:
        ix = ((ix + 1.) / 2.) * (IW-1)
        iy = ((iy + 1.) / 2.) * (IH-1)
    else:
        ix = ((ix + 1.) * IW - 1) / 2.
        iy = ((iy + 1.) * IH - 1) / 2.
    

    with torch.no_grad():
        iy_nw = torch.floor(iy)
        iy_ne = iy_nw
        iy_sw = iy_nw + 1
        iy_se = iy_nw + 1

        if clamp_x:
            # this is a special case: our x is used as an indicator,
            # so it should always be integer value
            ix = ix.round()
            ix_nw = ix
            ix_ne = ix_nw + 1 
            ix_sw = ix_nw
            ix_se = ix_nw + 1
        else:
            ix_nw = torch.floor(ix)
            ix_ne = ix_nw + 1
            ix_sw = ix_nw
            ix_se = ix_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    valid_nw = torch.ones_like(iy_nw)
    valid_ne = torch.ones_like(iy_ne)
    valid_sw = torch.ones_like(iy_sw)
    valid_se = torch.ones_like(iy_se)
    if padding_mode == 'zeros':

        bnd_z = 0 - eps
        bnd_W = IW - 1 + eps
        bnd_H = IH - 1 + eps

        valid_nw[ix_nw < bnd_z] = 0.
        valid_nw[ix_nw > bnd_W] = 0.
        valid_nw[iy_nw < bnd_z] = 0.
        valid_nw[iy_nw > bnd_H] = 0.

        valid_ne[ix_ne < bnd_z] = 0.
        valid_ne[ix_ne > bnd_W] = 0.
        valid_ne[iy_ne < bnd_z] = 0.
        valid_ne[iy_ne > bnd_H] = 0.

        valid_sw[ix_sw < bnd_z] = 0.
        valid_sw[ix_sw > bnd_W] = 0.
        valid_sw[iy_sw < bnd_z] = 0.
        valid_sw[iy_sw > bnd_H] = 0.

        valid_se[ix_se < bnd_z] = 0.
        valid_se[ix_se > bnd_W] = 0.
        valid_se[iy_se < bnd_z] = 0.
        valid_se[iy_se > bnd_H] = 0.

        valid_nw = valid_nw.view(N, -1, H * W)
        valid_ne = valid_ne.view(N, -1, H * W)
        valid_sw = valid_sw.view(N, -1, H * W)
        valid_se = valid_se.view(N, -1, H * W)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))
    if padding_mode == 'zeros':
        nw_val = nw_val * valid_nw
        ne_val = ne_val * valid_ne
        sw_val = sw_val * valid_sw
        se_val = se_val * valid_se

    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
        ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
        sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
        se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )

    return out_val 

def grid_sample_diff_3d(image, optical, padding_mode='zero', align_corners=False, eps=1e-7, clamp_x=False):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    if align_corners:
        ix = ((ix + 1.) / 2.) * (IW - 1)
        iy = ((iy + 1.) / 2.) * (IH - 1)
        iz = ((iz + 1.) / 2.) * (ID - 1)
        ix = ((ix + 1.) / 2.) * (IW-1)
        iy = ((iy + 1.) / 2.) * (IH-1)
    else:
        ix = ((ix + 1.) * IW - 1) / 2.
        iy = ((iy + 1.) * IH - 1) / 2.
        iz = ((iz + 1.) * ID - 1) / 2.

    with torch.no_grad():
        
        if clamp_x:
            ix_tnw = torch.round(ix)
            iy_tnw = torch.round(iy)
            iz_tnw = torch.round(iz)
        else:
            ix_tnw = torch.floor(ix)
            iy_tnw = torch.floor(iy)
            iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    tnw_val = tnw_val.view(N, C, D, H, W)
    tne_val = tne_val.view(N, C, D, H, W)
    tsw_val = tsw_val.view(N, C, D, H, W)
    tse_val = tse_val.view(N, C, D, H, W)
    bnw_val = bnw_val.view(N, C, D, H, W)
    bne_val = bne_val.view(N, C, D, H, W)
    bsw_val = bsw_val.view(N, C, D, H, W)
    bse_val = bse_val.view(N, C, D, H, W)

    out_val = (
        tnw_val * tnw.view(N, 1, D, H, W) +
        tne_val * tne.view(N, 1, D, H, W) +
        tsw_val * tsw.view(N, 1, D, H, W) +
        tse_val * tse.view(N, 1, D, H, W) +
        bnw_val * bnw.view(N, 1, D, H, W) +
        bne_val * bne.view(N, 1, D, H, W) +
        bsw_val * bsw.view(N, 1, D, H, W) +
        bse_val * bse.view(N, 1, D, H, W)
    )


    """
    # compute gradient w.r.t x,y,z deformation
    # d/d(ix)
    d_tnwx = -(iy_bse - iy) * (iz_bse - iz)
    d_tnex =  (iy_bsw - iy) * (iz_bsw - iz)
    d_tswx = -(iy - iy_bne) * (iz_bne - iz)
    d_tsex =  (iy - iy_bnw) * (iz_bnw - iz)
    d_bnwx = -(iy_tse - iy) * (iz - iz_tse)
    d_bnex =  (iy_tsw - iy) * (iz - iz_tsw)
    d_bswx = -(iy - iy_tne) * (iz - iz_tne)
    d_bsex=   (iy - iy_tnw) * (iz - iz_tnw)

    grad_x = (tnw_val * d_tnwx.view(N, 1, D, H, W) +
              tne_val * d_tnex.view(N, 1, D, H, W) +
              tsw_val * d_tswx.view(N, 1, D, H, W) +
              tse_val * d_tsex.view(N, 1, D, H, W) +
              bnw_val * d_bnwx.view(N, 1, D, H, W) +
              bne_val * d_bnex.view(N, 1, D, H, W) +
              bsw_val * d_bswx.view(N, 1, D, H, W) +
              bse_val * d_bsex.view(N, 1, D, H, W))
    
    # d/d(iy)
    d_tnwy = (ix_bse - ix) * -(iz_bse - iz)
    d_tney = (ix - ix_bsw) * -(iz_bsw - iz)
    d_tswy = (ix_bne - ix) *  (iz_bne - iz)
    d_tsey = (ix - ix_bnw) *  (iz_bnw - iz)
    d_bnwy = (ix_tse - ix) * -(iz - iz_tse)
    d_bney = (ix - ix_tsw) * -(iz - iz_tsw)
    d_bswy = (ix_tne - ix) *  (iz - iz_tne)
    d_bsey = (ix - ix_tnw) *  (iz - iz_tnw)

    grad_y = (tnw_val * d_tnwy.view(N, 1, D, H, W) +
              tne_val * d_tney.view(N, 1, D, H, W) +
              tsw_val * d_tswy.view(N, 1, D, H, W) +
              tse_val * d_tsey.view(N, 1, D, H, W) +
              bnw_val * d_bnwy.view(N, 1, D, H, W) +
              bne_val * d_bney.view(N, 1, D, H, W) +
              bsw_val * d_bswy.view(N, 1, D, H, W) +
              bse_val * d_bsey.view(N, 1, D, H, W))

    # d/d(iz)
    d_tnwz = (ix_bse - ix) * -(iy_bse - iy) 
    d_tnez = (ix - ix_bsw) * -(iy_bsw - iy)
    d_tswz = (ix_bne - ix) * -(iy - iy_bne)
    d_tsez = (ix - ix_bnw) * -(iy - iy_bnw)
    d_bnwz = (ix_tse - ix) *  (iy_tse - iy)
    d_bnez = (ix - ix_tsw) *  (iy_tsw - iy)
    d_bswz = (ix_tne - ix) *  (iy - iy_tne)
    d_bsez = (ix - ix_tnw) *  (iy - iy_tnw)

    grad_z = (tnw_val * d_tnwz.view(N, 1, D, H, W) +
              tne_val * d_tnez.view(N, 1, D, H, W) +
              tsw_val * d_tswz.view(N, 1, D, H, W) +
              tse_val * d_tsez.view(N, 1, D, H, W) +
              bnw_val * d_bnwz.view(N, 1, D, H, W) +
              bne_val * d_bnez.view(N, 1, D, H, W) +
              bsw_val * d_bswz.view(N, 1, D, H, W) +
              bse_val * d_bsez.view(N, 1, D, H, W))

    grads = torch.stack([grad_x, grad_y, grad_z], dim=0)
    """
    grads = None

    return out_val, grads

def unrolled_propagate(adj, w):
    '''
    Unrolled adjacency propagation (to save memory and maybe computation)
    '''
    o0 = adj[0, 0, 0] * w[:, 0] + adj[0, 0, 1] * w[:, 1] + adj[0, 0, 2] * w[:, 2] + adj[0, 0, 3] * w[:, 3]
    o1 = adj[0, 1, 1] * w[:, 1] + adj[0, 1, 0] * w[:, 0] + adj[0, 1, 4] * w[:, 4]
    o2 = adj[0, 2, 2] * w[:, 2] + adj[0, 2, 0] * w[:, 0] + adj[0, 2, 5] * w[:, 5]
    o3 = adj[0, 3, 3] * w[:, 3] + adj[0, 3, 0] * w[:, 0] + adj[0, 3, 6] * w[:, 6]
    o4 = adj[0, 4, 4] * w[:, 4] + adj[0, 4, 1] * w[:, 1] + adj[0, 4, 7] * w[:, 7]
    o5 = adj[0, 5, 5] * w[:, 5] + adj[0, 5, 2] * w[:, 2] + adj[0, 5, 8] * w[:, 8]
    o6 = adj[0, 6, 6] * w[:, 6] + adj[0, 6, 3] * w[:, 3] + adj[0, 6, 9] * w[:, 9]
    o7 = adj[0, 7, 7] * w[:, 7] + adj[0, 7, 4] * w[:, 4] + adj[0, 7, 10] * w[:, 10]
    o8 = adj[0, 8, 8] * w[:, 8] + adj[0, 8, 5] * w[:, 5] + adj[0, 8, 11] * w[:, 11]
    o9 = adj[0, 9, 9] * w[:, 9] + adj[0, 9, 6] * w[:, 6] + adj[0, 9, 12] * w[:, 12] + adj[0, 9, 13] * w[:, 13] + adj[0, 9, 14] * w[:, 14]
    o10 = adj[0, 10, 10] * w[:, 10] + adj[0, 10, 7] * w[:, 7]
    o11 = adj[0, 11, 11] * w[:, 11] + adj[0, 11, 8] * w[:, 8]
    o12 = adj[0, 12, 12] * w[:, 12] + adj[0, 12, 9] * w[:, 9] + adj[0, 12, 15] * w[:, 15]
    o13 = adj[0, 13, 13] * w[:, 13] + adj[0, 13, 9] * w[:, 9] + adj[0, 13, 16] * w[:, 16]
    o14 = adj[0, 14, 14] * w[:, 14] + adj[0, 14, 9] * w[:, 9] + adj[0, 14, 17] * w[:, 17]
    o15 = adj[0, 15, 15] * w[:, 15] + adj[0, 15, 12] * w[:, 12]
    o16 = adj[0, 16, 16] * w[:, 16] + adj[0, 16, 13] * w[:, 13] + adj[0, 16, 18] * w[:, 18]
    o17 = adj[0, 17, 17] * w[:, 17] + adj[0, 17, 14] * w[:, 14] + adj[0, 17, 19] * w[:, 19]
    o18 = adj[0, 18, 18] * w[:, 18] + adj[0, 18, 16] * w[:, 16] + adj[0, 18, 20] * w[:, 20]
    o19 = adj[0, 19, 19] * w[:, 19] + adj[0, 19, 17] * w[:, 17] + adj[0, 19, 21] * w[:, 21]
    o20 = adj[0, 20, 20] * w[:, 20] + adj[0, 20, 18] * w[:, 18] + adj[0, 20, 22] * w[:, 22]
    o21 = adj[0, 21, 21] * w[:, 21] + adj[0, 21, 19] * w[:, 19] + adj[0, 21, 23] * w[:, 23]
    o22 = adj[0, 22, 22] * w[:, 22] + adj[0, 22, 20] * w[:, 20]
    o23 = adj[0, 23, 23] * w[:, 23] + adj[0, 23, 21] * w[:, 21]
    o = torch.stack([o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10,
                     o11, o12, o13, o14, o15, o16, o17, o18, o19,
                     o20, o21, o22, o23], dim=1)

    return o


def get_entropy_rgb(confd, encoded, eps=1e-7):
    """
    if 'part_invalid' in encoded:
        part_valid = 1 - encoded['part_invalid']
        max_logit = (part_valid * confd).max(dim=-1, keepdim=True)[0]
        nominator = torch.exp(confd - max_logit) * part_valid
        denominator = torch.sum(nominator + eps, dim=-1, keepdim=True)
        prob = nominator / denominator
    else:
        prob = F.softmax(confd, dim=-1)
    """
    prob = F.softmax(confd, dim=-1)
    max_ent = torch.tensor(confd.shape[-1]).log()
    ent = -(prob * (prob + eps).log()).sum(-1)

    ratio = (ent / max_ent)[..., None]
    start = torch.tensor([0., 0., 1.]).reshape(1, 1, 3)
    end = torch.tensor([1., 0., 0.]).reshape(1, 1, 3)
    rgb = torch.lerp(start, end, ratio)
 
    return rgb


def get_confidence_rgb(confd, encoded):
    # TODO: currently assume skeleton is SMPL!
    # pre-defined 24 colors
    colors = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.5019607843137255, 0.0],
        [0.29411764705882354, 0.0, 0.5098039215686274],
        [1.0, 0.5490196078431373, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.7529411764705882, 0.796078431372549],
        [0.6039215686274509, 0.803921568627451, 0.19607843137254902],
        [0.7372549019607844, 0.5607843137254902, 0.5607843137254902],
        [1.0, 0.4980392156862745, 0.3137254901960784],
        [0.8235294117647058, 0.4117647058823529, 0.11764705882352941],
        [1.0, 0.8941176470588236, 0.7686274509803922],
        [1.0, 0.8431372549019608, 0.0],
        [0.6039215686274509, 0.803921568627451, 0.19607843137254902],
        [0.4980392156862745, 1.0, 0.8313725490196079],
        [0.0, 0.7490196078431373, 1.0],
        [0.0, 0.0, 0.5019607843137255],
        [0.8549019607843137, 0.4392156862745098, 0.8392156862745098],
        [0.5019607843137255, 0.0, 0.0],
        [0.6274509803921569, 0.3215686274509804, 0.17647058823529413],
        [0.5019607843137255, 0.5019607843137255, 0.0],
        [0.5647058823529412, 0.9333333333333333, 0.5647058823529412],
    ])
    selected_color = confd.argmax(dim=-1)
    rgb = colors[selected_color]
    return rgb


def init_volume_scale(base_scale, skel_profile, skel_type):
    if skel_type == SMPLSkeleton:
        scale = init_volume_scale_SMPL(base_scale, skel_profile, skel_type)
    elif skel_type == HARESkeleton:
        scale = init_volume_scale_animal(base_scale, skel_profile, skel_type)
    elif skel_type == WOLFSkeleton:
        scale = init_volume_scale_animal(base_scale, skel_profile, skel_type)
    elif skel_type == MixamoSkeleton:
        scale = init_volume_scale_mixamo(base_scale, skel_profile, skel_type)
    elif skel_type == ViTPoseSkeleton or skel_type == ViTPoseSkeletonv2:
        scale = init_volume_scale_vitpose(base_scale, skel_profile, skel_type)
    else:
        raise NotImplementedError
    return scale


def init_volume_scale_SMPL(base_scale, skel_profile, skel_type):
    # TODO: hard-coded some parts for now ...
    # TODO: deal with multi-subject
    joint_names = skel_type.joint_names
    N_joints = len(joint_names)
    bone_lens = skel_profile['bone_lens'][0]
    bone_lens_to_child = skel_profile['bone_lens_to_child'][0]

    # indices to all body parts
    head_idxs = skel_profile['head_idxs']
    torso_idxs = skel_profile['torso_idxs']
    arm_idxs = skel_profile['arm_idxs']
    leg_idxs = skel_profile['leg_idxs']
    collar_idxs = skel_profile['collar_idxs']

    # some widths
    shoulder_width = skel_profile['shoulder_width'][0]
    knee_width = skel_profile['knee_width'][0]
    collar_width = skel_profile['knee_width'][0]

    # init the scale for x, y and z
    # width, depth
    x_lens = torch.ones(N_joints) * base_scale
    y_lens = torch.ones(N_joints) * base_scale

    # half-width of thighs cannot be wider than the distant between knees in rest pose
    x_lens[leg_idxs] = knee_width * 0.5
    y_lens[leg_idxs] = knee_width * 0.5

    #  half-width of your body and head cannot be wider than shoulder distance (to some scale) 
    #x_lens[torso_idxs] = shoulder_width * 0.70
    #y_lens[torso_idxs] = shoulder_width * 0.70
    x_lens[torso_idxs] = shoulder_width * 0.50
    y_lens[torso_idxs] = shoulder_width * 0.50
    x_lens[collar_idxs] = collar_width * 0.40
    y_lens[collar_idxs] = collar_width * 0.40

    #  half-width of your arms cannot be wider than collar distance (to some scale) 
    x_lens[arm_idxs] = collar_width * 0.40
    y_lens[arm_idxs] = collar_width * 0.40

    # set scale along the bone direction
    # don't need full bone lens because the volume is supposed to centered at the middle of a bone
    z_lens = torch.tensor(bone_lens_to_child.copy().astype(np.float32))
    z_lens = z_lens * 0.8

    # deal with end effectors: make them grow freely
    """
    z_lens[z_lens < 0] = z_lens.max()
    # give more space to head as we do not have head-top joint
    z_lens[head_idxs] = z_lens.max() * 1.1 
    """
    x_lens[head_idxs] = shoulder_width * 0.30
    y_lens[head_idxs] = shoulder_width * 0.35
    # TODO: hack: assume at index 1 we have the head
    y_lens[head_idxs[1]] = shoulder_width * 0.6
    z_lens[head_idxs] = z_lens.max() * 0.30

    # find the lengths from end effector to their parents
    end_effectors = np.array([i for i, v in enumerate(z_lens) if v < 0 and i not in head_idxs])
    z_lens[end_effectors] = torch.tensor(skel_profile['bone_lens_to_child'][0][skel_type.joint_trees[end_effectors]].astype(np.float32)) 

    # handle hands and foots
    scale = torch.stack([x_lens, y_lens, z_lens], dim=-1)

    return scale


def init_volume_scale_animal(base_scale, skel_profile, skel_type):
    # TODO: hard-coded some parts for now ...
    # TODO: deal with multi-subject
    joint_names = skel_type.joint_names
    N_joints = len(joint_names)
    bone_lens = skel_profile['bone_lens'][0]

    # indices to all body parts
    head_idxs = skel_profile['head_idxs']
    torso_idxs = skel_profile['torso_idxs']
    arm_idxs = skel_profile['arm_idxs']
    leg_idxs = skel_profile['leg_idxs']
    collar_idxs = skel_profile['collar_idxs']
    tail_idxs = skel_profile['tail_idxs']
    ear_idxs = skel_profile['ear_idxs']

    # some widths
    hip_f_width = skel_profile['hip_f_width'][0]
    eye_width = skel_profile['eye_width'][0]
    if skel_type == HARESkeleton:
        thigh_b_width = skel_profile['though_b_width'][0]
        ear_width = skel_profile['ear_1_width'][0]
    elif skel_type == WOLFSkeleton:
        thigh_b_width = skel_profile['thigh_b_width'][0]
        ear_width = skel_profile['ear_width'][0]
    else:
        raise NotImplementedError

    # init the scale for x, y and z
    # width, depth
    x_lens = torch.ones(N_joints) * base_scale
    y_lens = torch.ones(N_joints) * base_scale

    # half-width of thighs cannot be wider than the distant between knees in rest pose
    x_lens[leg_idxs] = thigh_b_width * 0.5
    y_lens[leg_idxs] = thigh_b_width * 0.5

    #  half-width of your body and head cannot be wider than shoulder distance (to some scale) 
    #x_lens[torso_idxs] = shoulder_width * 0.70
    #y_lens[torso_idxs] = shoulder_width * 0.70
    x_lens[torso_idxs] = hip_f_width * 1.
    y_lens[torso_idxs] = hip_f_width * 1.


    #  half-width of your arms cannot be wider than collar distance (to some scale) 
    x_lens[arm_idxs] = hip_f_width * 0.5
    y_lens[arm_idxs] = hip_f_width * 0.5

    x_lens[tail_idxs] = hip_f_width * 0.2
    y_lens[tail_idxs] = hip_f_width * 0.2

    # set scale along the bone direction
    # don't need full bone lens because the volume is supposed to centered at the middle of a bone
    z_lens = torch.tensor(bone_lens.copy().astype(np.float32))
    z_lens = z_lens * 0.8
    z_lens_max = z_lens.max()

    # deal with end effectors: make them grow freely
    z_lens[z_lens <= 0] = z_lens.mean()
    """
    # give more space to head as we do not have head-top joint
    z_lens[head_idxs] = z_lens.max() * 1.1 
    """
    x_lens[head_idxs] = eye_width * 1.
    y_lens[head_idxs] = eye_width * 1.
    # TODO: hack: assume at index 1 we have the head
    y_lens[head_idxs[1]] = eye_width * 1.
    z_lens[head_idxs] = z_lens_max * 0.4

    # find the lengths from end effector to their parents
    joint_trees = np.array(skel_type.joint_trees)
    end_effectors = np.array([i for i, v in enumerate(z_lens) if v < 0 and i not in head_idxs])
    #mport pdb; pdb.set_trace()
    #z_lens[end_effectors] = torch.tensor(skel_profile['bone_lens_to_child'][0][joint_trees[end_effectors]].astype(np.float32))

    y_lens[skel_type.root_id] =  hip_f_width * 1.3
    #z_lens[skel_type.root_id] = z_lens_max * 0.5
    x_lens[ear_idxs] = ear_width * 0.5
    y_lens[ear_idxs] = ear_width * 0.5
    z_lens[ear_idxs] *= 1.1

    z_lens[tail_idxs] *= 1.1

    # handle hands and foots
    scale = torch.stack([x_lens, y_lens, z_lens], dim=-1)

    return scale

def init_volume_scale_mixamo(base_scale, skel_profile, skel_type):

    joint_names = skel_type.joint_names
    N_joints = len(joint_names)
    bone_lens = skel_profile['bone_lens'][0]

    # indices to all body parts
    head_idxs = skel_profile['head_idxs']
    torso_idxs = skel_profile['torso_idxs']
    arm_idxs = skel_profile['arm_idxs']
    leg_idxs = skel_profile['leg_idxs']

    upleg_width = skel_profile['upleg_width'][0]
    shoulder_width = skel_profile['shoulder_width'][0]

    # init the scale for x, y and z
    # width, depth
    x_lens = torch.ones(N_joints) * base_scale
    y_lens = torch.ones(N_joints) * base_scale

    # half-width of thighs cannot be wider than the distant between knees in rest pose
    x_lens[leg_idxs] = upleg_width * 0.5
    y_lens[leg_idxs] = upleg_width * 0.5

    x_lens[torso_idxs] = shoulder_width * 0.50
    y_lens[torso_idxs] = shoulder_width * 0.50

    x_lens[arm_idxs] = shoulder_width * 0.30
    y_lens[arm_idxs] = shoulder_width * 0.30

    z_lens = torch.tensor(bone_lens.copy().astype(np.float32))
    z_lens = z_lens * 0.8

    z_lens[leg_idxs] = z_lens[leg_idxs] * 0.8

    end_effectors = np.array([i for i, v in enumerate(z_lens) if v < 0 and i not in head_idxs])
    z_lens[end_effectors] = z_lens.max() * 0.40

    x_lens[head_idxs] = shoulder_width * 0.30
    y_lens[head_idxs] = shoulder_width * 0.35
    # TODO: hack: assume at index 1 we have the head
    y_lens[head_idxs[1]] = shoulder_width * 0.6
    z_lens[head_idxs] = z_lens.max() * 0.30
    z_lens[skel_type.root_id] = z_lens.max() * 0.3


    scale = torch.stack([x_lens, y_lens, z_lens], dim=-1)

    return scale

def init_volume_scale_vitpose(base_scale, skel_profile, skel_type):

    joint_names = skel_type.joint_names
    N_joints = len(joint_names)
    bone_lens = skel_profile['bone_lens'][0]

    # indices to all body parts
    head_idxs = skel_profile['head_idxs']
    torso_idxs = skel_profile['torso_idxs']
    arm_idxs = skel_profile['arm_idxs']
    leg_idxs = skel_profile['leg_idxs']

    upleg_width = skel_profile['thigh_width'][0]
    shoulder_width = skel_profile['shoulder_width'][0]

    # init the scale for x, y and z
    # width, depth
    x_lens = torch.ones(N_joints) * base_scale
    y_lens = torch.ones(N_joints) * base_scale

    # half-width of thighs cannot be wider than the distant between knees in rest pose
    x_lens[leg_idxs] = upleg_width * 0.6
    y_lens[leg_idxs] = upleg_width * 0.6

    x_lens[torso_idxs] = shoulder_width * 0.75
    y_lens[torso_idxs] = shoulder_width * 0.60

    x_lens[arm_idxs] = shoulder_width * 0.40
    y_lens[arm_idxs] = shoulder_width * 0.40

    z_lens = torch.tensor(bone_lens.copy().astype(np.float32))
    z_lens = z_lens * 0.85

    z_lens[leg_idxs] = z_lens[leg_idxs] * 0.85

    end_effectors = np.array([i for i, v in enumerate(z_lens) if v < 0 and i not in head_idxs])
    z_lens[end_effectors] = z_lens.max() * 0.50

    x_lens[head_idxs] = shoulder_width * 0.40
    y_lens[head_idxs] = shoulder_width * 0.45
    # TODO: hack: assume at index 1 we have the head
    y_lens[head_idxs[1]] = shoulder_width * 0.7
    z_lens[head_idxs] = z_lens.max() * 0.40
    z_lens[skel_type.root_id] = z_lens.max() * 0.4


    scale = torch.stack([x_lens, y_lens, z_lens], dim=-1)

    return scale

def random_perturb_on_tangent_plane(
    pts: torch.Tensor,
    normals: torch.Tensor,
    noise: float = 1e-3,
):
    rand_dirs = F.normalize(torch.randn_like(pts), dim=-1)
    tangent = torch.cross(normals, rand_dirs)
    #set the random moving direction to be the tangent direction now
    rand_dirs = tangent 

    return (pts.detach() + noise * torch.randn_like(pts[..., :1]) * rand_dirs).detach()
        

def spline_basis(t: torch.Tensor):
    """ Cubic Hermite Spline basis functions.

    Args:
        t: torch.Tensor, step for the spline interpolation
    """
    t_cubic = t.pow(3)
    t_sqr = t.pow(2.)
    h00 = 2 * t_cubic - 3 * t_sqr + 1
    h10 = t_cubic - 2 * t_sqr + t
    h01 = -2 * t_cubic + 3 * t_sqr
    h11 = t_cubic - t_sqr
    return h00, h10, h01, h11

def spline_feature_interp(
    f: torch.Tensor,
    m: torch.Tensor,
    x: torch.Tensor,
    align_corners: bool = True,
):
    """ Run Cubic Hermite Spline interpolation on the factorized 1D feature map.

    Args:
        f: torch.Tensor [B, F, R, 3], feature
        m: torch.Tensor [B, :, R, 3], tangent
        x: torch.Tensor [B, :, 3], query coordinates

    Returns:
        Interpolated feature map [B, F, :, 3]
    """
    # last dimension is axis dimension. Assumme factorized 1d
    B, F, R, _ = f.shape
    # for the sake of gathering
    fm = torch.cat([f, m], dim=1)

    ix = x
    if align_corners:
        ix = ((ix + 1.) / 2.) * (R - 1)
    else:
        ix = ((ix + 1.) * R - 1) / 2.
    
    with torch.no_grad():
        bnd_below = 0 - 1e-7
        bnd_above = R - 1 + 1e-7
        ix_0 = torch.floor(ix)
        ix_1 = ix_0 + 1

        valid_0 = torch.ones_like(ix).bool()
        valid_1 = torch.ones_like(ix).bool()

        valid_0[ix_0 < bnd_below] = 0
        valid_0[ix_0 > bnd_above] = 0
        valid_1[ix_1 < bnd_below] = 0
        valid_1[ix_1 > bnd_above] = 0
        valid = valid_0 & valid_1

        ix_0 = ix_0.clamp(0, R - 1)
        ix_1 = ix_1.clamp(0, R - 1)

    t = (ix - ix_0 + 1e-8) / (ix_1 - ix_0 + 1e-8)
    t[~valid] =0

    h00, h10, h01, h11 = spline_basis(t[:, None])

    gather_len= fm.shape[1]
    fm0 = torch.gather(fm, dim=2, index=ix_0[:, None].expand(-1, gather_len, -1, -1).long())
    fm1 = torch.gather(fm, dim=2, index=ix_1[:, None].expand(-1, gather_len, -1, -1).long())

    f0, m0 = fm0[:, :F], fm0[:, F:]
    f1, m1 = fm1[:, :F], fm1[:, F:]

    ix_d = (ix_1 - ix_0)[:, None]
    out = h00 * f0 + h10 * ix_d * m0 + h01 * f1 + h11 * ix_d * m1
    out = out * valid[:, None]
    return out

