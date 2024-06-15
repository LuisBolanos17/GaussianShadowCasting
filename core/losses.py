import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from core.utils.skeleton_utils import (
    rasterize_points
)

from typing import Mapping, Any, Optional, Callable
from einops import rearrange


class BaseLoss(nn.Module):
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class NeRFRGBMSELoss(BaseLoss):

    def __init__(self, fine: float = 1.0, coarse: float = 1.0, **kwargs):
        super(NeRFRGBMSELoss, self).__init__(**kwargs)
        self.fine = fine
        self.coarse = coarse

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):

        rgb_pred = preds['rgb_map']
        loss_fine = (rgb_pred - batch['target_s']).pow(2.).mean()

        loss_coarse = torch.tensor(0.0)

        if 'rgb0' in preds:
            rgb_pred = preds['rgb0']
            loss_coarse = (rgb_pred - batch['target_s']).pow(2.).mean()
        
        loss = loss_fine * self.fine + loss_coarse * self.coarse

        return loss, {'loss_fine': loss_fine.item(), 'loss_coarse': loss_coarse.item()}


class NeRFRGBLoss(BaseLoss):

    def __init__(self, fine: float = 1.0, coarse: float = 1.0, **kwargs):
        super(NeRFRGBLoss, self).__init__(**kwargs)
        self.fine = fine
        self.coarse = coarse

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):

        rgb_pred = preds['rgb_map']
        loss_fine = (rgb_pred - batch['target_s']).abs().mean()

        loss_coarse = torch.tensor(0.0)

        if 'rgb0' in preds:
            rgb_pred = preds['rgb0']
            loss_coarse = (rgb_pred - batch['target_s']).abs().mean()
        
        loss = self.weight*(loss_fine * self.fine + loss_coarse * self.coarse)

        return loss, {'loss_fine': loss_fine.item(), 'loss_coarse': loss_coarse.item()}



class SoftSoftmaxLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], model: Callable, **kwargs):

        a = preds['agg_logit']
        labels = ((preds['T_i'] * preds['alpha']) > 0).float()

        vol_invalid = preds['vol_invalid']
        vol_valid = 1 - vol_invalid

        p = model.get_agg_weights(a, vol_invalid, mask_invalid=False)
        p_valid = (p * vol_valid).sum(-1)
        soft_softmax_loss = (labels - p_valid).pow(2.).mean()

        loss = soft_softmax_loss * self.weight 

        valid_count = ((vol_valid.sum(-1) > 0) * labels).sum()
        sigmoid_act = (p * vol_valid).sum(-1) * labels
        act_avg = sigmoid_act.detach().sum() / valid_count
        act_max = sigmoid_act.detach().max()

        loss_stats = {
            'soft_softmax_loss': soft_softmax_loss.item(), 
            'sigmoid_avg_act': act_avg.item(), 
            'sigmoid_max_act': act_max.item(),
        } 
        return loss, loss_stats


class VolScaleLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        vol_scale = preds['vol_scale']
        valid = ((1 - preds['vol_invalid']).reshape(-1, len(vol_scale))).sum(dim=0) > 0
        scale_loss = (torch.prod(vol_scale, dim=-1) * valid).sum() 
        loss = scale_loss * self.weight

        scale_avg = vol_scale.detach().mean(0)
        scale_x, scale_y, scale_z = scale_avg
        
        return loss, {'opt_scale_x': scale_x, 'opt_scale_y': scale_y, 'opt_scale_z': scale_z}
    

class EikonalLoss(BaseLoss):

    def __init__(
        self, 
        *args, 
        use_valid: bool = True, 
        start_iter: Optional[int] = None, 
        warmup_iter: Optional[int] = None,
        **kwargs
    ):
        self.use_valid = use_valid
        self.start_iter = start_iter
        self.warmup_iter = warmup_iter
        super(EikonalLoss, self).__init__(*args, **kwargs)

    def get_weight(self, global_iter: int):
        if self.start_iter is None or self.warmup_iter is None:
            return self.weight

        t = np.clip((global_iter - self.start_iter) / self.warmup_iter, 0, 1.0)
        weight = t * self.weight
        return weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        if 'vol_invalid' in preds:
            valid = (1 - preds['vol_invalid']).sum(-1).clamp(max=1.0)
        else:
            valid = torch.ones(preds['surface_gradient'].shape[:-1])
        
        if self.start_iter is not None and batch['global_iter'] < self.start_iter:
            zero = torch.tensor(0.)
            return zero, {'eikonal_loss': zero, 'surface_normal': zero.item()}
        
        if self.use_valid:
            norm = preds['surface_gradient'][valid > 0].norm(dim=-1)
            eikonal_loss = (norm - 1).pow(2.).sum() / valid.sum()
        else:
            norm = preds['surface_gradient'].norm(dim=-1)
            eikonal_loss = (norm - 1).pow(2.).mean()

        weight = self.get_weight(batch['global_iter'])
        loss = eikonal_loss * weight

        return loss, {'eikonal_loss': eikonal_loss, 'surface_normal': norm.mean().item()}


class PointDeformLoss(BaseLoss):

    def __init__(self, *args, weight: float = 0.1, threshold: float = 0.04, **kwargs):
        '''
        threshold in meter
        '''
        self.threshold = threshold
        super(PointDeformLoss, self).__init__(*args, weight=weight, **kwargs)

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        vol_scale = preds['vol_scale'].reshape(1, -1, 1, 3)
        # in m-unit space
        if batch['device_cnt'] > 1:
            vol_scale = torch.chunk(vol_scale, chunks=batch['device_cnt'], dim=1)[0]
        dp = preds['dp'] if not ('dp_uc' in preds) else preds['dp_uc']

        N_joints = dp.shape[1]

        # TODO: hard coded
        threshold = self.threshold
        mag = dp.pow(2.).sum(dim=-1)
        mag = torch.where((mag + 1e-6).pow(0.5) > threshold, mag, torch.zeros_like(mag))
        point_deform_loss = mag.mean()
        loss = point_deform_loss * self.weight

        return loss, {'point_deform_loss': point_deform_loss, 
                      'max_deform_norm': dp.norm(dim=-1).max(), 
                      'mean_deform_norm': dp.norm(dim=-1).mean(), 
                      'median_deform_norm': dp.norm(dim=-1).median()
                     }


class PointCloudsEikonalLoss(BaseLoss):

    def __init__(
        self, 
        *args, 
        weight: float = 0.001, 
        decay_iter: Optional[int] = None,
        cooldown_iter: Optional[int] = None,
        **kwargs
    ):
        '''
        when schedule=True, linearly scale the weight to weight
        '''
        self.decay_iter = decay_iter
        self.cooldown_iter = cooldown_iter
        super().__init__()
        self.weight_ = weight

    def get_weight(self, global_iter: int):
        if self.decay_iter is None:
            return self.weight_
        if self.cooldown_iter is None:
            return 0.0

        t = 1 - np.clip((global_iter - self.decay_iter) / (self.cooldown_iter), 0, 1)
        weight = t * self.weight_
        return weight
    

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        weight = self.get_weight(batch['global_iter'])
        if weight == 0.:
            return torch.tensor(0.), {}
        pc_grad = preds['pc_grad']

        pc_norm = pc_grad.norm(dim=-1)
        pc_eikonal_loss = (pc_norm - 1).pow(2.).mean()
        loss = pc_eikonal_loss * weight

        return loss, {'pc_eikonal_loss': pc_eikonal_loss}


class PointCloudsSurfaceLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        if self.weight == 0.:
            return torch.tensor(0.), {}
        sdf = preds['pc_sigma']
        mask = (sdf > 0).float()
        pc_surface_loss = (sdf * mask).mean()
        # anchor points are on the surface -> sigma = 0
        loss = pc_surface_loss * self.weight

        return loss, {'pc_surface_loss': pc_surface_loss}


class PointCloudsBetaLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        pts_beta = preds['pts_beta']
        pc_beta_loss = pts_beta.pow(2.).mean()
        loss = pc_beta_loss * self.weight
        return loss, {'pc_beta_loss': pc_beta_loss, 'pc_beta_avg': pts_beta.mean(), 'pc_beta_max': pts_beta.max()}

class PointCloudsNeighborLoss(BaseLoss):
    """
    distance between neighboring points should be preserved
    """
    def __init__(self, *args, weight: float = 100., **kwargs):
        super().__init__()
        self.weight_ = weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        weight = self.weight_
        if weight == 0.:
            return torch.tensor(0.), {}
        # deformed anchor location in world space
        p_w = preds['p_w'] if not 'p_w_uc' in preds else preds['p_w_uc']

        N_graphs = p_w.shape[0]

        #if batch['device_cnt'] > 1:
        #    preds['nb_idxs'] = preds['nb_idxs'][:N_joints]
        #    preds['nb_diffs'] = preds['nb_diffs'][:N_joints]

        # neighbors for each point
        # note: the first one is always the point itself
        nb_idxs = preds['nb_idxs'][..., 1:]
        nb_dists = preds['nb_diffs'][..., 1:, :].norm(dim=-1)

        nb_pts = p_w.reshape(N_graphs, -1, 3)[:, nb_idxs]

        deform_dists = (nb_pts - p_w[..., None, :]).norm(dim=-1)
        dist_loss = (deform_dists - nb_dists).pow(2.).mean()
        loss = dist_loss * weight
        return loss, {'pc_dist_loss': dist_loss}


class BkgdLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        if self.weight == 0.:
            return torch.tensor(0.), {}
        bkgd_map = (1 - preds['acc_map'])[..., None].detach()
        pred_bgs = preds['bg_preds']
        target = batch['target_s_not_masked']

        bkgd_loss = (bkgd_map * (pred_bgs - target).pow(2.)).mean()
        loss = bkgd_loss * self.weight
        return loss, {'bkgd_loss': bkgd_loss}


class SigmaLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        T_i = preds['T_i']
        logp = torch.exp(-T_i.abs()) + torch.exp(-(1-T_i).abs())
        sigma_loss = -logp.mean() 
        loss = sigma_loss * self.weight

        return loss, {'sigma_loss': sigma_loss}


class GaussianVisibilityLoss(BaseLoss): # Fits gaussians to NeRF density with mean and sigma regularization

    def __init__(self, *args, gaussian_sigma_weight: float = 1.0, gaussian_mean_weight: float = 0.1,
                 hold_off_iters: int = 0, interpolation_iters: int = 2500, weight_a:  float = 0.5, weight_b: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        # gaussian_sigma_weight: the weight relative to the gaussian fitting loss for the scale loss
        # gaussian_mean_weight: the weight relative to the gaussian fitting loss for regularizing the position of the gaussian
        # hold_off_iters: number of iterations at the start to not do weight interpolation
        # interpolation_iters: number of iterations that the linear interpolation takes after hold_off_iters
        # a: starting value of the loss' weight
        # b: ending value of the loss' weight
        self.gaussian_sigma_weight = gaussian_sigma_weight
        self.gaussian_mean_weight = gaussian_mean_weight
        self.hold_off_iters = hold_off_iters
        self.interpolation_iters = interpolation_iters
        self.a = weight_a
        self.b = weight_b
    
    def get_weight(self, global_iter: int):
        t = np.clip((global_iter - self.hold_off_iters) / (self.interpolation_iters), 0.0, 1.0)
        # NOTE: linear interpolation
        weight = (1 - t) * (self.a) + (t) * (self.b)
        return weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        diff = preds['gaussian_density_diff']
        sigma = preds['gaussian_sigma'] + 1e-8
        mean = preds['gaussian_mean']
        density_loss = diff.pow(2.).mean()
        sigma_loss = torch.zeros_like(sigma)
        # sigma_loss[sigma < 0.05] = 0.1*(torch.exp((20*sigma[sigma < 0.05]-1)**30)-1)
        # sigma_loss[sigma >= 0.05] = 20*(sigma[sigma >= 0.05]-0.05)**2.7
        sigma_loss[sigma < 0.02] = 0.00002 / sigma[sigma<0.02]
        sigma_loss[sigma >= 0.02] = 100*(sigma[sigma>=0.02] - 0.02)**4 + 0.001
        sigma_loss = sigma_loss.mean()
        #rest_heads = torch.Tensor(batch['rest_heads'])
        #rest_pose = batch['rest_pose']
        #bone_middle = (rest_heads - rest_pose)/2
        bone_middle = batch['bone_middles']
        x = mean-bone_middle[:, None, :]
        #mean_loss = ((torch.exp(20*x) + torch.exp(-20*x))/1000).mean()
        mean_loss = ((100*x**4 + 1)**(1/4) - 1).mean() 
        weight = self.get_weight(batch['global_iter'])
        loss = (self.gaussian_sigma_weight*sigma_loss + self.gaussian_mean_weight*mean_loss + density_loss)  * weight
        return loss, {'gaussian_density_loss': density_loss, 'gaussian_sigma_loss': sigma_loss, 'gaussian_mean_loss': mean_loss}
    
class MaskLossL1(BaseLoss): # Pushes accumulation to be 1 inside the mask, 0 elsewhere

    def __init__(self, *args, hold_off_iters: int = 0, interpolation_iters: int = 2500, weight_a:  float = 0.1, weight_b: float = 0.1, **kwargs):
        super(MaskLossL1, self).__init__(*args, **kwargs)
        self.hold_off_iters = hold_off_iters
        self.interpolation_iters = interpolation_iters
        self.a = weight_a
        self.b = weight_b
    
    def get_weight(self, global_iter: int):
        t = np.clip((global_iter - self.hold_off_iters) / (self.interpolation_iters), 0.0, 1.0)
        # NOTE: linear interpolation
        weight = (1 - t) * (self.a) + (t) * (self.b)
        return weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        acc = preds['acc_map']
        mask = torch.clone(batch['target_s'][:,0])
        mask[batch['fgs'][:,0] > 0.5] = 1.0
        mask[batch['fgs'][:,0] <= 0.5] = 0.0
        loss_fine = (acc - mask).abs().mean()
        loss_coarse = torch.tensor(0.0)
        weight = self.get_weight(batch['global_iter'])
        if 'acc_map0' in preds:
            acc = preds['acc_map0']
            loss_coarse = (acc - mask).abs().mean()
        mask_loss = loss_fine + loss_coarse
        loss = mask_loss * weight

        return loss, {'mask_loss': mask_loss}

class GreyLoss(BaseLoss): # Pushes unlit rgb to be (0.5, 0.5, 0.5)

    def __init__(self, fine: float = 1.0, coarse: float = 1.0,
                 hold_off_iters: int = 0, interpolation_iters: int = 2500, weight_a:  float = 0.5, weight_b: float = 0.01, **kwargs):
        super(GreyLoss, self).__init__(**kwargs)
        self.fine = fine
        self.coarse = coarse
        self.hold_off_iters = hold_off_iters
        self.interpolation_iters = interpolation_iters
        self.a = weight_a
        self.b = weight_b

    def get_weight(self, global_iter: int):
        t = np.clip((global_iter - self.hold_off_iters) / (self.interpolation_iters), 0.0, 1.0)
        # NOTE: linear interpolation
        weight = (1 - t) * (self.a) + (t) * (self.b)
        return weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        rgb_pred = preds['rgb_map']
        target = torch.clone(batch['target_s'])
        target[batch['fgs'][:,0] > 0.5] = 0.75
        loss_fine = (rgb_pred - target).abs().mean()

        loss_coarse = torch.tensor(0.0)

        if 'rgb_map0' in preds:
            rgb_pred = preds['rgb_map0']
            loss_coarse = (rgb_pred - batch['target_s']).abs().mean()
        
        weight = self.get_weight(batch['global_iter'])
        loss = weight*(loss_fine * self.fine + loss_coarse * self.coarse)

        return loss, {'grey_loss_fine': loss_fine.item(), 'grey_loss_coarse': loss_coarse.item()}
    
class AlbedoLoss(BaseLoss): # Pushes unlit rgb to be image samples

    def __init__(self, fine: float = 1.0, coarse: float = 1.0,
                 hold_off_iters: int = 0, interpolation_iters: int = 2500, weight_a:  float = 0.5, weight_b: float = 0.00, **kwargs):
        super(AlbedoLoss, self).__init__(**kwargs)
        self.fine = fine
        self.coarse = coarse
        self.hold_off_iters = hold_off_iters
        self.interpolation_iters = interpolation_iters
        self.a = weight_a
        self.b = weight_b

    def get_weight(self, global_iter: int):
        t = np.clip((global_iter - self.hold_off_iters) / (self.interpolation_iters), 0.0, 1.0)
        # NOTE: linear interpolation
        weight = (1 - t) * (self.a) + (t) * (self.b)
        return weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        rgb_pred = preds['rgb_map']
        target = batch['target_s']
        loss_fine = (rgb_pred - target).abs().mean()

        loss_coarse = torch.tensor(0.0)

        if 'rgb_map0' in preds:
            rgb_pred = preds['rgb_map0']
            loss_coarse = (rgb_pred - batch['target_s']).abs().mean()
        
        weight = self.get_weight(batch['global_iter'])
        loss = weight*(loss_fine * self.fine + loss_coarse * self.coarse)

        return loss, {'albedo_loss_fine': loss_fine.item(), 'albedo_loss_coarse': loss_coarse.item()}
    
class RGBLoss(BaseLoss): # Pushes lit rgb to be image samples (gradient flows through light variables & unlit rgb)

    def __init__(self, hold_off_iters: int = 5000, interpolation_iters: int = 2500, weight_a:  float = 0.0, weight_b: float = 1.0, **kwargs):
        super(RGBLoss, self).__init__(**kwargs)
        self.hold_off_iters = hold_off_iters
        self.interpolation_iters = interpolation_iters
        self.a = weight_a
        self.b = weight_b

    def get_weight(self, global_iter: int):
        t = np.clip((global_iter - self.hold_off_iters) / (self.interpolation_iters), 0.0, 1.0)
        # NOTE: linear interpolation
        weight = (1 - t) * (self.a) + (t) * (self.b)
        return weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        rgb_pred = preds['rgb_lit_map']
        target = batch['target_s']
        loss_fine = (rgb_pred - target).abs().mean()
        
        weight = self.get_weight(batch['global_iter'])
        loss = weight*(loss_fine)

        return loss, {'rgb_loss': loss_fine.item()}
    
class AmbientLightLoss(BaseLoss):

    def __init__(self, hold_off_iters: int = 5000, interpolation_iters: int = 2500, weight_a:  float = 0.0, weight_b: float = 1.0, target_value=0.15, **kwargs):
        super(AmbientLightLoss, self).__init__(**kwargs)
        self.hold_off_iters = hold_off_iters
        self.interpolation_iters = interpolation_iters
        self.a = weight_a
        self.b = weight_b
        self.target_value = target_value

    def get_weight(self, global_iter: int):
        t = np.clip((global_iter - self.hold_off_iters) / (self.interpolation_iters), 0.0, 1.0)
        # NOTE: linear interpolation
        weight = (1 - t) * (self.a) + (t) * (self.b)
        return weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        ambient = batch['ambient']
        ambient_loss = ((ambient - self.target_value)**2).mean()
        
        weight = self.get_weight(batch['global_iter'])
        
        loss = weight*(ambient_loss)

        return loss, {'ambient_loss': ambient_loss.item()}


class NormalTVLoss(BaseLoss):

    def __init__(self, *args, hold_off_iters: int = 0, patch_size: int = 2, **kwargs):
        super(NormalTVLoss, self).__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.hold_off_iters = hold_off_iters
        assert self.patch_size > 1

    def get_weight(self, global_iter: int):
        if global_iter < self.hold_off_iters:
            return 0.0
        return self.weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        normal_map = rearrange(preds['normal_map'], '(r h w) c -> r h w c', h=self.patch_size, w=self.patch_size)
        normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True) + 1e-8)
        fgs = rearrange(batch['fgs'], '(r p) 1 -> r p', p=self.patch_size**2)
        valid = fgs.all(dim=-1, keepdim=True)

        diff_w = (normal_map[:, :, 1:, :] - normal_map[:, :, :-1, :]).pow(2.).sum(dim=-1)
        diff_h = (normal_map[:, 1:, :, :] - normal_map[:, :-1, :, :]).pow(2.).sum(dim=-1)
        tv_loss = (((diff_w + diff_h)) * valid[..., None]).mean()

        weight = self.get_weight(batch['global_iter'])
        loss = tv_loss * weight


        return loss, {'normal_tv_loss': tv_loss}
    

class CurvatureLoss(BaseLoss):

    def __init__(self, *args, detach_target: bool = False, angle_loss: bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.detach_target = detach_target
        self.angle_loss = angle_loss

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):

        if 'surface_gradient_is' not in preds:
            return torch.tensor(0), {'curvature_loss': torch.tensor(0.)}
        
        pred = F.normalize(preds['surface_gradient_is'] + 1e-8, dim=-1)
        pred_jitter = F.normalize(preds['surface_gradient_jitter'] + 1e-8, dim=-1)
        if not self.angle_loss:
            curvature_loss = ((pred * pred_jitter).sum(dim=-1) - 1).pow(2.).mean()
        else:
            dot = (pred * pred_jitter).sum(dim=-1)
            curvature = torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) / np.pi
            curvature_loss = curvature.mean()

        loss = curvature_loss * self.weight
        return loss, {'curvature_loss': curvature_loss}

class HDRiSqrtRegularization(BaseLoss):
    def __init__(self, **kwargs):
        super(HDRiSqrtRegularization, self).__init__(**kwargs)

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], model: Callable, **kwargs):
        ys,xs = torch.meshgrid(torch.arange(model.light.hdri_resolution[0]),torch.arange(model.light.hdri_resolution[1]))
        ys = (ys+0.5)/64
        distortion_weights = torch.sqrt(1-(2*ys-1)**2) # to normalize for stretching caused by equirectangular projection
        intensities = torch.linalg.norm(model.light.envs, dim=-1)
        #mask = intensities > 0.75
        HDRiSqrtLoss = (torch.sqrt(intensities * distortion_weights[None,:,:])).mean()
        loss = HDRiSqrtLoss * self.weight
        return loss, {'HDRi_Sqrt_loss': HDRiSqrtLoss}
    

class HDRiL1Regularization(BaseLoss):
    def __init__(self, **kwargs):
        super(HDRiL1Regularization, self).__init__(**kwargs)
        #self.hdri_distortion_weights = torch.ones_like(model.light.envs.shape)

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], model: Callable, **kwargs):
        ys,xs = torch.meshgrid(torch.arange(model.light.hdri_resolution[0]),torch.arange(model.light.hdri_resolution[1]))
        ys = (ys+0.5)/64
        distortion_weights = torch.sqrt(1-(2*ys-1)**2) # to normalize for stretching caused by equirectangular projection
        HDRiSqrtLoss = (torch.abs(model.light.envs) * distortion_weights[None,:,:,None]).mean()
        loss = HDRiSqrtLoss * self.weight
        return loss, {'HDRi_L1_loss': HDRiSqrtLoss}