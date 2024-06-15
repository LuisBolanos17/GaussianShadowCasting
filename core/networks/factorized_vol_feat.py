import torch
import torch.nn as nn
import torch.nn.functional as F

from core.networks.misc import (
    factorize_grid_sample,
    ParallelLinear,
)
from einops import rearrange


class RandomFeatures(nn.Module):

    def __init__(
        self,
        n_out: int,
        n_vols: int, 
        n_pts: int = 200,
        flatten: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.flatten = flatten

        feat = torch.randn(n_vols, n_pts, n_out) * 0.1
        self.register_parameter("feature", nn.Parameter(feat, requires_grad=True))
    
    def forward(self, *args, **kwargs):
        if not self.flatten:
            return self.feature
        return rearrange(self.feature, 'v p f -> (v p) f')

    def reset_parameters(self):
        feat = torch.randn_like(self.feature) * 0.1
        self.feature.data = feat


class Vector3DFeatures(nn.Module):
    """ Factorize 3D volume features into three axis-aligned feature vectors.
    """

    def __init__(
        self, 
        n_in: int, 
        n_out: int, 
        n_vols: int, 
        n_pts: int = 200, 
        feat_res: int = 16, 
        without_map: bool = False, 
        **kwargs
    ):
        super().__init__()
        #feat = torch.randn(n_vols, n_pts, n_out) * 0.1
        feat = torch.randn(n_vols, n_out//3, feat_res, 3) * 0.1
        self.without_map = without_map

        self.register_parameter(
            'feature',
            nn.Parameter(feat, requires_grad=True)
        )
        if not self.without_map:
            self.mapper = nn.Sequential(
                ParallelLinear(n_vols, n_out, 32),
                nn.ReLU(inplace=True),
                ParallelLinear(n_vols, 32, n_out),
            )
    
    def forward(self, pts: torch.Tensor, need_hessian: bool = False, *args, **kwargs):
        vol_feat = factorize_grid_sample(
            self.feature, 
            pts, 
            training=self.training, 
            need_hessian=need_hessian
        )
        vol_feat = rearrange(vol_feat, 'j f p c -> p j (f c)')

        if not self.without_map:
            vol_feat = self.mapper(vol_feat)
        vol_feat = rearrange(vol_feat, 'p j c -> j p c')

        return vol_feat

    def reset_parameters(self):
        pass


class Vector3DFeaturesWithIndividualFeature(nn.Module):

    def __init__(
        self, 
        n_in: int, 
        n_out: int, 
        n_vols: int, 
        n_pts: int = 200, 
        feat_res: int = 16, 
        n_ind: int = 18, 
        **kwargs
    ):
        super().__init__()
        self.n_ind = n_ind
        ind_feat = torch.randn(n_vols, n_pts, n_ind) * 0.1
        feat = torch.randn(n_vols, (n_out-n_ind)//3, feat_res, 3) * 0.1

        self.register_parameter(
            'feature',
            nn.Parameter(feat, requires_grad=True)
        )
        self.register_parameter(
            'ind_feature',
            nn.Parameter(ind_feat, requires_grad=True)
        )
        self.mapper = nn.Sequential(
            ParallelLinear(n_vols, n_out-n_ind, 32),
            nn.ReLU(inplace=True),
            ParallelLinear(n_vols, 32, n_out-n_ind),
        )
    
    def forward(self, pts: torch.Tensor, need_hessian: bool = False, *args, **kwargs):
        vol_feat = factorize_grid_sample(
            self.feature, 
            pts, 
            training=self.training, 
            need_hessian=need_hessian
        )
        vol_feat = rearrange(vol_feat, 'j f p c -> p j (f c)')
        vol_feat = rearrange(self.mapper(vol_feat), 'p j c -> j p c')

        if self.n_ind > 0:
            vol_feat = torch.cat([self.ind_feature, vol_feat], dim=-1)

        return vol_feat

    def reset_parameters(self):
        pass

class Vector3DFeaturesLatent(nn.Module):

    def __init__(
        self, 
        n_in: int = 32, 
        n_out: int = 36, 
        n_vols: int = 19, 
        n_pts: int = 200, 
        feat_res: int = 16, 
        n_ind: int = 0, 
        without_map: bool = False, 
        **kwargs
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_vols = n_vols
        self.feat_res = feat_res
        self.n_ind = n_ind
        self.without_map = without_map

        latent_feat = torch.randn(1, n_vols, n_in) * 0.1
        ind_feat = torch.randn(n_vols, n_pts, n_ind) * 0.1

        self.register_parameter(
            'latent_feature',
            nn.Parameter(latent_feat, requires_grad=True)
        )
        self.register_parameter(
            'ind_feature',
            nn.Parameter(ind_feat, requires_grad=True)
        )
        self.latent_to_feat = nn.Sequential(
            ParallelLinear(n_vols, n_in, (n_out - n_ind) * feat_res),
        )
        if not self.without_map:
            self.mapper = nn.Linear(n_out - n_ind, n_out - n_ind)
    
    def forward(self, pts: torch.Tensor, need_hessian: bool = False, *args, **kwargs):

        feat_res = self.feat_res

        feature = self.latent_to_feat(self.latent_feature)
        feature = rearrange(feature, '() j (f r c) -> j f r c', r=feat_res, c=3)

        vol_feat = factorize_grid_sample(
            feature, 
            pts, 
            training=self.training, 
            need_hessian=need_hessian
        )

        vol_feat = rearrange(vol_feat, 'j f p c -> p j (f c)')
        if not self.without_map:
            vol_feat = self.mapper(vol_feat)
        vol_feat = rearrange(vol_feat, 'p j c -> j p c')

        if self.n_ind > 0:
            vol_feat = torch.cat([self.ind_feature, vol_feat], dim=-1)

        return vol_feat

    def reset_parameters(self):
        pass
