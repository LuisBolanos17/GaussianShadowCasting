import torch
import torch.nn as nn
from core.utils.skeleton_utils import *
from typing import Optional, Union, List


class Optcodes(nn.Module):

    def __init__(
        self, 
        n_codes: int, 
        code_ch: int, 
        idx_map: Optional[Union[np.ndarray, torch.Tensor]] = None, 
        transform_code: bool = False, 
        mean: Optional[float] = None, 
        std: Optional[float] = None
    ):
        super().__init__()
        self.n_codes = n_codes
        self.code_ch = code_ch
        self.codes = nn.Embedding(n_codes, code_ch)
        self.transform_code = transform_code
        self.idx_map = None
        if idx_map is not None:
            self.idx_map = torch.LongTensor(idx_map)
        self.init_parameters(mean, std)

    def forward(self, idx: torch.Tensor, t: Optional[torch.Tensor] = None, *args, **kwargs):

        shape = idx.shape[:-1]
        if self.idx_map is not None:
            idx = self.idx_map[idx.long()].to(idx.device)
        if not self.training and idx.max() < 0:
            codes = self.codes.weight.mean(0, keepdims=True).expand(len(idx), -1)
        else:
            if idx.shape[-1] != 1:
                codes = self.codes(idx[..., :2].long()).squeeze(1)
                w = idx[..., 2]
                # interpolate given mixing weights
                codes = torch.lerp(codes[..., 0, :], codes[..., 1, :], w[..., None])
            else:
                if idx.max() > self.n_codes:
                    idx = idx.clamp(max=self.n_codes-1)
                    print('Warning! Out-of-range index detected in Optcodes input. Clamp it to self.n_codes-1')
                    print('Check the code if this is not expected')
                codes = self.codes(idx.long()).squeeze(1)

        if self.transform_code:
            codes = codes.view(t.shape[0], 4, -1).flatten(start_dim=-2)
        return codes

    def init_parameters(self, mean: Optional[float] = None, std: Optional[float] = None):
        if mean is None:
            nn.init.xavier_normal_(self.codes.weight)
            return

        if std is not None and std > 0.:
            nn.init.normal_(self.codes.weight, mean=mean, std=std)
        else:
            nn.init.constant_(self.codes.weight, mean)


class CondLatent(nn.Module):

    def __init__(
        self,
        n_codes: int = 300, 
        code_ch: int = 128,
    ):
        super().__init__()
        self.code = Optcodes(n_codes, code_ch)
