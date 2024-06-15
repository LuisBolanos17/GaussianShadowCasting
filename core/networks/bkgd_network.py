import torch
import torch.nn as nn
import torch.nn.functional as F
from core.networks.embedding import Optcodes
from core.positional_enc import PositionalEncoding

from typing import Tuple, Mapping, List, Union, Any


class BkgdNet(nn.Module):

    def __init__(
        self,
        input_size: int = 3,
        framecode_ch: int = 128,
        n_framecodes: int = 4,
        W: int = 256,
        img_res: Union[Tuple, List] = (1000, 1000),
        delay: int = 10000,
        warmup: int = 10000,
        **kwargs
    ):
        super().__init__()
        self.optcodes = Optcodes(n_framecodes, framecode_ch)

        self.posi_enc = PositionalEncoding(2, num_freqs=5)
        self.network = nn.Sequential(
            nn.Linear(framecode_ch+self.posi_enc.dims+input_size, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, 3)
        )
        self.img_res = img_res
        self.delay = delay
        self.warmup = warmup

        nn.init.normal_(self.network[-1].weight.data, 0, 0.001)
        if self.network[-1].bias is not None:
            nn.init.constant_(self.network[-1].bias.data, 0.)

    def forward(self, inputs: Mapping[str, Any], **kwargs):
        # 'bgs' is possibly perturbed for training purposes
        bg_perturbed = inputs['bgs']
        bg = inputs['bg_orig']
        pixel_locs = inputs['pixel_idxs']
        cam_idxs = inputs['cam_idxs']
        fgs = inputs['fgs']


        img_h, img_w = self.img_res
        pixel_h = pixel_locs[..., 0].float() / (img_h - 1) * 2. - 1.
        pixel_w = pixel_locs[..., 1].float() / (img_w - 1) * 2. - 1.
        pixel_locs = self.posi_enc(torch.stack([pixel_h, pixel_w], dim=-1))[0]

        framecode = self.optcodes(cam_idxs.reshape(-1, 1).long())

        input_feat = torch.cat([pixel_locs, bg, framecode], dim=-1)
        shade = self.network(input_feat)
        bg_preds = bg + shade
        #bg_preds = bg * (1 + torch.tanh(shade))
        if inputs['global_iter'] > (self.warmup + self.delay): 
            bg_preds = bg_preds #.detach()
            bgs = inputs['fgs'] * bg_preds + (1 - inputs['fgs']) * bg_perturbed
        else:
            bgs = bg_perturbed
        if inputs['global_iter'] < self.delay:
            bg_preds = bg_preds.detach()
        return {'bg_preds': bg_preds, 'bgs': bgs}
