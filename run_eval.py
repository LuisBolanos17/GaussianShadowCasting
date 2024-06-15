import os
import cv2
import torch
import hydra
import imageio
import torch.nn as nn
import lpips

from core.trainer import to_device
from core.utils.skeleton_utils import *
from torch.utils.data import Dataset, DataLoader

from typing import Tuple, Optional, Any, Dict
from run_render import (
    load_trained_model,
)
from hydra.utils import instantiate
from skimage.metrics import structural_similarity as compare_ssim
from einops import rearrange
from omegaconf import OmegaConf

from tqdm import tqdm

CONFIG_BASE = 'configs/eval'


def img2imguint8(img: np.array):
    return (img * 255).clip(0, 255).astype(np.uint8)

def img2psnr(render, target):
    mse = np.mean((render - target)**2)
    return -10 * np.log(mse) / np.log(10)


@torch.no_grad()
def img2lpips(
    model: nn.Module,
    render: np.array,
    target: np.array,
    device: torch.device = 'cuda',
):
    render_tensor = rearrange(torch.tensor(render)[None], 'b h w c -> b c h w')
    target_tensor = rearrange(torch.tensor(target)[None], 'b h w c -> b c h w')
    render_tensor = (render_tensor * 2 - 1.).to(device)
    target_tensor = (target_tensor * 2 - 1.).to(device)
    return model(target_tensor, render_tensor).item()


def evaluate_psnr(
    render: np.array,
    target: np.array,
    mask: Optional[np.array] = None,
    bbox: Optional[np.array] = None,
    fg_only: bool = False,
    metric_key: str = 'psnr',
):

    metric_mask_key = f'{metric_key}_mask'
    metric_bbox_key = f'{metric_key}_bbox'
    results = {metric_key: None, metric_mask_key: None, metric_bbox_key: None}
    if not fg_only:
        results[metric_key] = img2psnr(render, target)

    if mask is not None:
        results[metric_mask_key] = img2psnr(render[mask[..., 0] > 0], target[mask[..., 0] > 0])

    if bbox is not None:
        x, y, w, h = bbox
        results[metric_bbox_key] = img2psnr(render[y:y+h, x:x+w], target[y:y+h, x:x+w])
        
    # filter out empty results
    return {k: v for k, v in results.items() if v is not None} 


def evaluate_ssims(
    render: np.array,
    target: np.array,
    bbox: Optional[np.array] = None,
    mask: Optional[np.array] = None,
    metric_key: str = 'ssim',
    fg_only: bool = False,
    data_range: float = 1.0,
    **kwargs,
):

    ssim_kwargs = {
        'data_range': data_range,
        'channel_axis': -1,
    }
    metric_bbox_key = f'{metric_key}_bbox'
    results = {metric_key: None, metric_bbox_key: None}
    if not fg_only:
        results[metric_key] = compare_ssim(render, target, **ssim_kwargs, **kwargs)

    if (bbox is None) and (mask is not None):
        mask = (mask.copy() * 255).clip(0, 255)
        bbox = cv2.boundingRect(mask.astype(np.uint8))

    if bbox is not None:
        x, y, w, h = bbox
        results[metric_bbox_key] = compare_ssim(render[y:y+h, x:x+w], target[y:y+h, x:x+w], **ssim_kwargs, **kwargs)
            
    # filter out empty results
    return {k: v for k, v in results.items() if v is not None} 


def evaluate_lpips(
    model: nn.Module,
    render: np.array,
    target: np.array,
    mask: Optional[np.array] = None,
    bbox: Optional[np.array] = None,
    metric_key: str = 'lpips',
    fg_only: bool = False,
    **kwargs,
):
    metric_bbox_key = f'{metric_key}_bbox'
    results = {metric_key: None, metric_bbox_key: None}

    if not fg_only:
        results[metric_key] = img2lpips(model, render, target, **kwargs)


    if (bbox is None) and (mask is not None):
        mask = (mask.copy() * 255).clip(0, 255)
        bbox = cv2.boundingRect(mask.astype(np.uint8))
    
    if bbox is not None:
        x, y, w, h = bbox

        results[metric_bbox_key] = img2lpips(
            model, 
            render[y:y+h, x:x+w], 
            target[y:y+h, x:x+w], 
            **kwargs
        )

    # filter out empty results
    return {k: v for k, v in results.items() if v is not None} 


class Evaluator():
    """ An evaluator that compute loss metrics for a given model.

    Assume input image range is [0, 1]
    """

    def __init__(
        self,
        use_lpips: bool = False,
        img_key: str = 'rgb_imgs',
        fg_only: bool = False,
        apply_mask: bool = True,
        num_workers: int = 8,
        write_target: bool = False,
        write_render: bool = True,
        eval_precomputed: bool = False,
        precomputed_path: Optional[str] = None,
    ):
        """
        use_lpips: bool - LPIPS metric
        img_key: str - key to evaluate the metrics for
        fg_only: bool - only compute metric on foreground pixels / inside of bounding boxes
        apply_mask: bool - to apply foreground mask on ground truth image for metric computation
        """
        self.use_lpips = use_lpips
        self.num_workers = num_workers
        self.img_key = img_key
        self.fg_only = fg_only
        self.apply_mask = apply_mask
        self.write_target = write_target
        self.write_render = write_render
        self.eval_precomputed = eval_precomputed
        self.precomputed_path = precomputed_path

        if self.eval_precomputed:
            assert precomputed_path is not None
        
        self.lpips = None
        if use_lpips:
            self.lpips = lpips.LPIPS(net='vgg').cuda()
    
    def compute_metrics(
        self,
        data: Dict[str, Any],
        preds: Dict[str, Any],
    ):
        metrics = {}
        rgb_pred = preds[self.img_key][0].cpu().numpy()
        rgb_target = data['imgs'][0].cpu().numpy()
        bg = data['bgs'][0].cpu().numpy()

        mask, bbox = None, None
        if 'masks' in data:
            mask = data['masks'][0].cpu().numpy()
        if 'bboxes' in data:
            bbox = data['bboxes'][0].cpu().numpy()
        
        if self.apply_mask:
            rgb_target = rgb_target * mask + bg * (1 - mask)

        metrics.update(**evaluate_psnr(
                rgb_pred, 
                rgb_target,
                mask=mask,
                bbox=bbox,
                fg_only=self.fg_only,
            )
        )
        metrics.update(**evaluate_ssims(
                rgb_pred, 
                rgb_target,
                mask=mask,
                bbox=bbox,
                fg_only=self.fg_only,
            )
        )
        if self.lpips is not None:
            metrics.update(**evaluate_lpips(
                    self.lpips,
                    rgb_pred,
                    rgb_target,
                    mask=mask,
                    bbox=bbox,
                    fg_only=self.fg_only,
                )
            )
        return metrics
    
    def evaluate(
        self,
        dataset: Dataset,
        model: nn.Module,
        device: torch.device = 'cuda',
        raychunk: int = 1024 * 10,
        savedir: Optional[str]  = None,
    ):
        dataloader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=1, # one image is a bunch of rays
            shuffle=False,
        )

        if model is not None:
            model.eval()
        metrics = []
        for i, data in enumerate(tqdm(dataloader)):
            data = to_device(data, device)

            if self.eval_precomputed:
                preds = {self.img_key: torch.tensor(imageio.imread(os.path.join(self.precomputed_path, f'{i:05d}.png')) / 255.)[None].float()}
            else:
                preds = model(data, forward_type='render', raychunk=raychunk)
            metrics.append(self.compute_metrics(data, preds))

            # save only when we are not using precomputed results
            if self.write_render and not self.eval_precomputed:
                assert savedir is not None
                renderdir = os.path.join(savedir, 'render')
                os.makedirs(renderdir, exist_ok=True)
                imageio.imwrite(os.path.join(renderdir, f'{i:05d}.png'), img2imguint8(preds[self.img_key][0].cpu().numpy()))

            if self.write_target:
                assert savedir is not None
                targetdir = os.path.join(savedir, 'target')
                os.makedirs(targetdir, exist_ok=True)
                imageio.imwrite(os.path.join(targetdir, f'{i:05d}.png'), img2imguint8(data['imgs'][0].cpu().numpy()))


        metrics = {k: np.array([m[k] for m in metrics]) for k in metrics[0].keys()}
        avg_metrics = {f'{k}_avg': np.mean(v) for k, v in metrics.items()}
        metrics.update(**avg_metrics)
        
        print('eval results:')
        for k, v in avg_metrics.items():
            print(f'{k} -- {v}')
        np.save(os.path.join(savedir, 'metrics.npy'), metrics, allow_pickle=True)
        print(f'Results are saved to {savedir}')
    

def evaluate(config):

    forward_type = config.get('forward_type', 'render')
    raychunk = config.get('raychunk', 1024 * 10)

    # create dataset
    dataset = instantiate(config.eval_dataset)

    # initialize evaluator
    evaluator = instantiate(config.evaluator)

    # load model
    model = None
    if config.model_config is not None:
        model_config = OmegaConf.load(config.model_config)
        model = load_trained_model(model_config, ckpt_path=config.get('ckpt_path', None))
    else:
        assert evaluator.eval_precomputed

    # ckpt = torch.load('assets/8gs.th', map_location='cpu')
    # model_ckpt = ckpt['model']
    # Gs = model_ckpt['visibility.Gs']
    # model.visibility.Gs = nn.Parameter(Gs.cuda())

    savedir = config.get('output_path', None)
    evaldir = config.get('eval_name', None)

    if savedir is None:
        assert config.model_config is not None
        savedir = os.path.join(
            os.path.dirname(config.model_config),
            evaldir,
        )
    os.makedirs(savedir, exist_ok=True)

    evaluator.evaluate(dataset, model, raychunk=raychunk, savedir=savedir)


@hydra.main(version_base='1.3', config_path=CONFIG_BASE, config_name='')
def cli(config):
    return evaluate(config)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.multiprocessing.set_start_method('spawn')
    cli()