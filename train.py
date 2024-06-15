import os
import glob
import hydra
import imageio
import numpy as np

import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm, trange
from omegaconf import OmegaConf

from core.trainer import Trainer

from hydra.utils import (
    instantiate,
)
from torch.utils.tensorboard import SummaryWriter

from core.load_data import * # TODO: temp name
from core.utils.evaluation_helpers import evaluate_metric

from typing import Mapping, Optional, Any, List
from omegaconf import DictConfig

CONFIG_BASE = 'configs/'


def prepare_render_data(
    render_data: Mapping[str, Any], 
    required: List[str] = ['c2ws', 'skts', 'bones', 'kp3d', 'hwf', 'center', 'cam_idxs', 'bg_idxs']
):
    # TODO: put this elsewhere
    render_tensor = {}
    for k, v in render_data.items():
        if k not in required:
            continue
        if v is None:
            continue

        # some special case: deal with them first
        if k == 'bg_idxs':
            bg_imgs = render_data['bgs'][v]
            # TODO: a little bit hacky ... put it on cpu instead of GPU
            render_tensor['bgs'] = torch.tensor(bg_imgs).cpu()
            continue

        if isinstance(v, np.ndarray):
            render_tensor[k] = torch.tensor(v)
        elif isinstance(v, tuple):
            render_tensor[k] = [torch.tensor(v_) for v_ in v]
        else:
            raise NotImplementedError(f'{k} is in unknown datatype')
    return render_tensor


def build_model(config: DictConfig, data_attrs: Mapping[str, Any], ckpt=None):
    # TODO: put this elsewhere

    n_framecodes = data_attrs["n_views"]
    # don't use dataset near far: will derive it from cyl anyway
    data_attrs.pop('far', None)
    data_attrs.pop('near', None)
    model = instantiate(config, **data_attrs, n_framecodes=n_framecodes, _recursive_=False)

    if ckpt is not None:
        ret = model.load_state_dict(ckpt['model'], strict=False)
        tqdm.write(f'ckpt loading: {ret}')

    return model


def find_ckpts(config: DictConfig, log_path: str, ckpt_path: Optional[str] = None):

    start = 0
    if ckpt_path is None and 'ckpt_path' in config:
        ckpt_path = config.get('ckpt_path')
    elif ckpt_path is None:
        ckpt_paths = sorted(glob.glob(os.path.join(log_path, '*.th')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
    
    if ckpt_path is None:
        tqdm.write(f'No checkpoint found: start training from scratch')
        return None, start
    
    ckpt = torch.load(ckpt_path)
    
    if "ckpt_path" in config:
        start = 0
    else:
        start = ckpt['global_iter']

    tqdm.write(f'Resume training from {ckpt_path}, starting from step {start}')
    return ckpt, start


def train(config: DictConfig):

    # create directory and save config
    expname, basedir = config.expname, config.basedir
    log_path = os.path.join(basedir, expname)
    os.makedirs(log_path, exist_ok=True)
    OmegaConf.save(config=config, f=os.path.join(log_path, 'config.yaml'))

    # tensorboard
    writer = SummaryWriter(log_path)

    # prepare dataset and relevant information
    data_info = build_dataloader(config)

    dataloader = data_info['dataloader']
    render_data = data_info['render_data']
    trainingviz_data = data_info['trainingviz_data']
    data_attrs = data_info['data_attrs']

    # build model
    ckpt, start= find_ckpts(config, log_path)
    model = build_model(config.model, data_attrs, ckpt)
    
    # override light direction and ambient values
    # model.light.direction = nn.Parameter(torch.Tensor(config.model.light_module.direction), requires_grad=config.model.light_module.optimize)
    # model.light.ambient = nn.Parameter(torch.Tensor(config.model.light_module.ambient), requires_grad=config.model.light_module.optimize)
    # model.light.floor_normal = torch.Tensor(config.model.light_module.floor_normal)
    # model.light.floor_point = torch.Tensor(config.model.light_module.floor_point)
    # print('light direction overwritten to:', model.light.direction)
    # print('light ambient overwritten to:', model.light.ambient)
    # print('light floor normal overwritten to:', model.light.floor_normal)
    # print('light floor point overwritten to:', model.light.floor_point)

    # let's print it out
    tqdm.write(str(model))
    tqdm.write(f"#parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer = Trainer(
        config=config.trainer, 
        loss_config=config.losses, 
        full_config=config,
        model=model, 
        ckpt=ckpt
    )

    # training loop
    start = start + 1
    data_iter = iter(dataloader)

    for i in trange(start, config.iters+1):

        batch = next(data_iter)
        #preds = model(to_device(batch, 'cuda'))
        training_stats = trainer.train_batch(batch, global_iter=i)

        # save periodically
        if (i % config.i_save) == 0:
            trainer.save_ckpt(global_iter=i, path=os.path.join(log_path, f'{i:07d}.th'))
        
        if (i % config.i_print) == 0:
            # logging
            trainer.save_ckpt(global_iter=i, path=os.path.join(log_path, f'latest.th'))
            to_print = ['total_loss', 'psnr', 'alpha', 'lr'] # things to print out
            mem = torch.cuda.max_memory_allocated() / 1024. / 1024.
            output_str = f'Iter: {i:07d}'
            for k, v in training_stats.items():
                if k in to_print:
                    output_str = f'{output_str}, {k}: {v:.6f}'
                # write to tensorboard
                writer.add_scalar(f'Stats/{k}', v, i)
            output_str = f'{output_str}, peak_mem: {mem:.6f}'
            tqdm.write(output_str)

            # now get visual summary like volume size etc
            if hasattr(model, 'get_summaries'):
                summaries = model.get_summaries(batch)
                for k, v in summaries.items():
                    data, data_type = v
                    if data_type in ['png', 'jpg']:
                        writer.add_image(k, data, i, dataformats='HWC')
                    elif data_type in ['scalar']:
                        writer.add_scalar(f'Stats/{k}', data, i)
        
        if (i % config.i_testset) == 0:
            tqdm.write('Running validation data ...')
            model.eval()
            render_tensor = prepare_render_data(render_data)
            # use a single GPU to validate. There's not much difference and we can save GPU memory
            preds = model(render_tensor, render_factor=config.render_factor, forward_type='render', raychunk=config.get('render_raychunk', 8192))
            model.train()

            metrics = evaluate_metric(
                preds[config.eval_key].to('cuda'),
                torch.tensor(render_data['imgs']),
                render_data['fgs'],
                render_tensor['c2ws'],
                render_tensor['kp3d'],
                render_data['hwf'],
                render_data['center'],
                render_factor=config.render_factor,
            )

            output_str = f'Iter: {i:07d}'
            for k, v in metrics.items():
                writer.add_scalar(f'Val/{k}', v, i)
                output_str = f'{output_str}, {k}: {v:.6f}'
            tqdm.write(output_str)

            writer.add_video("RGB_TestSet", preds['rgb_imgs'].permute(0, 3, 1, 2)[None], i, fps=2)
            if len(preds['depth_imgs']) > 0:
                writer.add_video("Depth_TestSet", preds['depth_imgs'].permute(0, 3, 1, 2)[None].expand(-1, -1, 3, -1, -1), i, fps=2)
            if len(preds['acc_imgs']) > 0:
                writer.add_video("Acc_TestSet", preds['acc_imgs'].permute(0, 3, 1, 2)[None].expand(-1, -1, 3, -1, -1), i, fps=2)
            if len(preds['disp_imgs']) > 0:
                writer.add_video("Disp_TestSet", preds['disp_imgs'].permute(0, 3, 1, 2)[None].expand(-1, -1, 3, -1, -1), i, fps=2)
            if len(preds['rgb_lit_imgs']) > 0:
                    writer.add_video("RGBLit_TestSet", preds['rgb_lit_imgs'].permute(0, 3, 1, 2)[None], i, fps=2)
            if len(preds['shadow_imgs']) > 0:
                writer.add_video("Shadow_TestSet", preds['shadow_imgs'].permute(0, 3, 1, 2)[None].expand(-1, -1, 3, -1, -1), i, fps=2)
            if len(preds['diffuse_imgs']) > 0:
                writer.add_video("Diffuse_TestSet", preds['diffuse_imgs'].permute(0, 3, 1, 2)[None].expand(-1, -1, 3, -1, -1), i, fps=2)
            if len(preds['irradiance_imgs']) > 0:
                writer.add_video("Irradiance_TestSet", preds['irradiance_imgs'].permute(0, 3, 1, 2)[None], i, fps=2)
            if len(preds['normal_imgs']) > 0:
                writer.add_video("Normal_TestSet", preds['normal_imgs'].permute(0, 3, 1, 2)[None], i, fps=2)
                writer.add_video("NormalViz_TestSet", preds['normal_viz_imgs'].permute(0, 3, 1, 2)[None], i, fps=2)

        if (i % config.i_trainingviz) == 0:
            tqdm.write('Running training visualization...')
            model.eval()
            render_tensor = prepare_render_data(trainingviz_data)
            # use a single GPU to validate. There's not much difference and we can save GPU memory
            preds = model(render_tensor, render_factor=config.render_factor, forward_type='render', raychunk=config.get('render_raychunk', 8192))
            model.train()
            writer.add_image("RGB_TrainViz", preds['rgb_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['rgb_lit_imgs']) > 0:
                writer.add_image("RGBLit_TrainViz", preds['rgb_lit_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['depth_imgs']) > 0:
                writer.add_image("Depth_TrainViz", preds['depth_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['acc_imgs']) > 0:
                writer.add_image("Acc_TrainViz", preds['acc_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['disp_imgs']) > 0:
                writer.add_image("Disp_TrainViz", preds['disp_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['shadow_imgs']) > 0:
                writer.add_image("Shadow_TrainViz", preds['shadow_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['diffuse_imgs']) > 0:
                writer.add_image("Diffuse_TrainViz", preds['diffuse_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['irradiance_imgs']) > 0:
                writer.add_image("Irradiance_TrainViz", preds['irradiance_imgs'].permute(0, 3, 1, 2)[0], i)
            if len(preds['normal_imgs']) > 0:
                writer.add_image("Normal_TrainViz", preds['normal_imgs'].permute(0, 3, 1, 2)[0], i)
                writer.add_image("NormalViz_TrainViz", preds['normal_viz_imgs'].permute(0, 3, 1, 2)[0], i)


@hydra.main(version_base='1.3', config_path=CONFIG_BASE, config_name='danbo_vof.yaml')
def cli(config: DictConfig):
    return train(config)

if __name__== '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.multiprocessing.set_start_method('spawn')
    cli()

