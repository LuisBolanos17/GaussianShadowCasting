import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from copy import deepcopy
from omegaconf import OmegaConf
from hydra.utils import instantiate

from typing import Mapping, Any
from omegaconf import DictConfig


def ray_collate_fn(batch: Mapping[str, Any]):
    for k, v in batch[0].items():
        if v is None:
            print(f'{k}: is None')
    batch = default_collate(batch)
    # default collate results in shape (N_images, N_rays_per_images, ...)
    # flatten the first two dimensions.
    batch = {k: batch[k].flatten(end_dim=1) for k in batch}
    batch['rays'] = torch.stack([batch['rays_o'], batch['rays_d']], dim=0)
    return batch


def build_dataloader(config: DictConfig):
    config = deepcopy(config)
    OmegaConf.set_struct(config, False)

    # readout args
    iters = config.get('iters', 300000)
    num_workers = config.get('num_workers', 16)
    N_sample_images = config.get('N_sample_images', 16)
    N_rays = config.get('N_rays', 3072)
    N_samples = N_rays // N_sample_images


    assert N_samples <= N_rays, 'N_sample_images needs to be smaller than N_rand!'

    # instantiate dataset
    if isinstance(config.dataset, str):
        dataset_config = OmegaConf.load(config.dataset)
    else:
        dataset_config = config.dataset
    dataset = instantiate(dataset_config, N_samples=N_samples)

    # TODO: config these in a different way?
    data_attrs = dataset.get_meta()
    render_data = dataset.get_render_data()
    trainingviz_data = dataset.get_trainingviz_data(config.get('trainingviz_frame', 0))

    sampler = instantiate(
        config.sampler,
        data_source=dataset, 
        N_iter=iters + 10, 
        N_images=N_sample_images
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        num_workers=num_workers,
        collate_fn=ray_collate_fn,
        pin_memory=True,
    )

    return {
        'dataloader': dataloader,
        'data_attrs': data_attrs,
        'render_data': render_data,
        'trainingviz_data': trainingviz_data
    }
