import torch
import torch.nn as nn
import numpy as np

from hydra.utils import instantiate

from core.losses import *


def img2mse(img, target):
    return (img - target).pow(2.).mean()

def mse2psnr(mse):
    return  -10. * torch.log10(mse) / torch.log10(torch.Tensor([10.]))

def img2psnr(img, target):
    return mse2psnr(img2mse(img, target))

def to_device(data, device='cuda'):
    data_device = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            data_device[k] = v.to(device)
        else:
            data_device[k] = v
    return data_device

def get_lr_decay_fn(decay_type):
    if decay_type == 'standard':
        return decay_optimizer_lr
    elif decay_type == 'tava':
        return decay_optimizer_lr_delay
    elif decay_type == 'standard_delay':
        return decay_optimizer_lr_decay_delay
    else:
        raise ValueError(f'Unknown decay_type {decay_type}')

def decay_optimizer_lr(
        init_lr, 
        decay_steps,
        decay_rate, 
        optimizer,
        global_step=None, 
        group_scale=[1.0, 1.0],
        **kwargs,
    ):

    optim_step = global_step

    new_lrate = init_lr * (decay_rate ** (optim_step / decay_steps))
    for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lrate * group_scale[j]
    return new_lrate, None

def decay_optimizer_lr_decay_delay(
        init_lr,
        decay_steps,
        decay_rate, 
        optimizer,
        global_step=None, 
        delay_steps=0,
        group_scale=[1.0, 1.0],
        **kwargs,
    ):
    optim_step = max(global_step  - delay_steps, 0)
    new_lrate = init_lr * (decay_rate ** (optim_step / decay_steps))
    for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lrate * group_scale[j]
    return new_lrate, None

def decay_optimizer_lr_delay(
        init_lr,
        decay_steps,
        decay_rate,
        optimizer,
        delay_steps=0,
        delay_mult=0.01,
        global_step=None,
        group_scale=[1.0, 0.1],
        **kwargs,
    ):
    """ From TAVA
    """
    assert global_step is not None
    optim_step = global_step
    if delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = delay_mult + (1 - delay_mult) * np.sin(
            0.5 * np.pi * np.clip(optim_step / delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(optim_step / decay_steps, 0, 1)
    log_lerp = np.exp(np.log(init_lr) * (1 - t) + np.log(init_lr * decay_rate) * t)
    new_lrate = delay_rate * log_lerp
    for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lrate * group_scale[j]
    return new_lrate, None

@torch.no_grad()
def get_gradnorm(module):
    total_norm = 0.0
    max_norm = 0.0
    cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        # if torch.isnan(p.grad.data).any():
        #     print("grad is nan!!!")
        #     import pdb; pdb.set_trace()
        #     print()
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        # max_norm = max(max_norm, param_norm.item())
        cnt += 1
    avg_norm = (total_norm / cnt) ** 0.5
    total_norm = total_norm ** 0.5
    #print("total norm: {}, avg_norm: {}, max_norm: {}".format(total_norm, avg_norm, max_norm))
    return total_norm, avg_norm

class Trainer(object):
    """ For training models
    """

    def __init__(
        self, 
        config,
        loss_config,
        full_config,
        model,
        ckpt=None,
        **kwargs,
    ):
        self.config = config
        self.loss_config = loss_config
        self.full_config = full_config
        self.model = nn.DataParallel(model)

        # initialize optimizerni
        self.init_optimizer(ckpt)

        # initialize loss function
        self.init_loss_fns()

    def init_optimizer(self, ckpt=None):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        fix_body = self.config.get('fix_body_model', False)
        if fix_body:
            print('fixing body model')

        if hasattr(model, 'opt_param_groups'):
            model_params = model.opt_param_groups()
            self.optimizer = instantiate(self.config.optim, params=model_params[0]['params'])
            if len(model_params) == 2:
                self.optimizer.add_param_group(model_params[1])
            elif len(model_params) > 2:
                raise ValueError('Only support 2 param groups')
        else:
            if fix_body:
                model_params = model.light.parameters()
            else:
                model_params = model.parameters()
            self.optimizer = instantiate(self.config.optim, params=model_params)

        overwriting_model = 'ckpt_path' in self.full_config
        if ckpt is not None and not overwriting_model:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        
        if overwriting_model:
            print("loading from a specific ckpt_path, not loading optimizer from it!")

        self.optimizer.zero_grad()
        self.decay_fn = get_lr_decay_fn(self.config.lr_sched.decay_type)
    
    def update_optimizer(self, global_iter):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        self.optimizer = instantiate(self.config.optim, params=model.parameters())
        # update LR
        self.optimizer.zero_grad()
        new_lr, _ = self.decay_fn(
            init_lr=self.config.optim.lr,
            optimizer=self.optimizer,
            global_step=global_iter,
            **self.config.lr_sched,
        )
    
    def init_loss_fns(self):
        self.loss_fns = [
            eval(k)(**v)
        for k, v in self.loss_config.items()]
    
    def train_batch(self, batch, global_iter=1):

        device_cnt = 1
        if isinstance(self.model, nn.DataParallel):
            if len(self.model.device_ids) > 1:
                device_cnt = len(self.model.device_ids)

        # Step 1. model prediction
        batch = to_device(batch, 'cuda')
        batch['N_unique'] = self.full_config.N_sample_images // device_cnt
        batch['device_cnt'] = device_cnt
        batch['global_iter'] = global_iter
        if hasattr(self.model.module, 'rest_heads'):
            batch['rest_heads'] = self.model.module.rest_heads
        if hasattr(self.model.module, 'bone_middles'):
            batch['bone_middles'] = self.model.module.bone_middles
        if hasattr(self.model.module, 'rest_pose'):
            batch['rest_pose'] = self.model.module.rest_pose
        if hasattr(self.model.module, 'light') and hasattr(self.model.module.light, 'ambient'):
            batch['ambient'] = self.model.module.light.ambient
        preds = self.model(batch, pose_opt=self.config.get('pose_opt', False))

        # Step 2. compute loss
        # TODO: used to have pose-optimization here ..
        loss, stats = self.compute_loss(batch, preds, global_iter=global_iter)

        # clean up after step
        loss.backward()
        self.optimizer.step()
        total_norm, avg_norm = get_gradnorm(self.model)
        self.optimizer.zero_grad()

        # Step 3. post-update stuff

        # change/renew optimizer if needed

        # change learning rate
        new_lr, _ = self.decay_fn(
            init_lr=self.config.optim.lr,
            optimizer=self.optimizer,
            global_step=global_iter,
            **self.config.lr_sched,
        )
        stats.update(lr=new_lr)
        stats.update(avg_norm=avg_norm)

        # TODO: A-NeRF cutoff update
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, 'update_config'):
            need_optim_update = model.update_network(global_iter=global_iter)
            if isinstance(self.model, nn.DataParallel):
                self.model = nn.DataParallel(model)
            if need_optim_update:
                self.update_optimizer(global_iter=global_iter)
            self.model.train() # somehow training flag is changed..
        
        reset_rgb_head_iter = self.config.get('reset_rgb_head_iter', None)
        if reset_rgb_head_iter and reset_rgb_head_iter == global_iter:
            model.reset_rgb_head()
        fix_danbo_box_params = self.config.get('fix_danbo_box_params', None)
        if fix_danbo_box_params and fix_danbo_box_params <= global_iter:
            model.fix_danbo_box_params()

        return stats
    
    def compute_loss(self, batch, preds, global_iter=1):

        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        total_loss = torch.tensor(0.0)
        loss_stats = {}

        for key in preds.keys():
            if torch.isnan(preds[key]).any():
                print('nan prediction!')
                print(key)
                import pdb; pdb.set_trace()
                print()

        for loss_fn in self.loss_fns:
            loss, loss_stat = loss_fn(batch, preds, global_iter=global_iter, model=model)
            if torch.isnan(loss):
                print('loss is nan!')
                print(loss_stat)
                import pdb; pdb.set_trace()
                print()
            else:
                total_loss += loss
                loss_stats.update(**loss_stat)
        
        # get extra stats that's irrelevant to loss
        loss_stats.update(psnr=img2psnr(preds[self.config.psnr_key], batch['target_s']).item())
        if 'rgb0' in preds:
            loss_stats.update(psnr0=img2psnr(preds['rgb0'], batch['target_s']).item())
        loss_stats.update(alpha=preds['acc_map'].mean().item())
        if 'acc_map0' in preds:
            loss_stats.update(alpha0=preds['acc_map0'].mean().item())
        loss_stats.update(total_loss=total_loss.item())
        
        return total_loss, loss_stats
    
    def save_ckpt(self, global_iter, path):

        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_iter': global_iter,
            },
            path,
        )
        