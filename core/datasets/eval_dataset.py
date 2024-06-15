import os
import cv2
import h5py
import torch
import hydra
import imageio

from core.trainer import to_device
from core.utils.skeleton_utils import *
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from train import (
    build_model,
    find_ckpts,
)
from typing import Tuple, Optional, List
from run_render import (
    load_trained_model,
    BaseRenderDataset,
)
from hydra.utils import instantiate


CONFIG_BASE = 'configs/render'


class BaseEvalDataset(BaseRenderDataset):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(BaseEvalDataset, self).__init__(*args, idxs=None, **kwargs)

    def init_meta(self):
        super().init_meta()
        dataset = h5py.File(self.h5_path, 'r')
        self.skts = dataset['skts'][:].astype(np.float32)
        dataset.close()
        self.init_idx()
    
    def init_idx(self):
        """ Overwrite with your own evaluation rule.
        --> set self.idxs correctly
        """
        dataset = h5py.File(self.h5_path, 'r')
        self.idxs = np.arange(len(dataset['imgs']))
        dataset.close()

    def get_pose_data(self, idx):
        kp_idx = self.kp_idxs[idx]
        root_loc = self.root_locs[kp_idx]
        bone = self.bones[kp_idx]
        skt = self.skts[kp_idx]
        kp3d = self.kp3d[kp_idx]

        return {
            'kp3d': kp3d,
            'bones': bone,
            'skts': skt,
            'root_locs': root_loc,
        } 

    def get_img_data(self, idx):
        bkgd_idx = self.bg_idxs[idx]
        bg = self.bgs[bkgd_idx].reshape(*self.HW, 3) / 255.
        img = self.dataset['imgs'][idx].reshape(*self.HW, 3) / 255.
        mask = self.dataset['masks'][idx].reshape(*self.HW, 1)
        x, y, w, h = cv2.boundingRect(mask)
        return {
            'imgs': img.astype(np.float32),
            'masks': mask,
            'bgs': bg.astype(np.float32),
            'bboxes': np.array([x, y, w, h]).astype(np.int32),
        }
    
    def get_camera_data(self, idx):
        c2w, K, focal, center, cam_idx = super().get_camera_data(idx)

        if self.cam_overwrite is not None:
            cam_idx = cam_idx * 0 + self.cam_overwrite
        hwf = (*self.resolution, focal)

        return {
            'c2ws': c2w,
            'K': K, 
            'hwf': hwf,
            'focals': focal,
            'center': center,
            'cam_idxs': cam_idx,
        }


    def __getitem__(self, idx):
        if self.dataset is None:
            self.init_dataset()
        idx = self.idxs[idx] 
        pose_data = self.get_pose_data(idx)
        img_data = self.get_img_data(idx)
        cam_data = self.get_camera_data(idx)

        return {
            **img_data,
            **cam_data,
            **pose_data,
        }

