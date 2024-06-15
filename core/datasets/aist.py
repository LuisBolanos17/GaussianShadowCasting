import numpy as np
import h5py

from core.utils.skeleton_utils import *
from core.datasets import BaseH5Dataset


class AISTDataset(BaseH5Dataset):
    N_render = 10
    render_skip = 100

    training_frames = {
        'd04': np.arange(int(1584 * 0.9))[::2],
        'd08': np.arange(int(1728 * 0.9))[::2],
        'd12': np.arange(int(576 * 0.9)),
        'd16': np.arange(int(2016 * 0.9))[::3],
        'd20': np.arange(int(1469 * 0.9))[::2],
    }

    eval_idxs = {
        'd04': np.arange(1584)[-45:-10],
        'd08': np.arange(1728)[-45:-10],
        'd12': np.arange(576)[-45:-10],
        'd16': np.arange(2016)[-45:-10],
        'd20': np.arange(1469)[-45:-10],
    }

    def init_len(self):
        if self.split == 'train':
            self.data_len = len(self._idx_map)
            print(f'data_len {self.data_len}')
            # dataset organized as:
            # (N_views, N_frames)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                self.data_len = len(f['imgs'])

    def init_meta(self):
        super(AISTDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]

        if self.split == 'train':
            assert self.subject is not None
            data_len = len(dataset['imgs'])
            n_views = 3 # TODO: d16 and d20 is somehow different
            #n_views len(dataset['c2ws'])
            n_frames_per_views = data_len // n_views

            frame_ids = self.training_frames[self.subject]
            idx_map = np.concatenate(
                [frame_ids + v * n_frames_per_views for v in range(n_views)] 
            )
            self._idx_map = idx_map
        dataset.close()
    
    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.kp_idxs[idx], q_idx
    
    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.cam_idxs[idx], q_idx

    def _get_subset_idxs(self, render=False):
        '''
        get the part of data that you want to train on
        '''
        if self._idx_map is not None:
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))
        else:
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(self._N_total_img)
            _c_idxs = _cq_idxs = np.arange(self._N_total_img)

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

    def get_render_data(self):
        
        assert self.subject is not None
        eval_idxs = self.eval_idxs[self.subject]
        h5_path = self.h5_path

        H, W = self.HW
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][eval_idxs].reshape(-1, H, W, 1).astype(np.float32)
        render_bgs = np.ones_like(render_imgs[:1]).astype(np.float32)
        render_bg_idxs = np.zeros(len(eval_idxs)).astype(np.int64)

        kp3d = dataset['kp3d'][eval_idxs]
        skts = dataset['skts'][eval_idxs]
        bones = dataset['bones'][eval_idxs]

        c_idxs = dataset['img_pose_indices'][eval_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        dataset.close()

        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': c_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[c_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': np.arange(len(eval_idxs)),
            'kp_idxs_len': len(kp3d),
            'kp3d': kp3d,
            'skts': skts,
            'bones': bones,
        }
        return render_data
