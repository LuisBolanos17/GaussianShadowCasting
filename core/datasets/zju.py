import os
import h5py
import numpy as np

from core.datasets import BaseH5Dataset
from core.utils.skeleton_utils import *


class ZJUMocapDataset(BaseH5Dataset):

    N_render = 15
    render_skip = 63

    def __init__(self, h5_path, *args, halfres=False, load_cal=False, **kwargs):
        self.basedir = os.path.dirname(h5_path)
        self.halfres = halfres
        self.load_cal = load_cal
        if self.halfres:
            h5_path = h5_path.replace('.h5', '_halfres.h5')

        super(ZJUMocapDataset, self).__init__(h5_path, *args, **kwargs)

    def init_meta(self):
        if self.split == 'test':
            self.h5_path = self.h5_path.replace('train', 'test')
        super(ZJUMocapDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]

        if self.split == 'test':
            n_unique_cam = len(np.unique(self.cam_idxs))
            self.kp_idxs = self.kp_idxs // n_unique_cam

        print('WARNING: ZJUMocap does not support pose refinement for now (_get_subset_idxs is not implemented)')
        dataset.close()

        if self.load_cal:
            cal_c2ws = np.load(os.path.join(self.basedir, f'{self.subject}_cal.npy'))
            self.c2ws = cal_c2ws.copy()
            import pdb; pdb.set_trace()
            print

    def get_meta(self):
        data_attrs = super(ZJUMocapDataset, self).get_meta()
        return data_attrs

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

    def get_img_data(self, idx, pixel_idxs, *args, **kwargs):
        '''
        get image data (in np.uint8), convert to float
        '''

        if self.read_full_img:
            full_fg = self.dataset['masks'][idx].astype(np.float32)
            fg = full_fg[pixel_idxs]
            fg_idx = np.where(full_fg[..., 0] > 0)[0]
            full_img = np.zeros((np.prod(self.HW), 3), dtype=np.float32)
            full_img[fg_idx] = self.dataset['imgs'][idx, fg_idx] / 255.
        else:
            fg = self.dataset['masks'][idx, pixel_idxs].astype(np.float32)
            full_img = None
            full_fg = None

        img = self.dataset['imgs'][idx, pixel_idxs].astype(np.float32) / 255.

        bg, bg_orig = None, None
        img_not_masked = img.copy()
        if self.has_bg:
            bg_idx = self.bg_idxs[idx]
            bg_orig = self.bgs[bg_idx, pixel_idxs].astype(np.float32) / 255.

            if self.perturb_bg:
                noise = np.random.random(bg_orig.shape).astype(np.float32)
                # do not perturb foreground area
                # also, force the fill-in background to be black
                bg = (1 - fg) * noise + fg * bg_orig * 0.
            else:
                bg = bg_orig

            if self.mask_img:
                img = img * fg + (1. - fg) * bg

        return img, fg, bg, img_not_masked, bg_orig, full_img, full_fg
    
    def __getitem__(self, *args, **kwargs):
        ret = super().__getitem__(*args, **kwargs)
        # TODO: fix this
        real_cam_idxs = ret['cam_idxs'] %4 # % 23
        ret['real_cam_idx'] = real_cam_idxs
        return ret
    
    def get_render_data(self):
        zju_eval_frames = {
            387: np.arange(418)[:19],
            392: np.arange(361)[:19],
        }

        eval_idxs = zju_eval_frames[self.subject]

        if not self.halfres:
            h5_path = self.h5_path.replace('train', 'novel_pose')
        else:
            h5_path = self.h5_path.replace('train_halfres', 'novel_pose')

        # half resolution
        H, W = 512, 512
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][eval_idxs].reshape(-1, H, W, 1).astype(np.float32)

        bgs = dataset['bkgds'][:]
        bkgd_idxs = dataset['bkgd_idxs'][eval_idxs]
        render_bgs = 0. * bgs[bkgd_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bg_idxs = np.zeros(len(eval_idxs)).astype(np.int64)

        kp_idxs = dataset['kp_idxs'][eval_idxs]

        kp3d, skts, bones = [], [], []
        for i in kp_idxs:
            kp3d.append(dataset['kp3d'][i])
            skts.append(dataset['skts'][i])
            bones.append(dataset['bones'][i])
        kp3d = np.array(kp3d)
        skts = np.array(skts)
        bones = np.array(bones)

        c_idxs = dataset['img_pose_indices'][eval_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, dataset['focals'][:][c_idxs].astype(np.float32))
        center = dataset['centers'][:][c_idxs].copy().astype(np.float32)

        c2ws = dataset['c2ws'][:][c_idxs].astype(np.float32)
        dataset.close()
        if self.load_cal:
            cal_c2ws = np.load(os.path.join(self.basedir, f'{self.subject}_cal.npy'))
            c2ws = cal_c2ws[c_idxs]

        render_data = {
            #'imgs': render_imgs,
            'imgs': render_imgs * render_fgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': c_idxs * 0 - 1, # set to -1 to use avg framecode
            'cam_idxs_len': len(self.c2ws),
            'c2ws': c2ws,
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


class ZJUH36MDataset(ZJUMocapDataset):

    N_render = 30
    render_skip = 1

    def __init__(self, *args, pose_skip=None, color_bg=False, **kwargs):
        self.pose_skip = pose_skip
        self.color_bg = color_bg
        super(ZJUH36MDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        if self.split == 'test':
            self.h5_path = self.h5_path.replace('train', 'test')
        super(ZJUH36MDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]

        dataset.close()
    
    def init_temporal_validity(self):
        temp_val = np.ones((len(self.kp3d),)).astype(np.float32)
        temp_val[0] = 0
        temp_val[-1] = 0
        return temp_val
    
    def get_render_data(self):

        h36m_zju_eval_frames = {
            'S1': np.arange(34),
            'S5': np.arange(64),
            'S6': np.arange(39),
            'S7': np.arange(84),
            'S8': np.arange(57),
            'S9': np.arange(67),
            'S11': np.arange(48),
        }

        eval_idxs = h36m_zju_eval_frames[self.subject]
        h5_path = self.h5_path.replace('train', 'anim')

        H, W = self.HW
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][eval_idxs].reshape(-1, H, W, 1).astype(np.float32)
        bgs = dataset['bkgds'][:]
        bkgd_idxs = dataset['bkgd_idxs'][eval_idxs]

        render_bgs = 0. * bgs[bkgd_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
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
            'imgs': render_imgs * render_fgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': c_idxs * 0 - 1, # set to -1 to use avg framecode
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
    