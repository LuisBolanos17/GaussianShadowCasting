import cv2
import time

import h5py

import numpy as np

from core.datasets import BaseH5Dataset
from core.utils.skeleton_utils import *


class NeuralActorDataset(BaseH5Dataset):
    render_skip = 30
    N_render = 15

    def __init__(self, *args, discard_border=True, **kwargs):
        self.discard_border = discard_border
        super().__init__(*args, **kwargs)

    def init_dataset(self):

        if self.dataset is not None:
            return
        cache_size = int(np.prod(self.HW) * 4)
        print(f'init dataset with cache_size: {cache_size}')

        self.dataset = h5py.File(self.h5_path, 'r', rdcc_nbytes=cache_size)



    def __getitem__(self, q_idx):
        '''
        q_idx: index queried by sampler, should be in range [0, len(dataset)].
        Note - self._idx_map maps q_idx to indices of the sub-dataset that we want to use.
               therefore, self._idx_map[q_idx] may not lie within [0, len(dataset)]
        '''

        if self._idx_map is not None:
            idx = self._idx_map[q_idx]
        else:
            idx = q_idx


        # TODO: map idx to something else (e.g., create a seq of img idx?)
        # or implement a different sampler
        # as we may not actually use the idx here

        if self.dataset is None:
            self.init_dataset()

        # get camera information
        c2w, K, focal, center, cam_idxs = self.get_camera_data(idx, q_idx, self.N_samples)

        # get kp index and kp, skt, bone, cyl
        kp_idxs, kps, bones, skts, cyls = self.get_pose_data(idx, q_idx, self.N_samples)

        ## sample pixels
        #pixel_idxs, fg, sampling_mask = self.sample_pixels(idx, q_idx)

        # load the image, foreground and background,
        # and get values from sampled pixels
        #rays_rgb, fg, bg = self.get_img_data(idx, pixel_idxs, fg=fg)
        # different from other loader: directly read everything at once
        #time0 = time.time()
        pixel_idxs, rays_rgb, fg, bg, rgb_not_masked, bg_orig = self.sample_pixels(idx, q_idx)
        #print(f'fetch img data time {time.time()-time0}')

        # maybe get a version that computes only sampled points?
        rays_o, rays_d = self.get_rays(c2w, focal, pixel_idxs, center)

        H, W = self.HW
        pixel_idxs = np.stack([pixel_idxs // W, pixel_idxs % W],axis=-1) 
        return_dict = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'target_s': rays_rgb,
            'target_s_not_masked': rgb_not_masked,
            'kp_idx': kp_idxs,
            'kp3d': kps,
            'bones': bones,
            'skts': skts,
            'cyls': cyls,
            'cam_idxs': cam_idxs,
            #'cam_idxs': cam_idxs + q_idx, # TODO: figure out what makes h36m faster
            'fgs': fg,
            'bgs': bg,
            'bgs_orig': bg_orig,
            'pixel_idxs': pixel_idxs,
            'c2ws': c2w[None],
            'K': K[None],
        }

        return return_dict

    def sample_pixels(self, idx, q_idx):
        '''
        return sampled pixels (in (H*W,) indexing, not (H, W))
        '''
        p = self.patch_size
        N_rand = self.N_samples // int(p**2)
        # TODO: check if sampling masks need adjustment
        # assume sampling masks are of shape (N, H, W, 1)
        #img_data = self.dataset['imgs'][idx]
        #img = img_data[..., :3]
        #fg = img_data[..., 3:]

        #time0 = time.time()
        fg = self.dataset['imgs'][idx, ..., 3:]
        #print(f'read fg time {time.time()-time0}')

        full_img = full_fg = None
        if self.read_full_img:
            full_img = np.zeros((np.prod(self.HW), 3), dtype=np.float32)
            full_img[full_fg[..., 0] > 0] = self.dataset['imgs'][idx][full_fg[..., 0] > 0] / 255.

        # create sampling mask by dilation
        time0 = time.time()
        d_kernel = np.ones((5, 5))
        H, W = self.HW
        sampling_mask = cv2.dilate(
            fg.reshape(H, W, 1), 
            kernel=d_kernel,
            iterations=3,
        )[..., None]

        if self.discard_border:
            dilated = cv2.dilate(fg.reshape(H, W, 1), kernel=d_kernel)[..., None]
            eroded = cv2.erode(fg.reshape(H, W, 1), kernel=d_kernel)[..., None]
            sampling_mask[(dilated - eroded) == 1] = 0 
        sampling_mask = sampling_mask.reshape(-1)

        valid_idxs, = np.where(sampling_mask>0)
        if len(valid_idxs) == 0 or len(valid_idxs) < N_rand:
            valid_idxs = np.arange(len(sampling_mask))

        sampled_idxs = np.random.choice(
            valid_idxs,
            N_rand,
            replace=False
        )

        sampled_idxs = np.sort(sampled_idxs)
        #print(f'sample time {time.time()-time0}')
        # rays_rgb = img[sampled_idxs].astype(np.float32) / 255.
        #time0 = time.time()
        rays_rgb = self.dataset['imgs'][idx, sampled_idxs, :3].astype(np.float32) / 255.
        fg = fg[sampled_idxs].astype(np.float32).copy()
        #print(f'get sample time {time.time()-time0}')

        bg, bg_orig = None, None
        rgb_not_masked = rays_rgb.copy()
        if self.has_bg:
            bg_idx = self.bg_idxs[idx]
            bg_orig = self.bgs[bg_idx, sampled_idxs].astype(np.float32) / 255.

            if self.perturb_bg:
                noise = np.random.random(bg_orig.shape).astype(np.float32)
                #noise= (1 - fg) * noise # do not perturb foreground area!
                bg = (1 - fg) * noise + fg * bg_orig# do not perturb foreground area!
                #bg = noise
            else:
                bg = bg_orig

            if self.mask_img:
                rays_rgb = rays_rgb * fg + (1. - fg) * bg

        return sampled_idxs, rays_rgb, fg, bg, rgb_not_masked, bg_orig, full_img, full_fg

    def init_meta(self):
        super(NeuralActorDataset, self).init_meta()

        self.has_bg = True
        self.bgs = 255 * np.ones((1, np.prod(self.HW), 3), dtype=np.uint8)
        self.bg_idxs = np.zeros((self._N_total_img,), dtype=np.int64) * 0

        self.kp_map = None
        self.cam_map = None

        # load kp / camera map because the data is pre-shuffled
        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        self.kp_map = dataset['kp_idxs'][:]
        self.cam_map = dataset['img_pose_indices'][:]
        self.unique_cam = len(np.unique(self.cam_map))
        dataset.close()

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        # TODO: check if this is right
        if self.kp_map is None:
            return idx // len(self.kp3d), q_idx // len(self.kp3d)
        
        # only support 'no subset case'
        #assert (idx == q_idx).all()
        ret_idx = self.kp_map[idx].copy()
        return ret_idx, ret_idx
    
    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''

        if self.cam_map is None:
            return idx % len(self.kp3d), q_idx % len(self.kp3d)

        # only support 'no subset case'
        #assert (idx == q_idx).all()
        #assert (idx == q_idx)
        ret_idx = self.cam_map[idx].copy()
        return ret_idx, idx % self.unique_cam

    def _get_subset_idxs(self, render=False):
        '''return idxs for the subset data that you want to train on.
        Returns:
        k_idxs: idxs for retrieving pose data from .h5
        c_idxs: idxs for retrieving camera data from .h5
        i_idxs: idxs for retrieving image data from .h5
        kq_idxs: idx map to map k_idxs to consecutive idxs for rendering
        cq_idxs: idx map to map c_idxs to consecutive idxs for rendering
        '''
        if self._idx_map is not None:
            # queried_idxs
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))

        else:
            # queried == actual index
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(self._N_total_img)
            _c_idxs = _cq_idxs = np.arange(self._N_total_img)

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

    def get_meta(self):
        data_attrs = super(NeuralActorDataset, self).get_meta()
        data_attrs['n_views'] = self._N_total_img // len(self.kp3d)
        #data_attrs['n_views'] =  len(self.kp3d)
        return data_attrs

    def get_render_data(self):

        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs(render=True)

        # grab only a subset (15 images) for rendering
        kq_idxs = kq_idxs[::self.render_skip][:self.N_render]
        cq_idxs = cq_idxs[::self.render_skip][:self.N_render]
        i_idxs = i_idxs[::self.render_skip][:self.N_render]
        k_idxs = k_idxs[::self.render_skip][:self.N_render]
        c_idxs = c_idxs[::self.render_skip][:self.N_render]

        # get images if split == 'render'
        # note: needs to have self._idx_map
        H, W = self.HW
        img_data = dataset['imgs'][i_idxs].reshape(-1, H, W, 4).astype(np.float32)
        #dataset['imgs'][i_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        #render_fgs = dataset['masks'][i_idxs].reshape(-1, H, W, 1)
        render_imgs = img_data[..., :3] / 255.
        render_fgs = img_data[..., 3:].reshape(-1, H, W, 1)
        render_bgs = self.bgs.reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bg_idxs = self.bg_idxs[i_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])

        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        
        # TODO: c_idxs, k_idxs ... confusion

        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': cq_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[c_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': k_idxs,
            'kp_idxs_len': len(self.kp3d),
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
            'bones':self.bones[k_idxs],
        }

        dataset.close()

        return render_data
