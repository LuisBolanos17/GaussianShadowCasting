import cv2
import time
import bisect
import h5py, math
import torch
import random
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from core.networks.pose_opt import pose_ckpt_to_pose_data
from core.utils.skeleton_utils import (
    SMPLSkeleton, 
    cylinder_to_box_2d, 
    nerf_c2w_to_extrinsic, 
    build_intrinsic,
)


class BaseH5Dataset(Dataset):

    # TODO: poor naming
    def __init__(
        self, 
        h5_path, 
        N_samples=96, 
        patch_size=1, 
        split='full',
        N_nms=0, 
        subject=None, 
        mask_img=False, 
        multiview=False,
        perturb_bg=False, 
        inverse_sampling=False, 
        bkgd_mul=1.0,
        read_full_img=False, 
        dilate_mask=False,
        full_perturb_bg=False, 
        **kwargs
    ):
        '''
        Base class for multi-proc h5 dataset

        args
        ----
        h5_path (str): path to .h5 file
        N_samples (int): number of pixels to sample from each image
        patch_size (int): sample patches of rays of this size.
        split (str): split to use. splits are defined in a dataset-specific manner
        N_nms (float): number of pixel samples to sample from out-of-mask regions (in a bounding box).
        subject (str): name of the dataset subject
        mask_img (bool): replace background parts with estimated background pixels
        multiview (bool): to enable multiview optimization
        perturb_bg (bool): perturb background color during training
        inverse_sampling (bool): sampling non-foreground pixels
        bkgd_mul (bool): multiplier for background pixels. Sometimes giving 0.0 is better..
        read_full_img (bool): return full image
        dilate_mask (bool): dilate the sampling mask during training. NOTE: this depends on the individual dataset implementation
        '''
        self.h5_path = h5_path
        self.split = split
        self.dataset = None
        self.subject = subject
        self.mask_img = mask_img
        self.multiview = multiview
        self.perturb_bg = perturb_bg
        self.full_perturb_bg = full_perturb_bg
        self.inverse_sampling = inverse_sampling
        self.bkgd_mul = bkgd_mul
        self.read_full_img = read_full_img
        self.dilate_mask = dilate_mask

        self.N_samples = N_samples
        self.patch_size = patch_size
        self.N_nms = int(math.floor(N_nms)) if N_nms >= 1.0 else float(N_nms)
        self._idx_map = None # map queried idx to predefined idx
        self._render_idx_map = None # map idx for render

        self.init_meta()
        self.init_len()
        self.box2d = None
        if self.N_nms > 0.0:
            self.init_box2d()

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
        pose_data = self.get_pose_data(idx, q_idx, self.N_samples)

        # sample pixels
        pixel_idxs, fg, sampling_mask, sort_idxs = self.sample_pixels(idx, q_idx)

        # maybe get a version that computes only sampled points?
        rays_o, rays_d = self.get_rays(c2w, focal, pixel_idxs, center)

        # load the image, foreground and background,
        # and get values from sampled pixels
        rays_rgb, fg, bg, rgb_not_masked, bg_orig, full_img, full_fg = self.get_img_data(idx, pixel_idxs, fg=fg)

        # debug
        '''
        #H, W = 800, 800
        #H, W = 512, 512#1080, 1920
        #H = W = 1024
        H, W = 1080, 1920
        from core.utils.visualization import draw_skeletons_3d
        import imageio
        kps = pose_data['kp3d']
        img = self.dataset['imgs'][idx].reshape(H, W, 3)
        #msk = self.dataset['sampling_masks'][idx].reshape(H, W, 1)
        msk = self.dataset['masks'][idx].reshape(H, W, 1)
        img = img.copy() * msk.copy() + (1 - msk.copy()) * 128
        skel_img = draw_skeletons_3d(img[None], kps[:1], c2w[None], H, W, focal[None], center[None], skel_type=self.skel_type)
        #skel_img = draw_skeletons_3d(img[None], kps[:1], c2w[None], H, W, focal[None], center[None])
        imageio.imwrite(f'danbo_examine/{q_idx}.png', skel_img.reshape(H, W, 3))
        #print(cam_idxs)
        '''
        """
        # debug patch
        import imageio
        target_s = (rays_rgb.reshape(self.patch_size, self.patch_size, 3) * 255).astype(np.uint8)
        imageio.imwrite(f'h36m_debug/{idx}.png', target_s)
        """
        H, W = self.HW
        pixel_idxs = np.stack([pixel_idxs // W, pixel_idxs % W],axis=-1) 

        return_dict = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'target_s': rays_rgb,
            'target_s_not_masked': rgb_not_masked,
            **pose_data,
            'cam_idxs': cam_idxs,
            'fgs': fg,
            'bgs': bg,
            'bg_orig': bg_orig,
            'pixel_idxs': pixel_idxs,
            'c2ws': c2w[None],
            'K': K[None],
        }

        if full_img is not None:
            return_dict['full_img'] = full_img[None]
            return_dict['full_fg'] = full_fg[None]

        if self.patch_size > 1:
            # undo sorting to keep patch together
            unsort_keys = [
                'rays_o', 'rays_d', 'target_s', 'target_s_not_masked',
                'fgs', 'bgs', 'bg_orig', 'pixel_idxs',
            ]
            for unsort_key in unsort_keys:
                tar = np.zeros_like(return_dict[unsort_key])
                sort_idxs_ = sort_idxs.copy()
                if tar.ndim == 2:
                    sort_idxs_ = sort_idxs_[..., None].repeat(tar.shape[-1], -1)
                np.put_along_axis(
                    tar,
                    sort_idxs_, 
                    return_dict[unsort_key],
                    axis=0,
                )
                return_dict[unsort_key] = tar

        return return_dict

    def __len__(self):
        return self.data_len

    def init_len(self):
        if self._idx_map is not None:
            self.data_len = len(self._idx_map)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                self.data_len = len(f['imgs'])

    def init_dataset(self):

        if self.dataset is not None:
            return
        print('init dataset')

        self.dataset = h5py.File(self.h5_path, 'r')

    def init_meta(self):
        '''
        Init properties that can be read directly into memory (as they are small)
        '''
        print('init meta')
        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        self.dataset_keys = [k for k in dataset.keys()]

        # initialize some attributes
        self.has_bg = 'bkgds' in self.dataset_keys
        self.centers = None

        if 'centers' in dataset:
            self.centers = dataset['centers'][:]

        # precompute mesh (for ray generation) to reduce computational cost
        img_shape = dataset['img_shape'][:]
        self._N_total_img = img_shape[0]
        self.HW = img_shape[1:3]
        mesh = np.meshgrid(
            np.arange(self.HW[1], dtype=np.float32),
            np.arange(self.HW[0], dtype=np.float32),
            indexing='xy',
        )
        self.mesh = mesh

        i, j = mesh[0].reshape(-1), mesh[1].reshape(-1)
        if self.centers is None:
            offset_y, offset_x = self.HW[0] * 0.5, self.HW[1] * 0.5
        else:
            # have per-image center. apply those during runtime
            offset_y = offset_x = 0.

        # pre-computed direction, the first two cols
        # need to be divided by focal
        self._dirs = np.stack([
            (i-offset_x),
            -(j-offset_y),
            -np.ones_like(i)
        ], axis=-1)

        # pre-computed pixel indices
        self._pixel_idxs = np.arange(np.prod(self.HW)).reshape(*self.HW)

        # store pose and camera data directly in memory (they are small)
        self.gt_kp3d = dataset['gt_kp3d'][:] if 'gt_kp3d' in self.dataset_keys else None
        self.kp_map, self.kp_uidxs = None, None # only not None when self.multiview = True
        self.kp3d, self.bones, self.skts, self.cyls = self._load_pose_data(dataset)

        self.focals, self.c2ws = self._load_camera_data(dataset)
        self.temp_validity = self.init_temporal_validity()

        if self.has_bg:
            self.bgs = dataset['bkgds'][:].reshape(-1, np.prod(self.HW), 3)
            self.bg_idxs = dataset['bkgd_idxs'][:].astype(np.int64)

        # TODO: maybe automatically figure this out
        self.skel_type = SMPLSkeleton

        dataset.close()

    def _load_pose_data(self, dataset):
        '''
        read pose data from .h5 file
        '''
        kp3d, bones, skts, cyls = dataset['kp3d'][:], dataset['bones'][:], \
                                    dataset['skts'][:], dataset['cyls'][:]
        if self.multiview:
            return self._load_multiview_pose(dataset, kp3d, bones, skts, cyls)
        return kp3d, bones, skts, cyls

    def _load_multiview_pose(self, dataset, kp3d, bones, skts, cyls):
        '''
        Multiview data for pose optimization, depends on dataset
        '''
        assert self._idx_map is None, 'Subset is not supported for multiview optimization'
        raise NotImplementedError

    def _load_camera_data(self, dataset):
        '''
        read camera data from .h5 file
        '''
        return dataset['focals'][:], dataset['c2ws'][:]

    def init_box2d(self):
        '''
        pre-compute box2d
        '''
        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        l = len(self)

        H, W = self.HW
        self.box2d = []
        for i in range(len(dataset['imgs'])):
            q_idx = i
            idx = q_idx

            # get camera information
            c2w, focal, center, cam_idxs = self.get_camera_data(idx, q_idx, 1)

            # get kp index and kp, skt, bone, cyl
            _, _, _, _, cyls = self.get_pose_data(idx, q_idx, 1)
            tl, br, _ = cylinder_to_box_2d(
                cyls[0], 
                [H, W, focal], 
                nerf_c2w_to_extrinsic(c2w),
                center=center, 
                scale=1.3
            )
            self.box2d.append((tl, br))
        self.box2d = np.array(self.box2d)
        dataset.close()

    def init_temporal_validity(self):
        return None

    def get_camera_data(self, idx, q_idx, N_samples):
        '''
        get camera data
        '''

        # real_idx: the real data we want to sample from
        # cam_idx: for looking up camera code
        real_idx, cam_idx = self.get_cam_idx(idx, q_idx)
        focal = self.focals[real_idx]
        c2w = self.c2ws[real_idx].astype(np.float32)

        center = None
        if self.centers is not None:
            center = self.centers[real_idx]
            K = build_intrinsic(center[0], center[1], focal)
        else:
            K = build_intrinsic(self.HW[1] * 0.5, self.HW[0] * 0.5, focal)

        cam_idx = np.array(cam_idx).reshape(-1, 1).repeat(N_samples, 1).reshape(-1)


        return c2w, K, focal, center, cam_idx


    def get_img_data(self, idx, pixel_idxs, fg=None):
        '''
        get image data (in np.uint8), convert to float
        '''

        time0 = time.time()
        if fg is None:
            try:
                fg = self.dataset['masks'][idx, pixel_idxs].astype(np.float32)
            except:
                print(f'taking increasing order, got {pixel_idxs}')
                raise ValueError
        else:
            fg = fg[pixel_idxs].copy().astype(np.float32).reshape(-1, 1)

        if self.read_full_img:
            full_fg = self.dataset['masks'][idx].astype(np.float32)
            fg_idx = np.where(full_fg[..., 0] > 0)[0]
            full_img = np.zeros((np.prod(self.HW), 3), dtype=np.float32)
            full_img[fg_idx] = self.dataset['imgs'][idx, fg_idx] / 255.
        else:
            full_fg = None
            full_img = None
        
        img = self.dataset['imgs'][idx, pixel_idxs].astype(np.float32) / 255.

        bg, bg_orig = None, None
        img_not_masked = img.copy()
        if self.has_bg:
            bg_idx = self.bg_idxs[idx]
            bg_orig = self.bgs[bg_idx, pixel_idxs].astype(np.float32) / 255.

            if self.perturb_bg:
                noise = np.random.random(bg_orig.shape).astype(np.float32)
                #noise= (1 - fg) * noise # do not perturb foreground area!
                # do not perturb foreground area!
                if self.full_perturb_bg:
                    bg = noise
                else:
                    bg = (1 - fg) * noise + fg * bg_orig * self.bkgd_mul
                #bg = noise
            else:
                bg = bg_orig

            if self.mask_img:
                img = img * fg + (1. - fg) * bg

        return img, fg, bg, img_not_masked, bg_orig, full_img, full_fg

    def sample_pixels(self, idx, q_idx):
        '''
        return sampled pixels (in (H*W,) indexing, not (H, W))
        '''
        p = self.patch_size
        N_rand = self.N_samples // int(p**2)
        # TODO: check if sampling masks need adjustment
        # assume sampling masks are of shape (N, H, W, 1)
        #time0 = time.time()
        sampling_mask = self.dataset['sampling_masks'][idx].reshape(-1)
        if self.dilate_mask:
            border = 3
            kernel = np.ones((border, border), np.uint8)
            sampling_mask = cv2.dilate(sampling_mask.reshape(*self.HW), kernel=kernel, iterations=1).reshape(-1)
        #print(f'fetch mask time {time.time()-time0}')

        valid_idxs, = np.where(sampling_mask>0)
        if len(valid_idxs) == 0 or len(valid_idxs) < N_rand:
            valid_idxs = np.arange(len(sampling_mask))

        if self.patch_size == 1:
            sampled_idxs = np.random.choice(
                valid_idxs,
                N_rand,
                replace=False
            )
        if self.patch_size > 1:
            # dumb way to extract non-overlapping patch idxes
            H, W = self.HW
            ptr = 0
            valid_idxs = np.random.permutation(valid_idxs)
            sampled_tuples = None
            while sampled_tuples is None or len(sampled_tuples) < N_rand:
                sampled_idx = valid_idxs[ptr]
                hn = np.clip(sampled_idx // W, 0, H-p)
                wn = np.clip(sampled_idx % W, 0, W-p)

                if sampled_tuples is None:
                    sampled_tuples = np.array([[hn, wn]])
                    continue
                else:
                    diff = np.abs(sampled_tuples - np.array([[hn, wn]]))
                    if np.all(diff[:, 0] >= p) and np.all(diff[:, 1] >= p):
                        sampled_tuples = np.concatenate([sampled_tuples, np.array([[hn, wn]])], axis=0)
                ptr += 1
                if ptr == len(valid_idxs):
                    raise ValueError('Not enough valid pixels to sample from')
            
            sampled_idxs = []
            for sampled_tuple in sampled_tuples:
                h, w = sampled_tuple
                patch = self._pixel_idxs[h:h+p, w:w+p].reshape(-1)
                sampled_idxs.append(patch)
            sampled_idxs = np.array(sampled_idxs).reshape(-1)

        # if self.N_nms >= 1
        if isinstance(self.N_nms, int):
            N_nms = self.N_nms
        else:
            # roll a dice
            #dice = np.random.random()
            #N_nms = int(self.N_nms > dice)
            N_nms = int(self.N_nms > np.random.random())

        if N_nms > 0:
            # replace some empty-space samples of out-of-mask samples
            nms_idxs = self._sample_in_box2d(idx, q_idx, sampling_mask, N_nms)

            sampled_idxs = np.sort(sampled_idxs)
            sampled_idxs[np.random.choice(len(sampled_idxs), size=(N_nms,), replace=False)] = nms_idxs

        # hdf5 takes increasing index order
        sort_idxs = np.argsort(sampled_idxs)
        sampled_idxs = sampled_idxs[sort_idxs]
        return sampled_idxs, None, None, sort_idxs

    def _sample_in_box2d(self, idx, q_idx, fg, N_samples):

        H, W = self.HW
        # get bounding box
        real_idx, _ = self.get_cam_idx(idx, q_idx)
        tl, br = self.box2d[real_idx].copy()

        fg = fg.reshape(H, W)
        cropped = fg[tl[1]:br[1], tl[0]:br[0]]
        vy, vx = np.where(cropped < 1)

        # put idxs from cropped ones back to the non-cropped ones
        vy = vy + tl[1]
        vx = vx + tl[0]
        idxs = vy * W + vx

        #selected_idxs = np.random.choice(idxs, size=(N_samples,), replace=False)
        # This is faster for small N_samples
        selected_idxs = np.random.default_rng().choice(idxs, size=(N_samples,), replace=False)

        return selected_idxs

    def get_rays(self, c2w, focal, pixel_idxs, center=None):
        dirs = self._dirs[pixel_idxs].copy()
        if center is not None:
            center = center.copy()
            center[1] *= -1
            dirs[..., :2] -= center

        dirs[:, :2] /= focal

        I = np.eye(3)

        if np.isclose(I, c2w[:3, :3]).all():
            rays_d = dirs # no rotation required if rotation is identity
        else:
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1)[..., None]
        
        return rays_o.copy(), rays_d.copy()

    def get_pose_data(self, idx, q_idx, N_samples):

        # real_idx: the real data we want to sample from
        # kp_idx: for looking up the optimized kp in poseopt layer (or other purpose)
        real_idx, kp_idx = self.get_kp_idx(idx, q_idx)
        kp, bone, cyl, skt = self._get_pose_data(real_idx)

        # TODO: think this part through
        temp_val = None
        if self.temp_validity is not None:
            temp_val = self.temp_validity[real_idx:real_idx+1]

        kp_idx = np.array([kp_idx]).repeat(N_samples, 0)
        kp = kp.repeat(N_samples, 0)
        bone = bone.repeat(N_samples, 0)
        cyl = cyl.repeat(N_samples, 0)
        skt = skt.repeat(N_samples, 0)

        return {
            'kp_idx': kp_idx, 
            'kp3d': kp, 
            'bones': bone, 
            'skts': skt, 
            'cyls': cyl,
            #'real_kp_idx': (np.array([real_idx]*N_samples) / self.kp_idxs.max()).astype(np.float32),
            'real_kp_idx': np.array([real_idx]*N_samples),
        }
    def _get_pose_data(self, real_idx):
        return self.kp3d[real_idx:real_idx+1].astype(np.float32), \
               self.bones[real_idx:real_idx+1].astype(np.float32), \
               self.cyls[real_idx:real_idx+1].astype(np.float32), \
               self.skts[real_idx:real_idx+1].astype(np.float32)


    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx, q_idx

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx, q_idx

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
            _k_idxs = _kq_idxs = np.arange(len(self.kp3d))
            _c_idxs = _cq_idxs = np.arange(len(self.c2ws))

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

    def get_meta(self):
        '''
        return metadata needed for other parts of the code.
        '''

        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        if hasattr(self, 'rest_pose'):
            rest_pose = self.rest_pose.copy()
        else:
            rest_pose = dataset['rest_pose'][:]

        # get idxs to retrieve the correct subset of meta-data

        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs()

        # keep only the unique ones
        k_idxs = np.unique(k_idxs)
        c_idxs = np.unique(c_idxs)

        # prepare HWF
        H, W = self.HW
        if not np.isscalar(self.focals):
            H = np.repeat([H], len(c_idxs), 0)
            W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])

        # prepare center if there's one
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()

        # average beta
        if 'betas' in dataset:
            betas = dataset['betas'][:]
        else:
            betas = np.zeros((len(self.kp3d), 10)).astype(np.float32)
        if len(betas) > 1:
            betas = betas[k_idxs]
        betas = betas.mean(0, keepdims=True).repeat(len(betas), 0)

        
        data_attrs = {
            'hwf': hwf,
            'center': center,
            'c2ws': self.c2ws[c_idxs],
            'near': 60., 'far': 100., # don't really need this
            'n_views': self.data_len,
            # skeleton-related info
            'skel_type': self.skel_type,
            'rest_pose': rest_pose,
            'gt_kp3d': self.gt_kp3d[k_idxs] if self.gt_kp3d is not None else None,
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
            'bones': self.bones[k_idxs],
            'betas': betas,
            'kp_map': self.kp_map, # important for multiview setting
            'kp_uidxs': self.kp_uidxs, # important for multiview setting
        }

        dataset.close()

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
        render_imgs = dataset['imgs'][i_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][i_idxs].reshape(-1, H, W, 1)
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
            'cam_idxs': c_idxs,
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
    
    def get_trainingviz_data(self, viz_frame):

        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs(render=False)

        # grab only a subset (15 images) for rendering
        kq_idxs = kq_idxs[viz_frame:viz_frame+1]
        cq_idxs = cq_idxs[viz_frame:viz_frame+1]
        i_idxs = i_idxs[viz_frame:viz_frame+1]
        k_idxs = k_idxs[viz_frame:viz_frame+1]
        c_idxs = c_idxs[viz_frame:viz_frame+1]

        # get images if split == 'render'
        # note: needs to have self._idx_map
        H, W = self.HW
        render_imgs = dataset['imgs'][i_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][i_idxs].reshape(-1, H, W, 1)
        render_bgs = self.bgs.reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bg_idxs = self.bg_idxs[i_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])

        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        
        # TODO: c_idxs, k_idxs ... confusion
        trainingviz_data = {
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
            'kp_idxs': k_idxs,
            'kp_idxs_len': len(self.kp3d),
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
            'bones':self.bones[k_idxs],
        }

        dataset.close()

        return trainingviz_data


class PoseRefinedDataset(BaseH5Dataset):

    def __init__(self, *args, load_refined=False, **kwargs):
        self.load_refined = load_refined
        super(PoseRefinedDataset, self).__init__(*args, **kwargs)

    def _load_pose_data(self, dataset):
        '''
        read pose data from .h5 or refined poses
        NOTE: refined poses are defined in a per-dataset basis.
        '''
        if not self.load_refined:
            return super(PoseRefinedDataset, self)._load_pose_data(dataset)

        assert hasattr(self, 'refined_paths'), \
            f'Paths to refined poses are not defined for {self.__class__}.'

        refined_path, legacy = self.refined_paths[self.subject]
        print(f'Read refined poses from {refined_path}')
        # the first 4 is kp3d, bones, skts, cyls
        kp3d, bones, skts, cyls = pose_ckpt_to_pose_data(refined_path, ext_scale=0.001, legacy=legacy)[:4]

        if self.multiview:
            return self._load_multiview_pose(dataset, kp3d, bones, skts, cyls)
        return kp3d, bones, skts, cyls

