import numpy as np

import h5py

from core.utils.skeleton_utils import *
from core.utils.visualization import *
from core.networks.misc import *

from core.datasets import BaseH5Dataset
from core.datasets.eval_dataset import BaseEvalDataset


class GSCRealDataset(BaseH5Dataset):
    render_skip = 10
    N_render = 9
    n_vals = 30
    n_trains = 200
    n_total_cams = 3
    val_cams = [0, 1, 2]
    train_cams = [0, 1, 2]

    def __init__(self, *args, num_train_cams=3, **kwargs):
        self.num_train_cams = num_train_cams
        super().__init__(*args, **kwargs)

    def init_meta(self):
        super().init_meta()
        dataset = h5py.File(self.h5_path, 'r')

        self.kp_idxs = kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = cam_idxs = dataset['img_pose_indices'][:]

        # camera idxs are stored individually
        total_imgs = len(dataset['imgs'])
        idxs = np.arange(total_imgs)

        # because we duplicate the data for each cam/view
        img_per_cam = total_imgs // self.n_total_cams
        cam_idxs_ = idxs // img_per_cam

        train_cams = self.train_cams[:self.num_train_cams]
        kp_selected = np.where(self.kp_idxs < self.n_trains)[0]
        cam_selected = np.array([c for c, cid in enumerate(cam_idxs_) if cid in train_cams])
        self._train_idx_map = np.intersect1d(kp_selected, cam_selected)

        kp_selected = np.where(self.kp_idxs < self.n_vals)[0]
        cam_selected = np.array([c for c, cid in enumerate(cam_idxs_) if cid in self.val_cams])
        self._val_idx_map = np.intersect1d(kp_selected, cam_selected)

        self._idx_map = None
        if self.split == 'full':
            self._idx_map = None
        elif self.split == 'train':
            self._idx_map = self._train_idx_map.copy()
        elif self.split == 'test':
            self._idx_map = self._val_idx_map.copy()

        self.bgs = dataset['bkgds'][:]
        self.bg_idxs = dataset['bkgd_idxs'][:]
        dataset.close()
        self.skel_type = ViTPoseSkeletonv2

        self.has_bg = True
        #self.bgs = 255*np.ones((1, np.prod(self.HW), 3), dtype=np.uint8)
        #self.bg_idxs = np.zeros((len(self.kp_idxs),), dtype=np.int64)


        #import pdb; pdb.set_trace()
        #print

    def init_len(self):
        if self._idx_map is not None:
            self.data_len = len(self._idx_map)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                self.data_len = len(f['imgs']) 

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

    def get_meta(self):
        '''
        return metadata needed for other parts of the code.
        '''

        data_attrs = super().get_meta()
        dataset = h5py.File(self.h5_path, 'r')
        rest_heads = dataset['rest_heads'][:]
        dataset.close()

        data_attrs['rest_heads'] = rest_heads
        data_attrs['skel_type'] = ViTPoseSkeletonv2

        return data_attrs

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
            if not render:
                _idx_map = self._train_idx_map
            else:
                _idx_map = self._val_idx_map

            i_idxs = _idx_map
            _k_idxs = _idx_map
            _c_idxs = _idx_map
            _kq_idxs = np.arange(len(_idx_map))
            _cq_idxs = np.arange(len(_idx_map))

        else:
            # queried == actual index
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(len(self.kp_idxs))
            _c_idxs = _cq_idxs = np.arange(len(self.cam_idxs))

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

    def get_render_data(self):
        # if 'train' in self.h5_path:
        #     h5_path = self.h5_path.replace('train', 'test')
        
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
        render_imgs = render_imgs * render_fgs
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
            'cam_idxs': (c_idxs * 0).astype(np.int32),
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


class GSCRealSMPLDataset(BaseH5Dataset):
    N_render = 5
    render_skip = 20
    n_trains = 100
    n_vals = 100
    train_cams = [0, 1, 2]
    val_cams = [0, 1, 2]

    def __init__(self, *args, num_train_cams=3, **kwargs):
        self.num_train_cams = num_train_cams
        super().__init__(*args, **kwargs)

    def init_meta(self):
        if self.split == 'test':
            self.h5_path = self.h5_path.replace('train', 'val')
        super(GSCRealSMPLDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]


        # camera idxs are stored individually
        total_imgs = len(dataset['imgs'])
        idxs = np.arange(total_imgs)

        # because we duplicate the data for each cam/view
        train_cams = self.train_cams[:self.num_train_cams]
        kp_selected = np.where(self.kp_idxs < self.n_trains)[0]
        cam_selected = np.array([c for c, cid in enumerate(self.cam_idxs) if cid in train_cams])
        self._train_idx_map = np.intersect1d(kp_selected, cam_selected)

        self._idx_map = None
        if self.split == 'train':
            self._idx_map = self._train_idx_map.copy()

        self.bgs = dataset['bkgds'][:]
        self.bg_idxs = dataset['bkgd_idxs'][:]

        dataset.close()
        self.skel_type = SMPLSkeleton

        self.has_bg = True
        # turn c2ws to 4 x 4
        """ 
        c2ws = np.eye(4)[None].repeat(len(self.c2ws), 0)
        c2ws[..., :3, :4] = self.c2ws
        self.c2ws = c2ws.copy()
        """

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

        h5_path = self.h5_path
        if 'train' in self.h5_path:
            h5_path = self.h5_path.replace('train', 'val')

        dataset = h5py.File(h5_path, 'r')
        
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
        render_imgs = render_imgs * render_fgs
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
            'cam_idxs': (c_idxs * 0).astype(np.int32),
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
    

class GSCRealSMPLEvalDataset(BaseEvalDataset):

    def __init__(self, *args, ignore_cams: List[int] = [2], **kwargs):
        self.ignore_cams = ignore_cams
        super().__init__(*args, **kwargs)

    def init_idx(self):
        """ Overwrite with your own evaluation rule.
        --> set self.idxs correctly
        """
        dataset = h5py.File(self.h5_path, 'r')
        idxs = np.arange(len(dataset['imgs']))
        dataset.close()

        self.idxs = np.array([i for i in idxs if self.cam_idxs[i] not in self.ignore_cams])

class RANADataset(BaseH5Dataset):
    N_render = 5
    render_skip = 1
    n_trains = 150
    n_vals = 50
    train_cams = [0]

    def __init__(self, *args, **kwargs):
        self.num_train_cams = 1
        super().__init__(*args, **kwargs)

    def init_meta(self):
        # if self.split == 'test':
        #     self.h5_path = self.h5_path.replace('train', 'val')
        super(RANADataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]


        # camera idxs are stored individually
        total_imgs = len(dataset['imgs'])
        idxs = np.arange(total_imgs)

        # because we duplicate the data for each cam/view
        train_cams = self.train_cams[:self.num_train_cams]
        kp_selected = np.where(self.kp_idxs < self.n_trains)[0]
        cam_selected = np.array([c for c, cid in enumerate(self.cam_idxs) if cid in train_cams])
        self._train_idx_map = np.intersect1d(kp_selected, cam_selected)

        self._idx_map = None
        if self.split == 'train':
            self._idx_map = self._train_idx_map.copy()

        self.bgs = dataset['bkgds'][:]
        self.bg_idxs = dataset['bkgd_idxs'][:]

        dataset.close()
        self.skel_type = SMPLSkeleton

        self.has_bg = True
        # turn c2ws to 4 x 4
        """ 
        c2ws = np.eye(4)[None].repeat(len(self.c2ws), 0)
        c2ws[..., :3, :4] = self.c2ws
        self.c2ws = c2ws.copy()
        """

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

        h5_path = self.h5_path
        # if 'train' in self.h5_path:
        #     h5_path = self.h5_path.replace('train', 'val')

        dataset = h5py.File(h5_path, 'r')
        
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
        render_imgs = render_imgs * render_fgs
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
            'cam_idxs': (c_idxs * 0).astype(np.int32),
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