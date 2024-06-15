import numpy as np

import h5py

from core.utils.skeleton_utils import *
from core.utils.visualization import *
from core.networks.misc import *

from core.datasets import BaseH5Dataset
from core.datasets.eval_dataset import BaseEvalDataset


def A_pose_to_T_pose(
    rest_pose: np.array,
    rest_heads: np.array,
    kp3d: np.array,
    skts: np.array,
    skel_type: Skeleton,
):
    assert skel_type == MixamoSkeleton
    N_J = len(skel_type.joint_names)

    print('Transforming from A-pose to T-pose rest pose')
    # 14, 15, 16, 17 -> Left arms
    # 18, 19, 20, 21 -> Right arms
    bones = torch.zeros(1, N_J, 3)
    left_arm_idxs = np.array([14, 15, 16, 17])
    right_arm_idxs = np.array([18, 19, 20, 21])

    # rotate the shoulder to get T-pose arms
    bones[0, left_arm_idxs[0]] = torch.tensor([0, 0, 1.0471976])
    bones[0, right_arm_idxs[0]] = torch.tensor([0, 0, -1.0471976])

    # t2aposej from T-posed  to a-pose joint-centered
    tpose_rest_pose, t2aposej = calculate_kinematic(
        torch.tensor(rest_pose)[None], 
        bones,
        skel_type=MixamoSkeleton,
        unroll_kinematic_chain=False,
    )

    # create transformation that goes from original Joint spaces in a-pose
    # to joint spaces in T-pose
    tpose_rest_pose = tpose_rest_pose.cpu().numpy()[0]
    aposej2t = t2aposej.inverse().cpu().numpy()
    t2j = np.eye(4)[None, None].repeat(N_J, 1)
    t2j[..., :3, -1] -= tpose_rest_pose[None]

    # now set new rest heads
    tpose_rest_heads = rest_heads.copy()
    parent_idxs = skel_type.joint_trees

    # only the children rest heads are changed
    left_arm_parents = np.array(parent_idxs)[left_arm_idxs[1:]]
    right_arm_parents = np.array(parent_idxs)[right_arm_idxs[1:]]
    tpose_rest_heads[left_arm_idxs[1:]] = tpose_rest_pose[left_arm_parents]
    tpose_rest_heads[right_arm_idxs[1:]] = tpose_rest_pose[right_arm_parents]

    # skts is from world space to a-pose joint-centered space
    # from joint-centered space to A-pose joint space -> A-pose joint space to T-pose
    w2T = aposej2t @ skts
    # skts for T-pose
    skts_T = t2j @ w2T
    l2ws_T = np.linalg.inv(skts_T)
    kp3d_T = l2ws_T[..., :3, -1]
    assert np.allclose(kp3d_T, kp3d, atol=1e-6), 'kp3d should be the same even after transformation'

    tpose_rest_pose = tpose_rest_pose.astype(np.float32)
    tpose_rest_heads = tpose_rest_heads.astype(np.float32)
    skts_T = skts_T.astype(np.float32)
    return tpose_rest_pose, tpose_rest_heads, skts_T


class ASMRDataset(BaseH5Dataset):
    render_skip = 10
    N_render = 6#114
    n_vals = 200
    n_trains = 200
    val_cams = [0,1,2,3]
    train_cams = [0, 1, 2, 3]

    def __init__(
        self,
        *args,
        use_T_rest_pose: bool = False,
        **kwargs,
    ):
        self.use_T_rest_pose = use_T_rest_pose
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
        img_per_cam = total_imgs // (len(self.val_cams + self.train_cams))
        cam_idxs_ = idxs // img_per_cam

        kp_selected = np.where(self.kp_idxs < self.n_trains)[0]
        cam_selected = np.array([c for c, cid in enumerate(cam_idxs_) if cid in self.train_cams])
        self._train_idx_map = np.intersect1d(kp_selected, cam_selected)

        kp_selected = np.where(
            np.logical_and(
                self.kp_idxs >= self.n_trains,
                self.kp_idxs < (self.n_trains + self.n_vals)
            )
        )[0]
        cam_selected = np.array([c for c, cid in enumerate(cam_idxs_) if cid in self.val_cams])
        self._val_idx_map = np.intersect1d(kp_selected, cam_selected)

        if self.split == 'full':
            self._idx_map = None
        elif self.split == 'train':
            self._idx_map = self._train_idx_map.copy()
        elif self.split == 'val':
            self._idx_map = self._val_idx_map.copy()

        # set rest pose bone heads
        self.rest_pose = dataset['rest_pose'][:]
        self.rest_heads = dataset['rest_heads'][:]
        self.bgs = dataset['bkgds'][:]
        self.bg_idxs = dataset['bkgd_idxs'][:]
        dataset.close()
        self.skel_type = MixamoSkeleton

        self.has_bg = True
        #self.bgs = 255*np.ones((1, np.prod(self.HW), 3), dtype=np.uint8)
        #self.bg_idxs = np.zeros((len(self.kp_idxs),), dtype=np.int64)

        if self.use_T_rest_pose:
            self.transform_to_T_rest_pose()
    
    def transform_to_T_rest_pose(self):

        skel_type = self.skel_type
        rest_pose = self.rest_pose
        rest_heads = self.rest_heads
        N_J = len(skel_type.joint_names)

        print('Transforming from A-pose to T-pose rest pose')
        tpose_rest_pose, tpose_rest_heads, skts_T = A_pose_to_T_pose(
            rest_pose,
            rest_heads,
            self.kp3d,
            self.skts,
            MixamoSkeleton,
        )
        self.rest_pose = tpose_rest_pose.astype(np.float32)
        self.rest_heads = tpose_rest_heads.astype(np.float32)
        self.skts = skts_T.copy().astype(np.float32)


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
        #dataset = h5py.File(self.h5_path, 'r')
        #rest_heads = dataset['rest_heads'][:]
        #dataset.close()

        data_attrs['rest_pose'] = self.rest_pose
        data_attrs['rest_heads'] = self.rest_heads
        data_attrs['skel_type'] = MixamoSkeleton

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


class ASMREvalDataset(BaseEvalDataset):
    n_vals = 114
    n_trains = 200
    val_cams = [0,2]
    train_cams = [0, 1, 3]
    n_total_cams = 4

    def __init__(
        self,
        *args,
        use_T_rest_pose: bool = False,
        render_skip: int = 1,
        **kwargs,
    ):
        self.render_skip = render_skip
        super().__init__(*args, **kwargs)
        if use_T_rest_pose:
            self.transform_to_T_rest_pose()
    
    def init_meta(self, *args, **kwargs):
        super().init_meta(*args, **kwargs)
        self.skel_type = MixamoSkeleton

    def get_img_data(self, idx):
        bkgd_idx = self.bg_idxs[idx]
        bg = self.bgs[bkgd_idx].reshape(*self.HW, 3) / 255.
        img = self.dataset['imgs'][idx].reshape(*self.HW, 3) / 255.
        mask = self.dataset['masks'][idx].reshape(*self.HW, 1)
        x, y, w, h = cv2.boundingRect(mask)
        
        x, y, w, h = 0, 0, 1920, 1080
        return {
            'imgs': img.astype(np.float32),
            'masks': mask,
            'bgs': bg.astype(np.float32),
            'bboxes': np.array([x, y, w, h]).astype(np.int32),
        }

    def transform_to_T_rest_pose(self):

        skel_type = self.skel_type
        dataset = h5py.File(self.h5_path, 'r')
        rest_pose = dataset['rest_pose'][:]
        rest_heads = dataset['rest_heads'][:]
        dataset.close()

        print('Transforming from A-pose to T-pose rest pose')
        tpose_rest_pose, tpose_rest_heads, skts_T = A_pose_to_T_pose(
            rest_pose,
            rest_heads,
            self.kp3d,
            self.skts,
            MixamoSkeleton,
        )
        self.skts = skts_T.copy()

    def init_idx(self):

        dataset = h5py.File(self.h5_path, 'r')

        # camera idxs are stored individually
        total_imgs = len(dataset['imgs'])
        idxs = np.arange(total_imgs)

        img_per_cam = total_imgs // self.n_total_cams
        cam_idxs_ = idxs // img_per_cam
        kp_selected = np.where(
            np.logical_and(
                self.kp_idxs >= self.n_trains,
                self.kp_idxs < (self.n_trains + self.n_vals)
            )
        )[0]
        cam_selected = np.array([c for c, cid in enumerate(cam_idxs_) if cid in self.val_cams])
        self.idxs = np.intersect1d(kp_selected, cam_selected)[::self.render_skip]