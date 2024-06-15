import h5py
import numpy as np

from core.datasets import PoseRefinedDataset
from core.utils.skeleton_utils import *


class MonoPerfCapDataset(PoseRefinedDataset):
    n_vals = {'weipeng': 230, 'nadia': 327}

    # define the attribute for rendering data
    render_skip = 10
    N_render = 15

    refined_paths = {
        'weipeng': ('data/MonoPerfCap/weipeng_outdoor/weipeng_refined_new.tar', False),
        'nadia': ('data/MonoPerfCap/nadia_outdoor/nadia_refined_new.tar', False),
    }

    def __init__(self, *args, undo_scale=True, **kwargs):
        self.undo_scale = undo_scale

        if undo_scale and 'load_refined' in kwargs:
            assert kwargs['load_refined'] is False, 'Cannot load refined data when undoing scale.'
        super(MonoPerfCapDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        self.pose_scale = dataset['pose_scale'][()]
        self.rest_pose = dataset['rest_pose'][:]

        train_idxs = np.arange(len(dataset['imgs']))

        self._idx_map = None
        if self.split != 'full':
            n_val = self.n_vals[self.subject]
            val_idxs = train_idxs[-n_val:]
            train_idxs = train_idxs[:-n_val]

            if self.split == 'train':
                self._idx_map = train_idxs
            elif self.split == 'val':
                # skip redundant frames
                self._idx_map = val_idxs[::5]
            else:
                raise NotImplementedError(f'Split {self.split} is not implemented.')

        self.temp_validity = np.ones(len(train_idxs))
        self.temp_validity[0] = 0
        dataset.close()
        super(MonoPerfCapDataset, self).init_meta()
        # the estimation for MonoPerfCap is somehow off by a small scale (possibly due do the none 1:1 aspect ratio)
        if self.undo_scale:
            self.undo_pose_scale()
        self.c2ws[..., :3, -1] /= 1.05
    
    def undo_pose_scale(self):
        print(f'Undoing MonoPerfCap pose scale')
        pose_scale = self.pose_scale
        self.kp3d = self.kp3d / pose_scale
        l2ws = np.linalg.inv(self.skts)
        l2ws[..., :3, 3] /= pose_scale
        self.skts = np.linalg.inv(l2ws)
        self.cyls = get_kp_bounding_cylinder(self.kp3d, skel_type=SMPLSkeleton, head='-y') 
        self.c2ws[..., :3, -1] /= pose_scale

        self.rest_pose = self.rest_pose.copy() / pose_scale
        # assertions to check if everything is alright
        assert np.allclose(self.kp3d, np.linalg.inv(self.skts)[:, :, :3, 3], atol=1e-5)
        l2ws = np.linalg.inv(self.skts)

        l2ws_from_rest = np.array([get_smpl_l2ws(b, self.rest_pose) for b in self.bones]).astype(np.float32)
        l2ws_from_rest[..., :3, -1] += self.kp3d[:, :1]

        assert np.allclose(l2ws, l2ws_from_rest, atol=1e-5)

        print(f'Done undoing MonoPerfCap pose scale.')
    
    def init_temporal_validity(self):
        return self.temp_validity
    