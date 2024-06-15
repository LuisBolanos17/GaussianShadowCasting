import torch
import numpy as np
import h5py

from core.datasets import BaseH5Dataset
from core.utils.skeleton_utils import *
from core.utils.visualization import *


class AnimalDataset(BaseH5Dataset):
    subject_skel_type = {
        'hare': HARESkeleton,
        'wolf': WOLFSkeleton,
    }
    subject_meta_data = {
        'hare': 'data/animal/hare_meta.npz',
        'wolf': 'data/animal/wolf_meta.npz',
    }
    render_skip = 4

    def __init__(self, *args, pose_scale=1.0, align_skts=True, **kwargs):
        self.pose_scale = pose_scale
        self.align_skts = align_skts
        super().__init__(*args, **kwargs)
    
    def init_meta(self):
        super().init_meta()
        dataset = h5py.File(self.h5_path, 'r')
        self.cam_idxs = dataset['img_pose_indices'][:]
        self.kp_idxs = dataset['kp_idxs'][:]
        self.rest_pose = rest_pose = dataset['rest_pose'][:]
        self.rest_transform = dataset['rest_transform'][:]
        dataset.close()
        self.has_bg = True
        self.bgs = np.zeros((1, *self.HW, 3), dtype=np.uint8).reshape(1, -1, 3) + 255
        self.bg_idxs = np.zeros((len(self.kp_idxs),), dtype=np.int64)
        self.skel_type = self.subject_skel_type[self.subject]

        # IMPORTANT: subsume the rest_transform into skts
        # so that the transformation can be applied in the same
        # way as the other skeletons.
        # the original skt doesn't shift the joint to the origin
        # and is not sharing the same orientation as the rest pose.
        # So we need to apply (1) rest_transform so they have the same orientation
        # and (2) apply shift so the joint will be at the origin.
        rest_transform = self.rest_transform
        skts = self.skts.copy()
        shift = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32).reshape(1, 4, 4).repeat(len(rest_pose), axis=0)
        shift[..., :3, -1] = -rest_pose.copy()

        self.undo_transform = shift[None] @ rest_transform[None]
        skts = self.undo_transform @ skts

        self.skts = skts.copy()
        self.bones = self.create_bones(skts.copy(), self.kp3d.copy()).copy()

        if self.pose_scale != 1.0:
            rescaled = self.rescale_data(
                self.kp3d.copy(),
                self.skts.copy(),
                self.c2ws.copy(),
                self.rest_pose.copy(),
            )
            self.kp3d = rescaled['kp3d']
            self.skts = rescaled['skts']
            self.cyls = rescaled['cyls']
            self.c2ws = rescaled['c2ws']
            self.rest_pose = rescaled['rest_pose']

        print('Verifying if the skts are correct...')
        # now the skts will have the exact same definition as the SMPL skeleton
        # as in (1) applying skts send the joint to the origin
        # and   (2) to move to rest pose, we just add the rest pose location to it
        # now, verify if these are true.

        print('Verifying local-to-world')
        zeros = np.zeros_like(rest_pose)
        l2ws = np.linalg.inv(self.skts)
        world_locs = l2ws[..., :3, :3] @ zeros[..., None] + l2ws[..., :3, 3:4]
        world_locs = world_locs[..., 0]

        assert np.allclose(world_locs, self.kp3d, atol=1e-5), 'Skt is incorrect after fusing the rest_transform!'

        print('Verifying world-to-local')
        # check if we can transform kp3d to zeros.
        joint_locs = self.skts[..., :3, :3] @ self.kp3d[..., None] + self.skts[..., :3, 3:4]
        joint_locs = joint_locs[..., 0]
        assert np.allclose(joint_locs, np.zeros_like(joint_locs), atol=1e-5), 'Skt is incorrect after fusing the rest_transform!'
        print('Skt is correct after fusing the rest_transform!')
        print('Done')
    
    def subsume_skts(self, skts, rest_transform, rest_pose):
        shift = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32).reshape(1, 4, 4).repeat(len(rest_pose), axis=0)
        shift[..., :3, -1] = -rest_pose.copy()

        self.undo_transform = shift[None] @ rest_transform[None]
        skts = self.undo_transform @ skts
        return skts

    def rescale_data(self, kp3d, skts, c2ws, rest_pose):
        '''
        Rescale pose, c2ws to make it easier for us to train
        '''
        print('Rescaling animal data.')
        pose_scale = self.pose_scale
        kp3d = kp3d * pose_scale
        l2ws = np.linalg.inv(skts)
        l2ws[..., :3, 3] *= pose_scale
        skts = np.linalg.inv(l2ws)
        cyls = get_kp_bounding_cylinder(self.kp3d, self.skel_type, head='-y')

        c2ws = c2ws.copy()
        c2ws[..., :3, -1] *= pose_scale
        rest_pose = rest_pose.copy() * pose_scale
        assert np.allclose(kp3d, np.linalg.inv(skts)[:, :, :3, 3], atol=1e-5)

        print(f'Done rescaling animal data.')
        return {
            'kp3d': kp3d,
            'skts': skts,
            'cyls': cyls,
            'c2ws': c2ws,
            'rest_pose': rest_pose,
        }

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

        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        rest_pose = self.rest_pose #dataset['rest_pose'][:]

        # get idxs to retrieve the correct subset of meta-data

        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs()

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
        
        metadata = np.load(self.subject_meta_data[self.subject], allow_pickle=True)
        lbs_weights = metadata['lbs_weights']
        valid_joints = np.where(lbs_weights.max(axis=0) > 0)[0]
        rest_tails = metadata['rest_tails'][valid_joints]
        rest_heads = metadata['rest_heads'][valid_joints]
        assert np.allclose(rest_tails, rest_pose, atol=1e-5)

        data_attrs = {
            'hwf': hwf,
            'center': center,
            'c2ws': self.c2ws[c_idxs],
            'near': 60., 'far': 100., # don't really need this
            'n_views': self.data_len,
            # skeleton-related info
            'skel_type': self.subject_skel_type[self.subject], 
            'rest_pose': rest_pose,
            'rest_heads': rest_heads,
            'rest_transform': self.rest_transform,
            'gt_kp3d': self.gt_kp3d[k_idxs] if self.gt_kp3d is not None else None,
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
            'bones': self.bones[k_idxs],
            'kp_map': self.kp_map, # important for multiview setting
            'kp_uidxs': self.kp_uidxs, # important for multiview setting
        }

        dataset.close()

        return data_attrs

    def get_render_data(self):
        #h5_path = self.h5_path.replace('train', 'test')
        h5_path = self.h5_path.replace('train', 'val_ood')

        H, W = self.HW
        render_skip = self.render_skip
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][::render_skip].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][::render_skip].reshape(-1, H, W, 1).astype(np.float32)
        render_bgs = np.zeros_like(render_imgs[:1]).astype(np.float32)
        render_bg_idxs = np.zeros(len(render_imgs)).astype(np.int64)

        kp3d = dataset['kp3d'][::render_skip]
        skts = dataset['skts'][::render_skip]
        skts = self.undo_transform @ skts

        center = dataset['centers'][::render_skip]
        focals = dataset['focals'][::render_skip]
        c2ws = dataset['c2ws'][::render_skip]

        bones = self.create_bones(skts.copy(), kp3d.copy())

        rescaled = self.rescale_data(
            kp3d,
            skts,
            c2ws,
            self.rest_pose,
        )
        kp3d = rescaled['kp3d']
        skts = rescaled['skts']
        c2ws = rescaled['c2ws']

        c_idxs = dataset['img_pose_indices'][::render_skip]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, focals)

        dataset.close()

        render_data = {
            'imgs': render_imgs, 
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(render_bg_idxs),
            # camera data
            'cam_idxs': c_idxs * 0 - 1, # set to -1 to use avg framecode
            'cam_idxs_len': len(c2ws),
            'c2ws': c2ws,
            'hwf': hwf,
            'center': center,
            'kp_idxs': np.arange(len(kp3d)),
            'kp_idxs_len': len(kp3d),
            'kp3d': kp3d,
            'skts': skts,
            'bones': bones,
        }
        return render_data

    def create_bones(self, skts, kp3d):
        skel_type = self.skel_type
        parent_idxs = skel_type.joint_trees
        skts_parent = skts[:, parent_idxs]
        parent_to_child = skts @ np.linalg.inv(skts_parent)
        child_to_parent = np.linalg.inv(parent_to_child)
        rot = child_to_parent[..., :3, :3]
        N, J = skts.shape[:2]
        bones = rot_to_axisang(torch.tensor(rot)).cpu().numpy()
        bones = bones.reshape(N, J, 3)
        return bones
