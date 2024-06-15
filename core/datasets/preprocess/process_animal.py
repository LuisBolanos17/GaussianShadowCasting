import os
import math
import json
import torch
import imageio
import numpy as np
import h5py
import cv2

from core.utils.skeleton_utils import *
from core.utils.visualization import *
from core.datasets.preprocess.process_spin import write_to_h5py


def _dataset_view_split(parser, split):
    if split == "all":
        camera_ids = parser.camera_ids
    elif split == "train":
        camera_ids = parser.camera_ids[::2]
    elif split in ["val_ind", "val_ood", "val_view"]:
        camera_ids = parser.camera_ids[1::2]
    elif split == "test":
        camera_ids = parser.camera_ids[1:2]
    return camera_ids


def _dataset_frame_split(parser, split):
    if split in ["train", "val_view"]:
        splits_fp = os.path.join(parser.root_path, "splits/train.txt")
    else:
        splits_fp = os.path.join(parser.root_path, f"splits/{split}.txt")
    with open(splits_fp, mode="r") as fp:
        frame_list = np.loadtxt(fp, dtype=str).tolist()
    frame_list = [(action, int(frame_id)) for (action, frame_id) in frame_list]
    return frame_list


def _dataset_index_list(parser, split):
    camera_ids = _dataset_view_split(parser, split)
    frame_list = _dataset_frame_split(parser, split)
    index_list = []
    for action, frame_id in frame_list:
        index_list.extend(
            [(action, frame_id, camera_id) for camera_id in camera_ids]
        )
    return index_list


class AnimalParser:
    """ Class for parsing TAVA animal datasets
    """
    # rotation to apply in world space to make -y the up-right direction
    ROOT_ROT = np.array(
        [[1, 0, 0, 0],
        [0, 0,-1, 0],
        [0, 1, 0 ,0],
        [0, 0, 0, 1],],
        dtype=np.float32
    )

    SUBJECT_FOLDER = {
        'hare': 'Hare_male_full_RM',
        'wolf': 'Wolf_cub_full_RM_2'
    }


    def __init__(
        self,
        root_path='data/animal/',
        subject='hare',
        splits='train',
        legacy=True,
    ):
        assert subject in self.SUBJECT_FOLDER.keys(), f'Subject {subject} not supported'
        self.subject = self.SUBJECT_FOLDER[subject]
        self.root_path = os.path.join(root_path, self.subject)
        assert os.path.exists(os.path.join(self.root_path, 'splits')), f'No splits folder found! Please create it using tava code first'

        actions = sorted([
            fp for fp in os.listdir(self.root_path)
            if os.path.exists(os.path.join(self.root_path, fp, 'camera.json'))

        ])
        if legacy:
            actions.pop(0)
            g = torch.Generator()
            g = g.manual_seed(56789)
            idxs = torch.randperm(len(actions), generator=g)
            self.actions = [actions[i] for i in idxs]
        else:
            self.actions = actions
        
        self.indexing = self._create_indexing()
        self.skel_info = self._create_skel_info()
        print(self.actions)

    def load_camera(self, action, frame_id=None, camera_id=None):
        path = os.path.join(self.root_path, action, "camera.json")
        with open(path, mode="r") as fp:
            data = json.load(fp)
        if (frame_id is not None) and (camera_id is not None):
            return self.process_camera(data, frame_id, camera_id)
        else:
            return data

    def load_meta_data(self, action, frame_id=None):
        fp = os.path.join(self.root_path, action, "meta_data.npz")
        data = np.load(fp, allow_pickle=True)
        return self.process_pose_data(data, frame_id)
    
    def load_image(self, action, frame_id, camera_id):
        # NOTE: this is RGBA image
        path = os.path.join(self.root_path, action, 'image', camera_id, f'{int(frame_id):08d}.png')
        return imageio.imread(path)

    def process_camera(self, data, frame_id, camera_id):

        intrin = data[str(frame_id)][camera_id]["intrin"]
        extrin = data[str(frame_id)][camera_id]["extrin"]
        K = np.array(intrin, dtype=np.float32)
        # swap mat to turn it into our camera convention..
        c2w = self.ROOT_ROT @ swap_mat(np.linalg.inv(np.array(extrin, dtype=np.float32)))

        center = np.array([K[0, 2], K[1, 2]])
        focal = np.array([K[0, 0], K[1, 1]])
        return {
            'K': K,
            'c2w': c2w,
            'center': center,
            'focal': focal,
        }
    
    def process_pose_data(self, data, frame_id=None):
        """ Also apply the right rotation
        """
        if frame_id is None:
            frame_id = 0
        root_rot = self.ROOT_ROT
        # tails are all joints
        pose_heads = data['pose_heads'][frame_id]
        pose_tails = data['pose_tails'][frame_id]
        pose_matrix = data['pose_matrixs'][frame_id]
        pose_verts = data['pose_verts'][frame_id]
        lbs_weights = data['lbs_weights']
        rest_matrix = data['rest_matrixs'] # transformation to rest pose!

        if hasattr(self, 'skel_info') and 'valid_joints' in self.skel_info:
            valid_joints = self.skel_info['valid_joints']
            pose_heads = pose_heads[valid_joints]
            pose_tails = pose_tails[valid_joints]
            pose_matrix = pose_matrix[valid_joints]
            rest_matrix = rest_matrix[valid_joints]

        # apply rotation so the head is align with -y direction
        pose_heads = (root_rot[None, ..., :3, :3] @ pose_heads[..., None])[..., 0]
        pose_tails = (root_rot[None, ..., :3, :3] @ pose_tails[..., None])[..., 0]
        pose_matrix = root_rot[None] @ pose_matrix
        pose_verts = (root_rot[None, ..., :3, :3] @ pose_verts[..., None])[..., 0]
        pose_skts = np.linalg.inv(pose_matrix)

        # find rest_pose
        # transform to per-joint space first
        pose_j = pose_skts[..., :3, :3] @ pose_tails[..., None] + pose_skts[..., :3, 3:]
        rest_pose = rest_matrix[..., :3, :3] @ pose_j + rest_matrix[..., :3, 3:]
        rest_pose = rest_pose[..., 0]

        ret_data = {
            'pose_heads': pose_heads,
            'pose_tails': pose_tails,
            'pose_matrix': pose_matrix,
            'pose_verts': pose_verts,
            'lbs_weights': lbs_weights,
            'rest_matrix': rest_matrix,
            'pose_skts': pose_skts,
            'bnames': data['bnames'],
            'bnames_parent': data['bnames_parent'],
            'rest_pose': rest_pose,
            'pose_skts': pose_skts,
        }

        if hasattr(self, 'skel_info') and 'skel_type' in self.skel_info:
            skel_type = self.skel_info['skel_type']
            cyl = get_kp_bounding_cylinder(pose_tails, skel_type, head='-y')
            ret_data['cyl'] = cyl

            # create bone directions
            # let's just set it to 0, similar to TAVA
            bones = np.zeros_like(pose_tails).astype(np.float32)
            ret_data['bones'] = bones

        return ret_data
        

    @property
    def camera_ids(self):
        return self.indexing[self.actions[0]][0]

    @property
    def frame_ids(self):
        return sorted(list(self.indexing[self.actions[0]].keys()))

    def _create_indexing(self):
        indexing = {}
        for action in self.actions:
            indexing[action] = {}
            for frame_id, camera_data in self.load_camera(action).items():
                indexing[action][int(frame_id)] = sorted(
                    list(camera_data.keys())
                )
        return indexing
    
    def _create_skel_info(self):
        action = list(self.indexing.keys())[0]
        data = self.load_meta_data(action)
        bnames = data['bnames']
        bnames_parent = data['bnames_parent']

        # use only to identify which joints are valid
        valid_joints = np.where(data['lbs_weights'].max(axis=0) > 0)[0]

        # roughly estimate the number of hops we need for GNN
        # ok, probably 3-4 hops?
        #print(np.unique((lbs_weights > 0).sum(axis=1), return_counts=True))

        bnames = bnames[valid_joints]
        bnames_parent = bnames_parent[valid_joints]

        joint_names = bnames
        joint_trees = np.zeros(len(valid_joints), dtype=np.int32)

        for i, parent in enumerate(bnames_parent):
            for j, joint_name in enumerate(joint_names):
                if joint_name == parent:
                    joint_trees[i] = j
                    break

        animal_skel = Skeleton(
            joint_names=joint_names,
            joint_trees=joint_trees,
            root_id=np.where(joint_trees==np.arange(len(joint_trees)))[0],
            nonroot_id=1,
            cutoffs={},
            end_effectors=[],
        )

        return {
            'valid_joints': valid_joints,
            'skel_type': animal_skel,
        }


def process_animal_data(
    root_path='data/animal/', 
    subject='hare', 
    legacy=True, 
    split='train', 
    num_evals=0,
    H=800, 
    W=800,
    compression='gzip',
    chunk_size=64,
):
    parser = AnimalParser(root_path=root_path, subject=subject, legacy=legacy)
    index_list = _dataset_index_list(parser, split=split)
    if num_evals > 0:
        assert split != 'train', "num_evals only works for eval and test"
        get_every = math.ceil(len(index_list) / num_evals)
        index_list = index_list[::get_every]

    N_imgs = len(index_list)

    h5_path = os.path.join(root_path, f'{subject}_{split}.h5')
    if os.path.exists(h5_path):
        print(f'old {h5_path} exist, remove it')
        os.remove(h5_path)
    
    # set up h5
    h5_file = h5py.File(h5_path, 'w')
    img_shape = (N_imgs, H, W, 3)
    ds = h5_file.create_dataset('img_shape', (4, ), dtype=np.int32)
    ds[:] = np.array(img_shape, dtype=np.int32)

    # flatten shape for image data
    flatten_shape = (N_imgs, H*W)
    img_chunk = (1, int(chunk_size**2))
    ds_imgs = h5_file.create_dataset(
        'imgs', 
        flatten_shape + (3,), 
        np.uint8, 
        chunks=img_chunk + (3,), 
        compression=compression
    )
    ds_masks = h5_file.create_dataset(
        'masks', flatten_shape + (1,), 
        np.uint8, 
        chunks=img_chunk + (1,), 
        compression=compression
    )
    ds_sampling_masks = h5_file.create_dataset(
        'sampling_masks', flatten_shape + (1,), 
        np.uint8, 
        chunks=img_chunk + (1,), 
        compression=compression
    )

    #imgs = [], sampling_masks, masks = [], []
    cam_dicts = []
    pose_dicts = []
    print(len(index_list))
    n_data = len(index_list)
    kp3d, bones, skts, cyls, kp_idxs = [], [], [], [], []
    rest_matrices = []
    c2ws, focals, centers = [], [], []
    rest_poses = []

    for i, (action, frame_id, camera_id) in enumerate(index_list):
        cam_data = parser.load_camera(action, frame_id, camera_id)
        pose_data = parser.load_meta_data(action, frame_id)
        img = parser.load_image(action, frame_id, camera_id)

        mask = img[..., 3:4]
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        sampling_mask = cv2.dilate(mask.copy(), np.ones((5, 5), np.uint8), iterations=2)[..., None]

        ds_imgs[i] = img[..., :3].reshape(-1, 3)
        ds_masks[i] = mask.reshape(-1, 1)
        ds_sampling_masks[i] = sampling_mask.reshape(-1, 1)

        cam_dicts.append(cam_data)
        pose_dicts.append(pose_data)

        kp3d.append(pose_data['pose_tails'])
        skts.append(pose_data['pose_skts'])
        cyls.append(pose_data['cyl'])
        rest_poses.append(pose_data['rest_pose'])
        bones.append(pose_data['bones'])
        kp_idxs.append(frame_id)
        rest_matrices.append(pose_data['rest_matrix'])

        c2ws.append(cam_data['c2w'])
        focals.append(cam_data['focal'])
        centers.append(cam_data['center'])

    skel_type = parser.skel_info['skel_type']
    
    # TODO: create rest pose as well!
    kp3d = np.stack(kp3d).astype(np.float32)
    skts = np.stack(skts).astype(np.float32)
    cyls = np.stack(cyls).astype(np.float32)
    bones = np.stack(bones).astype(np.float32)
    rest_poses = np.stack(rest_poses).astype(np.float32)
    rest_matrices = np.stack(rest_matrices).astype(np.float32)
    assert np.allclose(rest_poses[:1], rest_poses, atol=1e-5)

    parents = skel_type.joint_trees
    parent_skts = skts[:, parents]
    joints_in_parents = (parent_skts[..., :3, :3] @ kp3d[..., None] + parent_skts[..., :3, 3:4])[..., 0]
    bone_norms = np.linalg.norm(joints_in_parents + 1e-6, axis=-1)
    bone_dirs = joints_in_parents / bone_norms[..., None]
    bones = bone_dirs
    bones[:, :1] = 0.0 # no rotation for root!

    c2ws = np.stack(c2ws).astype(np.float32)
    focals = np.stack(focals).astype(np.float32)
    centers = np.stack(centers).astype(np.float32)

    kp_idxs = np.arange(len(kp_idxs)).astype(np.int64)
    img_pose_indices = np.arange(len(kp_idxs)).astype(np.int64)

    np.save(f'{root_path}/{subject}_{split}_meta.npy', {'skel_type': skel_type}, allow_pickle=True)

    data_dict = {
        'img_pose_indices': img_pose_indices,
        'kp_idxs': kp_idxs,
        'c2ws': c2ws,
        'centers': centers,
        'focals': focals,
        'bones': bones,
        'kp3d': kp3d,
        'skts': skts,
        'cyls': cyls,
        'rest_pose': rest_poses[0],
        'rest_transform': rest_matrices[0],
    }

    for k in data_dict:
        if np.issubdtype(data_dict[k].dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(data_dict[k].dtype, np.integer):
            dtype = np.int64
        else:
            raise NotImplementedError(f'Unknown datatype for key {k}: {data_dict[k].dtype}')
        ds = h5_file.create_dataset(
            k, 
            data_dict[k].shape, 
            dtype,
            compression=compression
        )
        ds[:] = data_dict[k][:]
    h5_file.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for processing animal data')
    parser.add_argument("-s", "--subject", type=str, default='hare',
                        help="subject to process")
    parser.add_argument("--split", type=str, default='train',
                        help='split to process')
    parser.add_argument("--root_path", type=str, default='data/animal/',)
    parser.add_argument("--num_evals", type=int, default=0,
                        help='only used when we want to evaluate on a subset of the data')
    args = parser.parse_args()
    
    root_path = args.root_path
    subject = args.subject
    split = args.split

    print(f'Processing {subject} {split} data')
    data = process_animal_data(
        root_path=args.root_path, 
        subject=args.subject, 
        split=args.split,
        num_evals=args.num_evals,
    )
    print('process done!')
    write_to_h5py(os.path.join(root_path, f'{subject}_{split}.h5'), data)
