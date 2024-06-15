import os
import cv2
import glob

import numpy as np

import h5py
import imageio
import json

from core.utils.skeleton_utils import *
from core.utils.visualization import *
from core.networks.misc import *


def process_asmr_dataset(
    data_path='data/ASMR', 
    subject='Luis', 
    split='train',
    compression='gzip',
    chunk_size=64,
):
    # rotation to apply in world space to make -y the up-right direction
    ROOT_ROT = np.array(
        [[1, 0, 0, 0],
        [0, 0,-1, 0],
        [0, 1, 0 ,0],
        [0, 0, 0, 1],],
        dtype=np.float32
    )

    H, W = 1080, 1920

    basedir = os.path.join(data_path, subject)
    h5_path = os.path.join(data_path, f'{subject}_{split}.h5')

    #imgs = np.array([imageio.imread(p) for p in img_paths[:15]])

    # [< 128] == 9 [>= 128] == 1
    # Step 1. pose-related info
    rest_pose_dict = json.load(open(os.path.join(basedir, 'rest_pose_bones.json')))
    rest_transform = np.stack([v['tail'] for k, v in rest_pose_dict.items()], axis=0)
    rest_pose_tails = np.stack([v['tail'] for k, v in rest_pose_dict.items()], axis=0)[..., :3, -1]
    rest_heads = np.stack([v['head'] for k, v in rest_pose_dict.items()], axis=0)[..., :3, -1]
    rest_pose = rest_pose_tails.astype(np.float32)

    l2ws_dict = json.load(open(os.path.join(basedir, 'l2ws.json')))
    l2ws = np.stack([v['tail'] for k, v in l2ws_dict.items()], axis=1)
    # so that y is upright
    l2ws = ROOT_ROT[None, None] @ l2ws

    # align test local space to the canonical space
    # so that canonical space and local space is differ only by a translation
    skts = np.linalg.inv(l2ws)
    shift = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ], dtype=np.float32).reshape(1, 4, 4).repeat(len(rest_pose), axis=0)
    shift[..., :3, -1] = -rest_pose.copy()
    undo_transform = shift[None] @ rest_transform[None]
    skts = undo_transform @ skts
    l2ws = np.linalg.inv(skts)
    kp3d = l2ws[..., :3, -1]

    bone_rots = json.load(open(os.path.join(basedir, 'joint_rotations.json')))
    bone_rots = np.stack([v for k, v in bone_rots.items()], axis=1)

    cyls = get_kp_bounding_cylinder(kp3d, MixamoSkeleton, head='-y')

    # Step 2. camera-related info
    cam_data = json.load(open(os.path.join(basedir, 'cam_params.json')))
    Ks, c2ws, focals, centers = [], [], [], []

    for cam_id, cam_params in cam_data.items():
        # all the other entries are the same
        K = np.array(cam_params['K'])[0]
        focal = np.stack([K[0, 0], K[1, 1]], axis=-1)
        center = np.stack([K[0, 2], K[1, 2]], axis=-1)
        c2w = np.array(cam_params['Rt']).astype(np.float32)[0]
        c2w = ROOT_ROT @ c2w

        Ks.append(K)
        c2ws.append(c2w)
        focals.append(focal)
        centers.append(center)
    Ks = np.stack(Ks, axis=0).astype(np.float32)
    c2ws = np.stack(c2ws, axis=0).astype(np.float32)
    focals = np.stack(focals, axis=0).astype(np.float32)
    centers = np.stack(centers, axis=0).astype(np.float32)

    # create fake bkgd here.
    bkgd = np.zeros((1, H, W, 3), dtype=np.uint8)

    # set up image shape
    N_imgs = len(cam_data) * len(kp3d)
    img_shape = np.array((N_imgs, H, W, 3)).astype(np.int64)

    img_pose_indices = np.arange(len(cam_data))[:, None].repeat(len(kp3d), axis=1).reshape(-1)
    img_pose_indices = img_pose_indices.astype(np.int64)

    kp_idxs = np.arange(len(kp3d))[None, :].repeat(len(cam_data), axis=0).reshape(-1)
    kp_idxs = kp_idxs.astype(np.int64)

    # all the metadata to write
    data_dict = {
        'c2ws': c2ws,
        'bkgd_idxs': np.zeros((len(kp3d),)).astype(np.int32),
        'img_pose_indices': img_pose_indices,
        'kp_idxs': kp_idxs,
        'kp3d': kp3d,
        'bones': bone_rots.astype(np.float32),
        'skts': skts.astype(np.float32),
        'cyls': cyls.astype(np.float32),
        'rest_pose': rest_pose.astype(np.float32),
        'rest_heads': rest_heads.astype(np.float32),
        'focals': focals,
        'centers': centers,
        'K': Ks,
        'img_shape': img_shape,
    }

    if os.path.exists(h5_path):
        print(f'old {h5_path} exist, remove it')
        os.remove(h5_path)
    h5_file = h5py.File(h5_path, 'w')

    # first, write the basic data
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


    flatten_shape = (img_shape[0], H * W,)

    img_chunk = (1, chunk_size**2,)
    ds_imgs = h5_file.create_dataset(
        'imgs', 
        flatten_shape + (3,), 
        np.uint8, 
        chunks=img_chunk + (3,),
        compression=compression,
    )
    ds_masks = h5_file.create_dataset(
        'masks',
        flatten_shape + (1,),
        np.uint8,
        chunks=img_chunk + (1,),
        compression=compression,
    )
    ds_sampling_masks = h5_file.create_dataset(
        'sampling_masks',
        flatten_shape + (1,),
        np.uint8,
        chunks=(1, H * W, 1),
        compression=compression,
    )
    ds_bkgd = h5_file.create_dataset(
        'bkgds',
        (1, H * W, 3),
        np.uint8,
        chunks=(1, H * W, 3),
        compression=compression,
    )
    ds_bkgd[:] = bkgd.reshape(-1, H*W, 3)
    
    for c, cam_id in enumerate(cam_data.keys()):
        print(f'Processing camera {cam_id}')
        imgbase = os.path.join(basedir, cam_id)
        img_paths = sorted(glob.glob(os.path.join(imgbase, '*.jpg')))[:len(kp3d)]
        if len(img_paths) == 0:
            img_paths = sorted(glob.glob(os.path.join(imgbase, 'img*.png')))[:len(kp3d)]

        for i, p in enumerate(img_paths):
            if (i + 1) % 100 == 0:
                print(f'Processing {i + 1}/{len(img_paths)}')
            ptr = i + c * len(kp3d)
            img = imageio.imread(p)
            mask = imageio.imread(p.replace('jpg', 'png'))
            if mask.shape[-1] > 1:
                mask = mask[..., :1]

            mask[mask < 128] = 0
            mask[mask > 128] = 1
            kernel = np.ones((5, 5), np.uint8)
            sampling_mask = cv2.dilate(mask, kernel=kernel, iterations=2)
            ds_imgs[ptr] = img.reshape(-1, 3)
            ds_masks[ptr] = mask.reshape(-1, 1)
            ds_sampling_masks[ptr] = sampling_mask.reshape(-1, 1)

    print(f'Done, saved to {h5_path}')


if __name__ in '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-d", "--data_path", type=str, default='data/ASMR',
                        help='path to the dataset')
    parser.add_argument("-s", "--subject", type=str, default="Luis",
                        help='subject to extract')
    parser.add_argument("--split", type=str, default="train",
                        help='split to use')
    args = parser.parse_args()
    data_path = args.data_path
    subject = args.subject
    split = args.split

    process_asmr_dataset(
        data_path=data_path, 
        subject=subject, 
        split=split,
    )