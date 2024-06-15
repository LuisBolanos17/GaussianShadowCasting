import cv2
import copy
import time
import os, glob

import json
import h5py
import imageio

import torch
import numpy as np

from smplx import SMPL
from core.utils.skeleton_utils import *
from core.datasets.preprocess.process_spin import SMPL_JOINT_MAPPER


def read_cameras(data_path):
    
    intrinsic_paths = sorted(glob.glob(os.path.join(data_path, 'intrinsic', '*.txt')))
    c2w_paths = sorted(glob.glob(os.path.join(data_path, 'pose', '*.txt')))
    assert len(intrinsic_paths) == len(c2w_paths)
    
    intrinsics, c2ws = [], []
    for int_path, c2w_path in zip(intrinsic_paths, c2w_paths):
        intrinsics.append(np.loadtxt(int_path).astype(np.float32))
        c2ws.append(np.loadtxt(c2w_path).astype(np.float32))
    intrinsics = np.array(intrinsics)
    focals = np.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]],axis=-1)
    centers = intrinsics[..., :2, -1]
    return np.array(intrinsics), focals, centers, swap_mat(np.array(c2ws))


def read_poses(data_path, frames=None):
    json_paths = sorted(glob.glob(os.path.join(data_path, 'transform_smoth3e-2_withmotion', '*.json')))
    if frames is not None:
        json_paths = np.array(json_paths)[frames]
    kp3ds, poses, motions, joints_RTs, Rs, Ts = [], [], [], [], [], []
    for i, p in enumerate(json_paths):

        if i % 100 == 0:
            print(f'Reading poses {i}/{len(json_paths)}')
        with open(p, 'r') as f:
            json_data = json.load(f)
            kp3d = np.array(json_data['joints']).astype(np.float32)
            pose = np.array(json_data['pose']).reshape(-1, 3).astype(np.float32)
            joints_RT = (np.array(json_data['joints_RT']).astype(np.float32).transpose(2, 0, 1))
            # important: the rotation is from world-to-local. .T to make it local-to-worl
            R = np.array(json_data['rotation']).astype(np.float32).T
            T = np.array(json_data['translation']).astype(np.float32)
            motion = np.array(json_data['motion'])
            kp3ds.append(kp3d)
            poses.append(pose)
            joints_RTs.append(joints_RT)
            Ts.append(T)
            Rs.append(R)
            motions.append(motion)
    return np.array(kp3ds), np.array(poses), np.array(motions), np.array(joints_RTs), np.array(Rs), np.array(Ts)


@torch.no_grad()
def get_smpls_with_global_trans(
        bones,
        betas,
        Rg,
        Tg,
        smpl_path='smpl/SMPL_300',
        gender='MALE',
        joint_mapper=SMPL_JOINT_MAPPER,
        device=None,
    ):
    '''
    Rg: additional global rotation
    Tg: additional global transloation
    '''
    bones = torch.tensor(bones).float().clone().to(device)
    betas = torch.tensor(betas).float().clone().to(device)
    Rg = torch.tensor(Rg).float().to(device)
    Tg = torch.tensor(Tg).float().to(device)
    
    smpl_model = SMPL(
        smpl_path, 
        gender=gender, 
        num_betas=betas.shape[-1], 
        joint_mapper=SMPL_JOINT_MAPPER
    ).to(device)

    # directly incorporate global rotation R into pose parmaeter
    # Original equation: SMPL(pose) = Rp X + Tp
    # Neural actor additionally add global rotation/translation by: Rg (Rp X + Tp) + Tg
    # Now, we want Rk = RgRp, but still translate to the same location
    # -> (Rk X + Tp) - Tp + RgTp + Tg
    
    # Step 1: get Tp
    dummy = torch.eye(3).reshape(1, 1, 3, 3).expand(len(bones), 24, -1, -1).to(device)
    # assume the body has the same shape since they are the same person
    betas = betas.mean(0, keepdim=True) 
    rest_pose = smpl_model(
        betas=betas,
        body_pose=dummy[:, 1:], 
        global_orient=dummy[:, :1], 
        pose2rot=False,
    ).joints
    Tp = rest_pose[:, :1].clone()
    rest_pose = rest_pose[0].cpu().numpy()
    # center rest pose
    rest_pose -= rest_pose[:1] 
    # this is actually Tp^T Rg^T = Rg
    RgTp = Tp @ Rg.permute(0, 2, 1)
    
    # Step 2: make Rk
    axisang_Rg = rot_to_axisang(Rg)
    bones[:, :1] = axisang_Rg.reshape(-1, 1, 3)
    Rk = axisang_to_rot(bones)
    
    # Step 3: run SMPL to get (Rk X + Tp)
    print('running SMPL forward')
    RkX_Tp = smpl_model(
        betas=betas,
        body_pose=Rk[:, 1:], 
        global_orient=Rk[:, :1], 
        pose2rot=False,
    ).joints

    kp3d = (RkX_Tp - Tp  + RgTp + Tg).cpu().numpy()
    root_locs = kp3d[:, :1]
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose=rest_pose) for bone in bones.cpu().numpy()])
    l2ws[..., :3, -1] += root_locs
    skts = np.linalg.inv(l2ws)
    
    betas = betas.cpu().numpy()
    bones = bones.cpu().numpy()

    return betas, kp3d, bones, skts, rest_pose


def farthest_point_sampling(pts, n_pts=10, init_idx=0):
    idxs = np.zeros((n_pts,)).astype(np.int64)
    idxs[0] = init_idx
    
    distance = ((pts - pts[init_idx:init_idx+1])**2).sum(-1)
    for i in range(1, n_pts):
        idxs[i] = np.argmax(distance)
        d = ((pts - pts[idxs[i]:idxs[i]+1])**2).sum(-1)
        distance = np.where(d < distance, d, distance)
    return idxs


def process_neural_actor_data(
    data_path,
    save_path,
    subject='vlad',
    ext_scale=0.001,
    split='train',
    subsplit=0,
    frames=np.arange(100, 17001),
    training_views=None,
    test_views=[7, 18, 27, 40],
    n_views=20,
    H=940,
    W=1285,
    skel_type=SMPLSkeleton,
    compression='gzip',
    chunk_size=64,
    pre_extract_pose_path=None,
    shuffle=False,
):
    
    # set up path for data reading
    subject_path = os.path.join(data_path, subject)
    video_path = os.path.join(subject_path, 'training' if split.startswith('train') else 'testing')
    h5_path = os.path.join(save_path, f'{subject}_{split}_{subsplit}_shuffle.h5')
    
    _, focals, centers, c2ws = read_cameras(subject_path)
    views = training_views if split == 'train' else test_views
    if split == 'train':
        #views = training_views
        views = np.array([i for i in range(101) if i not in test_views])
    else:
        views = test_views
        
    if pre_extract_pose_path is not None:
        pre_extract_pose_path = os.path.join(subject_path, pre_extract_pose_path)
    # Note: the bones here do not have global rotation. We will process it later

    if not os.path.exists(pre_extract_pose_path):
        kp3ds, bones, _, _, Rs, Ts= read_poses(video_path, frames=frames)
        
        # read body shape from reference data
        betas = json.load(open(os.path.join(subject_path, 'raw_smpl', '000000.json'), 'r'))[0]['shapes']
        betas = np.array(betas).repeat(len(frames), 0)
        
        # compute pose-related data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Processing device {device} for {h5_path}')
        _, kp3d, bones, skts, rest_pose = get_smpls_with_global_trans(
            bones,
            betas,
            Rs, 
            Ts,
            device=device,
        )
        
        cyls = get_kp_bounding_cylinder(
            kp3d,
            ext_scale=ext_scale,
            skel_type=skel_type,
            extend_mm=250,
            top_expand_ratio=1.00,
            bot_expand_ratio=0.25,
            head='y'
        )
    else:
        pose_h5 = h5py.File(pre_extract_pose_path, 'r')
        betas = pose_h5['betas'][:]
        kp3d = pose_h5['kp3d'][:]
        bones = pose_h5['bones'][:]
        skts = pose_h5['skts'][:]
        rest_pose = pose_h5['rest_pose'][:]
        cyls = pose_h5['cyls'][:]
        pose_h5.close()
        assert len(frames) == len(kp3d)

    c2ws = c2ws[views]
    focals = focals[views]
    centers = centers[views]
    cam_idxs = np.arange(len(views)).reshape(-1, 1).repeat(len(frames), 1).reshape(-1)
    kp_idxs = np.arange(len(frames)).reshape(1, -1).repeat(len(views), 0).reshape(-1)

    # create a shuffled dataset
    if shuffle:
        shuffle_idxs = np.arange(len(kp_idxs))
        np.random.shuffle(shuffle_idxs)
        cam_idxs = cam_idxs[shuffle_idxs]
        kp_idxs = kp_idxs[shuffle_idxs]

        shuffle_inv_map = {}
        for i, s in enumerate(shuffle_idxs):
            shuffle_inv_map[s] = i

    # all data except for the images are ready.
    # since the amount of data is large, we can't collect all frames and write at once
    # have to read and write simultaneously

    if os.path.exists(h5_path):
        print(f'old {h5_path} exist, remove it')
        os.remove(h5_path)
    
    data_dict = {
        'c2ws': c2ws.astype(np.float32),
        'img_pose_indices': cam_idxs.astype(np.int64),
        'kp_idxs': kp_idxs.astype(np.int64),
        'centers': centers.astype(np.float32),
        'focals': focals.astype(np.float32),
        'kp3d': kp3d.astype(np.float32),
        'betas': betas.astype(np.float32),
        'bones': bones.astype(np.float32),
        'skts': skts.astype(np.float32),
        'cyls': cyls.astype(np.float32),
        'rest_pose': rest_pose.astype(np.float32),
    }

    if shuffle:
        data_dict['shuffle_idxs'] = shuffle_idxs
    
    h5_file = h5py.File(h5_path, 'w')
    img_shape = (len(frames) * len(views), H, W, 3)
    
    # first, write the basic data
    ds = h5_file.create_dataset('img_shape', (4,), np.int32)
    ds[:] = np.array(img_shape)
    
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

    # next, write image data
    # create datasets
    flatten_shape = (len(views) * len(frames), H * W,)
    #img_chunk = (1, 47*100,)
    img_chunk = (1, chunk_size**2,)
    ds_imgs = h5_file.create_dataset(
        'imgs', 
        flatten_shape + (4,), 
        np.uint8, 
        chunks=(1, H * W, 4), 
        compression=compression
    )

    # dilation kernel for mask
    d_kernel = np.ones((5, 5))
    bkgd = 255 * np.ones((H, W, 3), dtype=np.uint8)
    
    for i, view in enumerate(views):
        cur_video_path = os.path.join(video_path, 'rgb_video', f'{view:03d}.avi'), 'avi'
        try:
            reader = imageio.get_reader(
                os.path.join(video_path, 'rgb_video', f'{view:03d}.avi'), 'avi'
            )
        except:
            print(f'{cur_video_path}: broken when reading this')
            continue

        meta = reader.get_meta_data()
        n_frames = int(meta['fps'] * meta['duration'])
        
        view_ptr = len(frames) * i
        print(f'processing view {i}: cam {view}')
        # set the reader to the right starting point
        reader.get_data(0)
        for j, frame_ptr in enumerate(frames):
            try:
                img = reader.get_data(frame_ptr).copy()
            except:
                print(f'processing error frames {frame_ptr}')
                continue
            if j % 100 == 0:
                print(f'process frame {j:05d}/{len(frames):05d}')
            backsub = cv2.createBackgroundSubtractorMOG2()
            
            # for background subtraction
            backsub.apply(bkgd)
            mask = backsub.apply(img)[..., None]
            mask[mask < 127] = 0
            mask[mask >= 127] = 1

            save_ptr = view_ptr + j
            if shuffle:
                #save_ptr = shuffle_idxs[save_ptr]
                save_ptr = shuffle_inv_map[save_ptr]
            
            ds_imgs[save_ptr] = np.concatenate([img.reshape(H * W, 3), mask.reshape(H * W, 1)], axis=-1)
        reader.close()


if __name__ == '__main__':
    #from renderer import Renderer
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for processing neural actor data')
    parser.add_argument("-s", "--subject", type=str, default="vlad",
                        help='subject to extract')
    parser.add_argument("-p", "--path", type=str, 
                        default="/scratch/st-rhodin-1/users/shihyang/dataset/neuralactor",
                        help='path to save the .ht')
    parser.add_argument("--split", type=str, default="train",
                        help='split to use')
    parser.add_argument("--subsplit", type=str, default=0,
                        help='split to use')
    args = parser.parse_args()
    subject = args.subject
    split = args.split
    save_path = os.path.join(args.path, subject)

    data_path = 'data/neuralactor'
    print(f"Processing {subject}_{split}...")
    if split == 'train':
        #frames = np.arange(100, 17000+1)
        frames = np.arange(100, 17000+1)
        process_neural_actor_data(
            data_path=data_path, 
            save_path=save_path,
            subject=subject, 
            split=split,
            subsplit=args.subsplit,
            n_views=None,
            frames=frames,
            pre_extract_pose_path=f'{subject}_pose.h5',
            shuffle=True,
        )
    elif split == 'train_novelview':
        frames = np.arange(100, 17000+1)[::10]
        process_neural_actor_data(
            data_path=data_path, 
            save_path=save_path,
            subject=subject, 
            split=split,
            n_views=4,
            frames=frames,
        )
    elif split.startswith('test'):
        if split == 'test':
            frames = np.arange(100, 7000+1)[::10]
        elif split == 'test2':
            frames = np.arange(99, 7000)[::10]
        process_neural_actor_data(
            data_path=data_path, 
            save_path=save_path,
            subject=subject, 
            split='test',
            frames=frames,
        )
    else:
        raise NotImplementedError(f'Split {split} not defined')

