import os
import h5py
import torch
import imageio
import numpy as np
from smplx import SMPL

from core.datasets.preprocess.process_spin import SMPL_JOINT_MAPPER, write_to_h5py
from core.utils.skeleton_utils import *


# to align the ground plan to x-z plane
zju_to_nerf_rot = np.array([
    [1, 0, 0],
    [0, 0,-1],
    [0, 1, 0]
], dtype=np.float64)

num_train_frames = {
    '313': 60,
    '315': 400,
    '377': 300,
    '386': 300,
    '387': 300,
    '390': 300, # begin ith frame == 700
    '392': 300,
    '393': 300,
    '394': 300,
    '395': 300,
    '396': 540, # begin ith frame == 810
}

num_novel_pose_frames = {
    '313': 1000,
    '315': 1000,
    '377': 1000,
    '386': 1000,
    '387': 1000,
    '390': 700,
    '392': 1000,
    '393': 1000,
    '394': 1000,
}

begin_ith_frames = {
    '313': 0,
    '315': 0,
    '377': 0,
    '386': 0,
    '387': 0,
    '390': 700,
    '392': 0,
    '393': 0,
    '394': 0,
}


def get_mask(path, img_path, erode_border=False):
    '''
    Following NeuralBody repo
    https://github.com/zju3dv/neuralbody/blob/master/lib/datasets/light_stage/can_smpl.py#L46    '''

    mask_path = os.path.join(path, 'mask', img_path[:-4] + '.png')
    mask = None
    if os.path.exists(mask_path):
        mask = imageio.imread(mask_path)
        mask = (mask != 0).astype(np.uint8)

    mask_path = os.path.join(path, 'mask_cihp', img_path[:-4] + '.png')
    mask_cihp = None
    if os.path.exists(mask_path):
        mask_cihp = imageio.imread(mask_path)
        mask_cihp = (mask_cihp != 0).astype(np.uint8)
    
    if mask is not None and mask_cihp is not None:
        mask = (mask | mask_cihp).astype(np.uint8)
    elif mask_cihp is not None:
        mask = mask_cihp
    
    border = 5
    kernel = np.ones((border, border), np.uint8)
    #mask_erode = cv2.erode(mask.copy(), kernel)
    sampling_mask = cv2.dilate(mask.copy(), kernel, iterations=3)

    #sampling_mask = mask.copy()
    #sampling_mask[(mask_dilate - mask_erode) == 1] = 100
    if erode_border:
        dilated = cv2.dilate(mask.copy(), kernel) 
        eroded = cv2.erode(mask.copy(), kernel) 
        #sampling_mask[(dilated - eroded) == 1] = 0
        sampling_mask[(dilated - eroded) == 1] = 100 # indicating the boundary
    #eroded = cv2.erode(mask.copy(), kernel, iterations=1) 
    #mask = eroded.copy()

    return mask, sampling_mask

def post_process_mask(img, bg, fg_mask, sampling_mask, percentile=30):
    img = img.copy()
    bg = bg.copy()
    fg_mask = fg_mask.copy()
    sampling_mask = sampling_mask.copy()
    
    # first, set background color to remove shadowing/light effect
    masked_img = fg_mask * img.copy() + (1 - fg_mask) * bg
    
    # near boundary points are labeled with value=100
    bnd = (sampling_mask > 1) 
    
    # set boundary points to background color
    masked_img[bnd[..., 0] > 0, :] = bg[bnd[..., 0] > 0, :]
    
    # check how much difference is there if we replace the pixel with background
    diff = ((((masked_img - img)/255.)**2)**0.5).sum(-1)
    # don't care about none boundary point
    diff[bnd[..., 0] < 1] = 0
    # identify a threshold. If the error is lower than the threshold,
    # then the pixel probably belongs to background,
    # so we remove these pixels from background and add them to sampling mask
    threshold = np.percentile(diff[bnd[..., 0] > 0], percentile)
    # set the points that we don't care to have high error
    # -> these points wouldn't be removed from foreground, nor will they change
    #    the sampling mask.
    diff[bnd[..., 0] == 0] = threshold + 10
    fg_mask[diff < threshold, 0] = 0
    sampling_mask[diff < threshold, 0] = 1
    
    # set boundary indicator to 0, don't need them anymore
    sampling_mask[sampling_mask > 1] = 0
    return fg_mask, sampling_mask

@torch.no_grad()
def get_smpls(
    path, 
    kp_idxs, 
    gender='neutral', 
    ext_scale=1.0, 
    scale_to_ref=False,
    ref_pose=smpl_rest_pose, 
    param_path=None, 
    model_path=None, 
    vertices_path=None
):
    '''
    Note: it's yet-another-smpl-coordinate system
    bodies: the vertices from ZJU dataset
    '''

    if param_path is None:
        param_path = 'new_params'
    if model_path is None:
        model_path = 'smpl'
    if vertices_path is None:
        vertices_path = 'new_vertices'

    bones, betas, root_bones, root_locs = [], [], [], []
    zju_vertices = []
    for kp_idx in kp_idxs:
        params = np.load(os.path.join(path, param_path, f'{kp_idx}.npy'), allow_pickle=True).item()
        bone = params['poses'].reshape(-1, 24, 3)
        beta = params['shapes']
        # load the provided vertices
        zju_vert = np.load(os.path.join(path, vertices_path, f'{kp_idx}.npy')).astype(np.float32)

        zju_vertices.append(zju_vert)
        bones.append(bone)
        betas.append(beta)
        root_bones.append(params['Rh'])
        root_locs.append(params['Th'])

    bones = np.concatenate(bones, axis=0).reshape(-1, 3)
    betas = np.concatenate(betas, axis=0)
    root_bones = np.concatenate(root_bones, axis=0)
    # note: this is in global space, but different in ZJU's system
    Tp = torch.DoubleTensor(np.concatenate(root_locs, axis=0))

    # intentionally separate these for clarity
    Rn = torch.DoubleTensor(zju_to_nerf_rot[None]) # rotation to align ground plane to x-z
    zju_global_orient = axisang_to_rot(torch.DoubleTensor(root_bones))
    rots = axisang_to_rot(torch.DoubleTensor(bones)).view(-1, 24, 3, 3)
    rots[..., 0, :, :] = Rn @ zju_global_orient
    root_bones = rot_to_axisang(rots[..., 0, :, :].clone()).numpy()
    betas = torch.DoubleTensor(betas)

    # note that the rotation definition is different for ZJU_mocap
    # SMPL is joints = (RX + T), where the parenthesis implies the thing happened in SMPL()
    # but ZJU is joints =  R'(RX + T) + T', where R' and T' is the global rotation/translation
    # here, we directly include R' in R (and also a Rn to align the ground plane), so the vertices we get become
    # joints = (RnR'RX + T)
    # so to get the correct vertices in ZJU's system, we need to do
    # joints = (RnR'RX + T) - T + RnR'T + RnT'
    # WARNING: they define the T in R = 0 matrix!

    # 1. get T
    dummy = torch.zeros(1, 24, 3, 3).double()
    smpl = SMPL(model_path=model_path, gender=gender, joint_mapper=SMPL_JOINT_MAPPER)

    T = smpl(
        betas=betas.mean(0)[None].float(),
        body_pose=dummy[:, 1:].float(),
        global_orient=dummy[:, :1].float(),
        pose2rot=False
    ).joints[0, 0]

    # 2. get the rest pose
    dummy = torch.eye(3).view(1, 1, 3, 3).expand(-1, 24, -1, -1)

    rest_info = smpl(
        betas=betas.mean(0)[None].float(),
        body_pose=dummy[:, 1:].float(),
        global_orient=dummy[:, :1].float(),
        pose2rot=False
    )

    rest_verts = rest_info.vertices[0].numpy()
    rest_pose = rest_info.joints[0]
    rest_pose = rest_pose.numpy()
    rest_verts -= rest_pose[0]
    rest_pose -= rest_pose[0] # center rest pose


    # scale the rest pose if needed
    if scale_to_ref:
        ref_pose = ref_pose * ext_scale
        bone_len = calculate_bone_length(rest_pose).mean()
        ref_bone_len = calculate_bone_length(ref_pose).mean()
        pose_scale = ref_bone_len / bone_len
    else:
        pose_scale = 1.0
    rest_pose = rest_pose * pose_scale
    rest_verts = rest_verts * pose_scale

    # 3. get RhR'T
    T = T.view(1, 1, 3).double()
    RnRpT = (T @ rots[:, 0].permute(0, 2, 1))

    # 4. get the posed joints/vertices
    smpl_output = smpl(
        betas=betas.float(),
        body_pose=rots[:, 1:].float(),
        global_orient=rots[:, :1].float(),
        pose2rot=False
    )
    # apply (RhR'RX + T) - T + RhR'T + RhT'
    RnTp = (Rn @ Tp.view(-1, 3, 1)).view(-1, 1, 3)
    joints = smpl_output.joints.double() - T + RnRpT + RnTp
    vertices = smpl_output.vertices.double() - T + RnRpT + RnTp
    joints *= pose_scale
    vertices *= pose_scale

    root_locs = joints[:, 0].double().numpy()
    bones = bones.reshape(-1, 24, 3)
    bones[:, 0] = root_bones
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose=rest_pose) for bone in bones])
    l2ws[..., :3, -1] += root_locs[:, None]

    kp3d = l2ws[..., :3, -1]
    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])

    return betas, kp3d, bones, skts, rest_pose, rest_verts, pose_scale


def process_zju_data(
    data_path, 
    subject='377', 
    training_view=[0, 6, 12, 18],
    i_intv=1, split='train',
    ext_scale=0.001, res=None, skel_type=SMPLSkeleton
):
    '''
    ni, i_intv, intv, begin_i: setting from NeuralBody
    '''
    assert ext_scale == 0.001 # TODO: deal with this later
    # TODO: might have to check if this is true for all
    H = W = 1024 # default image size.
    ni = num_train_frames[subject]
    begin_i = begin_ith_frames[subject]

    if res is not None:
        H = int(H * res)
        W = int(W * res)

    subject_path = os.path.join(data_path, f"CoreView_{subject}")
    annot_path = os.path.join(subject_path, "annots.npy")
    annots = np.load(annot_path, allow_pickle=True).item()
    cams = annots['cams']
    num_cams = len(cams['K'])
    i = begin_i
    if split == 'train':
        view = training_view
        idxs = slice(i, i + ni * i_intv)
    elif split == 'calibrate':
        view = np.arange(num_cams)
        ni = 60
        i_intv = 10
        idxs = slice(i, i + ni * i_intv)
    elif split == 'novel_pose': # it's actually everything: novel view + novel pose
        view = [j for j in range(num_cams) if j not in training_view]
        #i = (i + ni) * i_intv
        i = 0
        i_intv = 30
        ni = num_novel_pose_frames[subject]
        #if subject == '390':
        #    i = 0
        idxs = slice(i, i + ni * i_intv)

    else:
        view = [1,4,5,10,17,20]
        if subject != '392':
            idxs = np.concatenate([np.arange(1, 31), np.arange(400, 601)])
        else:
            idxs = np.concatenate([np.arange(1, 31), np.arange(400, 556)])
        i_intv = 1


    # extract image and the corresponding camera indices
    img_paths = np.array([
        np.array(imgs_data['ims'])[view]
        for imgs_data in np.array(annots['ims'])[idxs][::i_intv]
    ]).ravel()

    cam_idxs = np.array([
        np.arange(len(imgs_data['ims']))[view]
        for imgs_data in np.array(annots['ims'])[idxs][::i_intv]
    ]).ravel()

    # Extract information from the dataset.
    imgs = np.zeros((len(img_paths), H, W, 3), dtype=np.uint8)
    masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    sampling_masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    kp_idxs = []
    for i, (img_path, cam_idx) in enumerate(zip(img_paths, cam_idxs)):

        if i % 50 == 0:
            print(f'{i+1}/{len(img_paths)}')

        K = np.array(cams['K'][cam_idx])
        D = np.array(cams['D'][cam_idx])

        img = imageio.imread(os.path.join(subject_path, img_path))
        mask, sampling_mask = get_mask(subject_path, img_path, erode_border=True)

        img = cv2.undistort(img, K, D)
        mask = cv2.undistort(mask, K, D)[..., None]
        sampling_mask = cv2.undistort(sampling_mask, K, D)[..., None]
        mask[mask > 1] = 1

        # resize the corresponding data as needed
        if res is not None and res != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            sampling_mask = cv2.resize(sampling_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * res
            mask = mask[..., None]
            sampling_mask = sampling_mask[..., None]

        if subject == '313' or subject == '315':
            kp_idx = int(os.path.basename(img_path).split('_')[4])
        else:
            kp_idx = int(os.path.basename(img_path)[:-4])

        imgs[i] = img
        masks[i] = mask
        
        sampling_masks[i] = sampling_mask
        kp_idxs.append(kp_idx)

    unique_cams = np.unique(cam_idxs)
    bkgds = np.zeros((num_cams, H, W, 3), dtype=np.uint8)
    # get background
    for c in unique_cams:
        cam_imgs = imgs[cam_idxs==c].reshape(-1, H, W, 3)
        cam_masks = masks[cam_idxs==c].reshape(-1, H, W, 1)
        N_cam_imgs = len(cam_imgs)
        for h_ in range(H):
            for w_ in range(W):
                vals = []
                is_bg = np.where(cam_masks[:, h_, w_] < 1)[0]
                if len(is_bg) == 0:
                    med = np.array([0, 0, 0]).astype(np.uint8)
                else:
                    med = np.median(cam_imgs[is_bg, h_, w_], axis=0)
                bkgds[c, h_, w_] = med.astype(np.uint8) 

    # get extrinsic data
    c2ws, focals, centers = [], [], []
    for c in range(num_cams):
        R = np.array(cams['R'][c])
        T = np.array(cams['T'][c]) / 1000. # in 1m system.
        K = np.array(cams['K'][c])

        # get camera-to-world matrix from extrinsic
        ext = np.concatenate([R, T], axis=-1)
        ext = np.concatenate([ext, np.array([[0, 0, 0., 1.]])], axis=0)
        c2w = np.linalg.inv(ext)
        c2w[:3, -1:] = zju_to_nerf_rot @ c2w[:3, -1:]
        c2w[:3, :3] = zju_to_nerf_rot @ c2w[:3, :3]
        c2ws.append(c2w)

        # save intrinsic data
        if res is not None:
            K[:2] = K[:2] * res
        focals.append([K[0, 0], K[1, 1]])
        centers.append(K[:2, -1])

    focals = np.array(focals)
    centers = np.array(centers)
    c2ws = np.array(c2ws).astype(np.float32)
    c2ws = swap_mat(c2ws) # to NeRF format

    # get pose-related data
    betas, kp3d, bones, skts, rest_pose, rest_verts, pose_scale = get_smpls(
        subject_path, 
        np.unique(kp_idxs),
        scale_to_ref=False
    )

    cyls = get_kp_bounding_cylinder(
        kp3d,
        ext_scale=ext_scale,
        skel_type=skel_type,
        extend_mm=250,
        top_expand_ratio=1.00,
        bot_expand_ratio=0.25,
        head='-y'
    )

    if split == 'test':
        kp_idxs = np.arange(len(kp_idxs))
    elif subject == '313' or subject == '315':
        kp_idxs = np.array(kp_idxs) - 1
    elif subject == '390':
        kp_idxs = np.array(kp_idxs) - 700
    kp_idxs = np.array(kp_idxs) // i_intv

    return {
        'imgs': np.array(imgs),
        'bkgds': np.array(bkgds),
        'bkgd_idxs': cam_idxs,
        'masks': np.array(masks).reshape(-1, H, W, 1),
        'sampling_masks': np.array(sampling_masks).reshape(-1, H, W, 1),
        'c2ws': c2ws.astype(np.float32),
        'img_pose_indices': cam_idxs,
        'kp_idxs': np.array(kp_idxs),
        'centers': centers.astype(np.float32),
        'focals': focals.astype(np.float32),
        'kp3d': kp3d.astype(np.float32),
        'betas': betas.numpy().astype(np.float32),
        'bones': bones.astype(np.float32),
        'skts': skts.astype(np.float32),
        'cyls': cyls.astype(np.float32),
        'rest_pose': rest_pose.astype(np.float32),
    }


def set_h36m_zju_config(ann_file, num_train_frame, num_eval_frame, begin_ith_frame=0, frame_interval=5,
                        smpl='new_smpl', params='new_params', vertices='new_vertices', erode_border=True, 
                        smpl_path='smplx'):
    return {
        'ann_file': ann_file, 
        'num_train_frame': num_train_frame,
        'num_eval_frame': num_eval_frame,
        'begin_ith_frame': begin_ith_frame, 
        'frame_interval': frame_interval,
        'smpl': smpl,
        'params': params,
        'vertices': vertices,
        'erode_border': erode_border,
        'smpl_path': smpl_path
    }

h36m_zju_configs = {
    'S1': set_h36m_zju_config('Posing/annots.npy', 150, 49),
    'S5': set_h36m_zju_config('Posing/annots.npy', 250, 127),
    'S6': set_h36m_zju_config('Posing/annots.npy', 150, 83),
    'S7': set_h36m_zju_config('Posing/annots.npy', 300, 200),
    'S8': set_h36m_zju_config('Posing/annots.npy', 250, 87),
    'S9': set_h36m_zju_config('Posing/annots.npy', 260, 133),
    'S11': set_h36m_zju_config('Posing/annots.npy', 200, 82),
}

h36m_zju_eval_frames = {
    """
    'S1': np.arange(49),
    'S5': np.arange(127),
    'S6': np.arange(83),
    'S7': np.arange(200),
    'S8': np.arange(87),
    'S9': np.arange(133),
    'S11': np.arange(82),
    """
    'S1': np.arange(34),
    'S5': np.arange(64),
    'S6': np.arange(39),
    'S7': np.arange(84),
    'S8': np.arange(57),
    'S9': np.arange(67),
    'S11': np.arange(48),
}


def process_h36m_zju_data(
    data_path, 
    subject='S1', 
    training_view=[0,1,2], 
    split='train', 
    res=None,
    ext_scale=0.001, 
    skel_type=SMPLSkeleton
):
    '''
    Note: they only use the Posing sequence for training
    '''
    H = 1000
    W = 1000 
    assert ext_scale == 0.001
    
    if res is not None and res != 1.0:
        H = int(H * res)
        W = int(W * res)
    
    config = h36m_zju_configs[subject]
    subject_path = os.path.join(data_path, f"{subject}")
    annot_path = os.path.join(subject_path, config['ann_file'])
    annots = np.load(annot_path, allow_pickle=True).item()
    subject_path = os.path.join(subject_path, 'Posing') # h36m-zju only use this sequence
    
    cams = annots['cams']
    num_cams = len(cams['K'])

    # following animatable NeRF's format
    i = config['begin_ith_frame']
    i_intv = config['frame_interval']
    ni = config['num_train_frame']
    
    if split == 'train':
        view = training_view
    else:
        
        view = np.array([i for i in range(num_cams) if i not in training_view])
        if len(view) == 0:
            view = [0]

        if split == 'test':
            i = config['begin_ith_frame'] + config['num_train_frame'] * i_intv
            ni = config['num_eval_frame']
        
    # extract image and the corresponding camera indices
    if split != 'anim':
        img_paths = np.array([
            np.array(imgs_data['ims'])[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        cam_idxs = np.array([
            np.arange(len(imgs_data['ims']))[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
    else: 
        # works differently: grab (1) novel view and (2) novel pose
        i = config['begin_ith_frame']
        i_intv = config['frame_interval']
        ni = config['num_train_frame']

        sample_interval = 6
        novel_view_img_paths = np.array([
            np.array(imgs_data['ims'])[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv][::sample_interval]  
        ]).ravel()

        novel_view_cam_idxs = np.array([
            np.arange(len(imgs_data['ims']))[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv][::sample_interval]  
        ]).ravel()

        i = config['begin_ith_frame'] + config['num_train_frame'] * i_intv
        ni = config['num_eval_frame']

        novel_pose_img_paths = np.array([
            np.array(imgs_data['ims'])[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv][::sample_interval]  
        ]).ravel()

        novel_pose_cam_idxs = np.array([
            np.arange(len(imgs_data['ims']))[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv][::sample_interval]  
        ]).ravel()

        img_paths = np.concatenate([novel_view_img_paths, novel_pose_img_paths])
        cam_idxs = np.concatenate([novel_view_cam_idxs, novel_pose_cam_idxs])
    
    kp_ids = []
    imgs = np.zeros((len(img_paths), H, W, 3), dtype=np.uint8)
    masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    sampling_masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)

    for i, (img_path, cam_idx) in enumerate(zip(img_paths, cam_idxs)):
        
        if i % 50 == 0:
            print(f'{i+1}/{len(img_paths)}')
        
        # get camera parameters
        K = np.array(cams['K'][cam_idx])
        D = np.array(cams['D'][cam_idx])
        
        # retrieve images
        img = imageio.imread(os.path.join(subject_path, img_path))
        mask, sampling_mask = get_mask(subject_path, img_path, erode_border=config['erode_border'])
        
        # process image and mask
        img = cv2.undistort(img, K, D)
        mask = cv2.undistort(mask, K, D)[..., None]
        sampling_mask = cv2.undistort(sampling_mask, K, D)[..., None]
        mask[mask > 1] = 1

        # resize the corresponding data as needed
        if res is not None and res != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            sampling_mask = cv2.resize(sampling_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * res

        # grab img_id for creating kp_idxs
        kp_id = img_path.split('/')[-1][:-4]
        imgs[i] = img
        masks[i] = mask
        sampling_masks[i] = sampling_mask
        kp_ids.append(kp_id)
    kp_ids = [int(kp_id) for kp_id in kp_ids]
    kp_ids, kp_idxs = np.unique(kp_ids, return_inverse=True)
    unique_cams = np.unique(cam_idxs)
    bkgds = np.zeros((num_cams, H, W, 3), dtype=np.uint8)

    # get background
    for c in unique_cams:
        cam_imgs = imgs[cam_idxs==c].reshape(-1, H, W, 3)
        cam_masks = masks[cam_idxs==c].reshape(-1, H, W, 1)
        N_cam_imgs = len(cam_imgs)
        for h_ in range(H):
            for w_ in range(W):
                vals = []
                is_bg = np.where(cam_masks[:, h_, w_] < 1)[0]
                if len(is_bg) == 0:
                    med = np.array([0, 0, 0]).astype(np.uint8)
                else:
                    med = np.median(cam_imgs[is_bg, h_, w_], axis=0)
                bkgds[c, h_, w_] = med.astype(np.uint8) 
    
    # get camera data
    c2ws, focals, centers = [], [], []
    for c in range(num_cams):
        R = np.array(cams['R'][c])
        T = np.array(cams['T'][c]) / 1000. # in 1m system.
        K = np.array(cams['K'][c])

        # get camera-to-world matrix from extrinsic
        ext = np.concatenate([R, T], axis=-1)
        ext = np.concatenate([ext, np.array([[0, 0, 0., 1.]])], axis=0)
        c2w = np.linalg.inv(ext)
        c2w[:3, -1:] = zju_to_nerf_rot @ c2w[:3, -1:]
        c2w[:3, :3] = zju_to_nerf_rot @ c2w[:3, :3]
        c2ws.append(c2w)

        # save intrinsic data
        if res is not None:
            K[:2] = K[:2] * res
        focals.append([K[0, 0], K[1, 1]])
        centers.append(K[:2, -1])

    focals = np.array(focals)
    centers = np.array(centers)
    c2ws = np.array(c2ws).astype(np.float32)
    c2ws = swap_mat(c2ws) # to NeRF format
    
    # get pose-related data
    betas, kp3d, bones, skts, rest_pose, vertices, pose_scale = get_smpls(
        subject_path, kp_ids,
        scale_to_ref=False,
        model_path=os.path.join(data_path, config['smpl_path'], 'smpl'),
        param_path=config['params'],
        vertices_path=config['vertices'],
    )
    cyls = get_kp_bounding_cylinder(
        kp3d,
        ext_scale=ext_scale,
        skel_type=skel_type,
        extend_mm=250,
        top_expand_ratio=1.00,
        bot_expand_ratio=0.25,
        head='-y'
    )

    return {
        'imgs': np.array(imgs),
        'bkgds': np.array(bkgds),
        'bkgd_idxs': cam_idxs,
        'masks': np.array(masks).reshape(-1, H, W, 1),
        'sampling_masks': np.array(sampling_masks).reshape(-1, H, W, 1),
        'c2ws': c2ws.astype(np.float32),
        'img_pose_indices': cam_idxs,
        'kp_idxs': np.array(kp_idxs),
        'centers': centers.astype(np.float32),
        'focals': focals.astype(np.float32),
        'kp3d': kp3d.astype(np.float32),
        'betas': betas.numpy().astype(np.float32),
        'bones': bones.astype(np.float32),
        'skts': skts.astype(np.float32),
        'cyls': cyls.astype(np.float32),
        'rest_pose': rest_pose.astype(np.float32),
    }

    
if __name__ == '__main__':
    #from renderer import Renderer
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-d", "--dataset", type=str, default='mocap')
    parser.add_argument("-s", "--subject", type=str, default="377",
                        help='subject to extract')
    parser.add_argument("--split", type=str, default="train",
                        help='split to use')
    args = parser.parse_args()
    dataset = args.dataset
    subject = args.subject
    split = args.split

    if dataset == 'mocap':
        data_path = 'data/zju_mocap/'
        print(f"Processing {subject}_{split}...")
        data = process_zju_data(data_path, subject, split=split, res=0.5)
    elif dataset == 'h36m':
        data_path = 'data/h36m_zju'
        print(f"Processing {subject}_{split}...")
        data = process_h36m_zju_data(data_path, subject, split=split, res=1.0)
    else:
        raise NotImplementedError(f'Unknown dataset {dataset}')

    write_to_h5py(os.path.join(data_path, f"{subject}_{split}_cal.h5"), data)
