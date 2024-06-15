import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import h5py
import torch
import hydra
import imageio
import numpy as np
import mcubes, trimesh
import math

from tqdm import tqdm, trange

from core.trainer import to_device
from core.utils.skeleton_utils import *
from core.networks.lighting import ImportanceHDRiLight, DirectionalLightRelight
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

# TODO: these functions should actually come from elsewhere
from train import (
    build_model,
    find_ckpts,
)
from hydra.utils import instantiate


CONFIG_BASE = 'configs/render'


def get_camera_motion(*args, camera_trajectory: str = 'bubble', **kwargs):
    if camera_trajectory == 'bubble':
        return get_camera_motion_bubble(*args, **kwargs)
    elif camera_trajectory == 'bullet':
        return get_camera_motion_bullet(*args, **kwargs)


def get_camera_motion_bubble(c2w: np.ndarray, z_t: float = 0.1, x_deg: float = 15., y_deg: float = 25., n_step: int = 30):
    x_rad = x_deg * np.pi / 180.
    y_rad = y_deg * np.pi / 180.
    motions = np.linspace(0., 2 * np.pi, n_step, endpoint=True)
    z_t = z_t * c2w[2, -1]

    x_motions = (np.cos(motions) - 1.) * x_rad
    y_motions = np.sin(motions) * y_rad
    z_trans = (np.sin(motions) + 1.) * z_t
    cam_motions = []
    for x_motion, y_motion in zip(x_motions, y_motions):
        cam_motion = rotate_x(x_motion) @ rotate_y(y_motion)
        cam_motions.append(cam_motion)

    bubbles = []
    for cam_motion, z_tran in zip(cam_motions, z_trans):
        c = c2w.copy()
        c[2, -1] += z_tran
        bubbles.append(cam_motion @ c)
    return np.stack(bubbles, axis=0)

def get_camera_motion_bullet(c2w, n_bullet=30, axis='y'):
    if axis == 'y':
        rotate_fn = rotate_y
    elif axis == 'x':
        rotate_fn = rotate_x
    elif axis == 'z':
        rotate_fn = rotate_z
    else:
        raise NotImplementedError(f'rotate axis {axis} is not defined')

    y_angles = np.linspace(0, math.radians(360), n_bullet+1)[:-1]
    c2ws = []
    for a in y_angles:
        c = rotate_fn(a) @ c2w
        c2ws.append(c)
    return np.array(c2ws)


class BaseRenderDataset(Dataset):
    """ Base dataset for spitting out render data
    Detail contents should be implemented separately
    """
    def __init__(
        self,
        h5_path,
        idxs,
        bkgd_to_use=None,
        indexing='image',
        resolution=(1000, 1000),
        retrieve_gt_imgs=False,
        cam_overwrite=None,
        cam_scale=1.0,
        pose_scale=None,
        undo_pose_scale=False,
        apply_pose_scale=False,
        center_kps=False,
        rotate_y_deg=0.,
        rotate_x_deg=0.,
        rotate_z_deg=0.,
        cam_dist=1.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        h5_path: str, path to .h5 dataset
        idxs: list of int, specifying the index of data to read from
        bkgd_to_use: str, type of background to use. Option: 'white', 'black', None.
                     None uses the background from the dataset.
        indexing: str, how we read the dataset. 'body_pose' means that idxs is used 
                  for finding body_poses 
        resolution: tuple of int, resolution of the output image
        cam_overwrite: overwrite the returned camera index.
        """
        super().__init__()
        self.h5_path = h5_path
        if idxs is not None:
            self.idxs = eval(idxs) # TODO: is there a better way?
        self.bkgd_to_use = bkgd_to_use
        self.indexing = indexing
        self.retrieve_gt_imgs = retrieve_gt_imgs
        self.resolution = np.array(resolution)
        self.dataset = None
        self.cam_overwrite = cam_overwrite
        self.cam_scale = cam_scale
        self.undo_pose_scale = undo_pose_scale
        self.apply_pose_scale = apply_pose_scale
        self.pose_scale = pose_scale 
        self.center_kps = center_kps
        self.rotate_y_deg = rotate_y_deg
        self.rotate_x_deg = rotate_x_deg
        self.rotate_z_deg = rotate_z_deg
        self.cam_dist = cam_dist

        self.init_meta()
    

    def __len__(self):
        return len(self.idxs)

    def init_dataset(self):

        if self.dataset is not None:
            return
        self.dataset = h5py.File(self.h5_path, 'r')
    
    def init_meta(self):
        self.skel_type = SMPLSkeleton
        dataset = h5py.File(self.h5_path, 'r')

        img_shape = dataset['img_shape'][:]
        HW = img_shape[1:3]
        self.HW = HW

        print('reading camera data')
        self.c2ws = dataset['c2ws'][:].astype(np.float32)


        if 'centers' in dataset:
            self.centers = dataset['centers'][:].astype(np.float32)
        else:
            self.centers = (self.HW[::-1] // 2)[None].repeat(len(self.c2ws), 0).astype(np.float32)
        self.focals = dataset['focals'][:].astype(np.float32)

        if 'img_pose_indices' in dataset:
            self.cam_idxs = dataset['img_pose_indices'][:]
        else:
            self.cam_idxs = np.arange(len(self.c2ws))

        print('reading pose data')
        self.bones = dataset['bones'][:].astype(np.float32)
        self.kp3d = dataset['kp3d'][:].astype(np.float32)
        self.root_locs = self.kp3d[:, self.skel_type.root_id].astype(np.float32)

        if self.undo_pose_scale:
            assert self.pose_scale is None
            self.pose_scale = dataset['pose_scale'][()]

        if 'kp_idxs' in dataset:
            self.kp_idxs = dataset['kp_idxs'][:]
        else:
            """ 
            assert len(dataset['kp3d']) == len(dataset['imgs']), \
                  f'kp_idxs not provided in the dataset and the image and pose lengths do not match' \
                    f'{len(dataset["kp3d"])} vs {len(dataset["imgs"])}'
            self.kp_idxs = np.arange(len(dataset['kp3d']))
            """
            # TODO: hack, fix it later
            self.kp_idxs = np.arange(len(dataset['kp3d']))[None].repeat(9, 0).reshape(-1)

        if self.bkgd_to_use is None:
            self.bgs = dataset['bkgds'][:].reshape(-1, *HW, 3).astype(np.uint8)
            self.bg_idxs = dataset['bkgd_idxs'][:].astype(np.int64)
        else:
            if self.bkgd_to_use == 'white':
                self.bgs = np.ones((1, *HW, 3), dtype=np.uint8) * 255
            elif self.bkgd_to_use == 'black':
                self.bgs = np.zeros((1, *HW, 3), dtype=np.uint8)
            else:
                raise NotImplementedError(f'Unknow bkgd type {self.bkgd_to_use}')
            if 'bkgd_idxs' in dataset:
                self.bg_idxs = np.zeros((len(dataset['bkgd_idxs']),)).astype(np.int64)
            else:
                self.bg_idxs = np.zeros((len(self.c2ws),)).astype(np.int64)
        dataset.close()

        if len(self.focals.shape) < 2:
            self.focals = self.focals[:, None].repeat(2, 1)

        if self.cam_scale != 1.0:
            self.c2ws[..., :3, -1] /= self.cam_scale
            print('Camera scale done')

        # dealing with different resolution
        if (HW != self.resolution).any():
            scale_hw = (self.resolution / HW).astype(np.float32)
            scale_wh = scale_hw[::-1].reshape(1, 2)
            self.focals = self.focals * scale_wh
            if self.centers is not None:
                self.centers = self.centers * scale_wh

            bgs = []
            for bg in self.bgs:
                bg = cv2.resize(bg, self.resolution[::-1].tolist(), interpolation=cv2.INTER_AREA)
                bgs.append(bg)
            self.bgs = np.stack(bgs, axis=0)
        
        if self.pose_scale is not None and self.undo_pose_scale:
            self.kp3d = self.kp3d.copy() / self.pose_scale
            self.root_locs = self.kp3d[:, self.skel_type.root_id].astype(np.float32)
            self.c2ws[..., :3, -1] /= self.pose_scale
            print('Pose scale undone')
        

        if self.pose_scale is not None and self.apply_pose_scale:
            assert not self.undo_pose_scale
            self.kp3d = self.kp3d.copy() * self.pose_scale
            self.root_locs = self.kp3d[:, self.skel_type.root_id].astype(np.float32)
            self.c2ws[..., :3, -1] *= self.pose_scale
            print('Pose scale applied')
        
    def __getitem__(self, idx):
        if self.dataset is None:
            self.init_dataset()
        idx = self.idxs[idx]
        img = self.dataset['imgs'][idx].reshape(*self.HW, 3)
        bone, root_loc = self.get_pose_data(idx) 
        c2w, K, focal, center, cam_idx = self.get_camera_data(idx)
        bg = self.get_bg(idx)

        if self.cam_overwrite is not None:
            cam_idx = cam_idx * 0 + self.cam_overwrite

        if self.center_kps:
            c2w = c2w.copy()
            c2w[..., :3, -1] -= root_loc
            root_loc = root_loc.copy() * 0
            if self.cam_dist != 1.0:
                c2w[..., :3, -1] *= self.cam_dist
        
        if self.rotate_y_deg != 0.:
            rotate_y_mat = rotate_y(math.radians(self.rotate_y_deg))
            c2w = rotate_y_mat @ c2w

        if self.rotate_x_deg != 0.:
            rotate_x_mat = rotate_x(math.radians(self.rotate_x_deg))
            c2w = rotate_x_mat @ c2w
        
        if self.rotate_z_deg != 0.:
            rotate_z_mat = rotate_z(math.radians(self.rotate_z_deg))
            c2w = rotate_z_mat @ c2w

        return {
            'img': img,
            'c2ws': c2w,
            'K': K,
            'root_locs': root_loc,
            'bones': bone,
            'hwf': (*self.resolution, focal),
            'center': center,
            'cam_idxs': cam_idx,
            'bgs': bg,
        }
    
    def get_pose_data(self, idx):
        kp_idx = self.kp_idxs[idx]
        root_loc = self.root_locs[kp_idx]
        bone = self.bones[kp_idx]

        return bone, root_loc 
    
    def get_camera_data(self, idx):
        cam_idx = self.cam_idxs[idx]
        c2w = self.c2ws[cam_idx]
        focal = self.focals[cam_idx]

        center = None
        if self.centers is not None:
            center = self.centers[cam_idx]
            K = build_intrinsic(center[0], center[1], focal)
        else:
            K = build_intrinsic(self.HW[1] * 0.5, self.HW[0] * 0.5, focal)

        return c2w, K, focal, center, cam_idx
    
    def get_bg(self, idx):
        bkgd_idx = self.bg_idxs[idx]
        return (self.bgs[bkgd_idx] / 255.).astype(np.float32)

class ASMRRenderDataset(BaseRenderDataset):

    def __init__(
        self,
        *args,
        use_T_rest_pose: bool = False,
        **kwargs,
    ):
        self.use_T_rest_pose = use_T_rest_pose
        super(ASMRRenderDataset, self).__init__(*args, **kwargs)
    
    def init_meta(self, *args, **kwargs):
        super().init_meta(*args, **kwargs)
        self.skel_type = MixamoSkeleton
        dataset = h5py.File(self.h5_path, 'r')
        rest_pose = dataset['rest_pose'][:]
        rest_heads = dataset['rest_heads'][:]
        self.skts = dataset['skts'][:]
        self.kp3d = dataset['kp3d'][:]
        dataset.close()

        if self.use_T_rest_pose:
            tpose_rest_pose, tpose_rest_heads, skts_T = A_pose_to_T_pose(
                rest_pose,
                rest_heads,
                self.kp3d,
                self.skts,
                MixamoSkeleton,
            )
            self.rest_pose = tpose_rest_pose.astype(np.float32)
            self.rest_heads = tpose_rest_heads.astype(np.float32)
            self.skts = skts_T.astype(np.float32)

    def __getitem__(self, idx):

        ret = super(ASMRRenderDataset, self).__getitem__(idx)
        kp_idx = self.kp_idxs[self.idxs[idx]]
        ret['kp3d'] = self.kp3d[kp_idx].copy()
        ret['skts'] = self.skts[kp_idx].copy()
        return ret
    
class RelightASMRRenderDataset(ASMRRenderDataset):

    def __init__(
        self,
        *args,
        use_T_rest_pose: bool = False,
        angles: np.array = np.array([0]),
        **kwargs
    ):
        self.angles = angles
        self.use_T_rest_pose = use_T_rest_pose
        super(RelightASMRRenderDataset, self).__init__(*args, **kwargs)
    
    def init_meta(self, *args, **kwargs):
        super().init_meta(*args, **kwargs)

    def __getitem__(self, idx):
        ret = super(RelightASMRRenderDataset, self).__getitem__(idx)
        angle = self.angles[idx % len(self.angles)]
        ret['angle'] = angle
        return ret


class AnimalRenderDataset(BaseRenderDataset):
    subject_skel_type = {
        'hare': HARESkeleton,
        'wolf': WOLFSkeleton,
    }
    def __init__(
        self,
        *args,
        subject='hare',
        **kwargs,
    ):
        self.subject = subject
        super(AnimalRenderDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        super().init_meta()
        self.skel_type = self.subject_skel_type[self.subject] 
        dataset = h5py.File(self.h5_path, 'r')
        skts = dataset['skts'][:]
        self.rest_pose = rest_pose = dataset['rest_pose'][:]
        self.rest_transform = rest_transform = dataset['rest_transform'][:]
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
        self.kp3d = np.linalg.inv(self.skts)[..., :3, -1]
        self.bones = self.create_bones(skts, self.kp3d)

        dataset.close()

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
        print(bones.shape)
        return bones

    def __getitem__(self, idx):
        ret = super(AnimalRenderDataset, self).__getitem__(idx)
        kp_idx = self.kp_idxs[self.idxs[idx]]
        ret['kp3d'] = self.kp3d[kp_idx].copy()
        ret['skts'] = self.skts[kp_idx].copy()
        return ret

class RelightAnimalRenderDataset(AnimalRenderDataset):

    def __init__(
        self,
        *args,
        angles: np.array = np.array([0]),
        **kwargs
    ):
        self.angles = angles
        super(RelightAnimalRenderDataset, self).__init__(*args, **kwargs)
    
    def init_meta(self, *args, **kwargs):
        super().init_meta(*args, **kwargs)

    def __getitem__(self, idx):
        ret = super(RelightAnimalRenderDataset, self).__getitem__(idx)
        angle = self.angles[idx % len(self.angles)]
        ret['angle'] = angle
        return ret

# TODO: novel view rendering
class NovelViewRenderDataset(BaseRenderDataset):
    """ To create novel views for selected poses with pre-defined trajectory.
    Note that the pose is fixed
    """
    trajectories = ['bullet', 'bubble']
    def __init__(
        self,
        *args,
        center_pose=True,
        camera_trajectory='bullet',
        cam_kwargs={},
        **kwargs,
    ):
        assert camera_trajectory is not None
        assert camera_trajectory in self.trajectories
        self.center_pose = center_pose
        self.camera_trajectory = camera_trajectory
        self.cam_kwargs = cam_kwargs
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.c2ws)

    def init_meta(self):
        super().init_meta()

        # ingredients
        # c2ws
        # K
        # hwf
        # center
        # cam_idxs

        # step 1: collects all cameras and poses
        bones, root_locs = [], []
        c2ws, focals, centers, cam_idxs = [], [], [], []
        for i in self.idxs:
            bone, root_loc = super().get_pose_data(i)
            c2w, K, focal, center, cam_idx = super().get_camera_data(i)

            bones.append(bone)
            root_locs.append(root_loc)
            c2ws.append(c2w)
            focals.append(focal)
            centers.append(center)
            cam_idxs.append(cam_idx)
    
        bones = np.stack(bones, axis=0)
        root_locs = np.stack(root_locs, axis=0)
        c2ws = np.stack(c2ws, axis=0)
        focals = np.stack(focals, axis=0)
        centers = np.stack(centers, axis=0)
        cam_idxs = np.stack(cam_idxs, axis=0)

        motion_c2ws = []
        motion_bones, motion_root_locs = [], []
        motion_focals, motion_centers, motion_cam_idxs = [], [], []
        for i, c2w in enumerate(c2ws):
            if self.center_pose:
                c2w = c2w.copy()
                c2w[..., :3, -1] = c2w[..., :3, -1] - root_locs[i]
            motion_c2ws_ = get_camera_motion(
                c2w=c2w,
                camera_trajectory=self.camera_trajectory,
                **self.cam_kwargs,
            ).tolist()

            motion_focal = focals[i:i+1].repeat(len(motion_c2ws_), axis=0)
            motion_center = centers[i:i+1].repeat(len(motion_c2ws_), axis=0)
            # override the camera index
            motion_cam_idx = np.arange(len(motion_c2ws_)) + i * len(motion_c2ws_)
            motion_bone = bones[i:i+1].repeat(len(motion_c2ws_), axis=0)
            motion_root_loc = root_locs[i:i+1].repeat(len(motion_c2ws_), axis=0)
            if self.center_pose:
                motion_root_loc = np.zeros_like(motion_root_loc)

            motion_c2ws.extend(motion_c2ws_)
            motion_focals.extend(motion_focal)
            motion_centers.extend(motion_center)
            motion_cam_idxs.extend(motion_cam_idx)
            motion_bones.extend(motion_bone)
            motion_root_locs.extend(motion_root_loc)
        
        self.c2ws = np.array(motion_c2ws).astype(np.float32)
        self.focals = np.array(motion_focals).astype(np.float32)
        self.centers = np.array(motion_centers).astype(np.float32)
        self.cam_idxs = np.array(motion_cam_idxs)
        self.bones = np.array(motion_bones).astype(np.float32)
        self.root_locs = np.array(motion_root_locs).astype(np.float32)

        self.kp_idxs = np.arange(len(self.bones))
        self.bg_idxs = np.zeros((len(self.bones),), dtype=np.int64)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.init_dataset()
        #idx = self.idxs[idx] # don't need this no more
        bone, root_loc = self.get_pose_data(idx) 
        c2w, K, focal, center, cam_idx = self.get_camera_data(idx)
        bg = self.get_bg(idx)

        if self.cam_overwrite is not None:
            cam_idx = cam_idx * 0 + self.cam_overwrite

        return {
            'c2ws': c2w,
            'K': K,
            'root_locs': root_loc,
            'bones': bone,
            'hwf': (*self.resolution, focal),
            'center': center,
            'cam_idxs': cam_idx,
            'bgs': bg,
        }

    def get_bg(self, idx):
        bkgd_idx = self.bg_idxs[idx]
        return (self.bgs[bkgd_idx] / 255.).astype(np.float32)

# TODO: pose interpolation?
class PoseInterpolateRenderDataset(BaseRenderDataset):

    def __init__(
        self,
        *args,
        num_interp=5,
        **kwargs,
    ):
        self.num_interp = num_interp
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.c2ws)

    def init_meta(self):
        super().init_meta()

        # step 1: collects all cameras and poses
        bones, root_locs = [], []
        c2ws, focals, centers, cam_idxs = [], [], [], []
        for i in self.idxs:
            bone, root_loc = super().get_pose_data(i)
            c2w, K, focal, center, cam_idx = super().get_camera_data(i)

            bones.append(bone)
            root_locs.append(root_loc)
            c2ws.append(c2w)
            focals.append(focal)
            centers.append(center)
            cam_idxs.append(cam_idx)
    
        bones = np.stack(bones, axis=0)
        root_locs = np.stack(root_locs, axis=0)
        c2ws = np.stack(c2ws, axis=0)
        focals = np.stack(focals, axis=0)
        centers = np.stack(centers, axis=0)
        cam_idxs = np.stack(cam_idxs, axis=0)

        motion_c2ws = []
        motion_bones, motion_root_locs = [], []
        motion_focals, motion_centers, motion_cam_idxs = [], [], []

        for i, (bone, root_loc) in enumerate(zip(bones, root_locs)):
            if i + 1 >= len(bones):
                motion_bones.extend(bone[None])
                motion_root_locs.extend(root_loc[None])
                motion_c2ws.extend(c2ws[i:i+1])
                motion_focals.extend(focals[i:i+1])
                motion_centers.extend(centers[i:i+1])
                motion_cam_idxs.extend(cam_idxs[i:i+1])
                break
            
            bone_next, root_loc_next = bones[i+1], root_locs[i+1]
            bone = torch.tensor(bone)
            bone_next = torch.tensor(bone_next)
            bone_6d = axisang_to_rot6d(bone)
            bone_next_6d = axisang_to_rot6d(bone_next)
            steps = torch.linspace(0., 1., self.num_interp)[..., :-1].reshape(-1, 1, 1)


            interp_bones = bone_6d[None] * (1 - steps) + bone_next_6d[None] * steps
            interp_bones = rot6d_to_axisang(interp_bones.reshape(-1, 6)).reshape(self.num_interp-1, -1, 3)
            interp_bones = interp_bones.cpu().numpy()
            steps = steps.reshape(-1, 1).cpu().numpy()
            interp_root_loc = root_loc[None] * (1 - steps) + root_loc_next[None] * steps

            motion_c2w = c2ws[i:i+1].repeat(self.num_interp, axis=0)
            motion_focal = focals[i:i+1].repeat(self.num_interp, axis=0)
            motion_center = centers[i:i+1].repeat(self.num_interp, axis=0)
            motion_cam_idx = i + 0 * cam_idxs[i:i+1].repeat(self.num_interp, axis=0)

            motion_bones.extend(interp_bones)
            motion_root_locs.extend(interp_root_loc)
            motion_c2ws.extend(motion_c2w)
            motion_focals.extend(motion_focal)
            motion_centers.extend(motion_center)
            motion_cam_idxs.extend(motion_cam_idx)

        self.c2ws = np.array(motion_c2ws).astype(np.float32)
        self.focals = np.array(motion_focals).astype(np.float32)
        self.centers = np.array(motion_centers).astype(np.float32)
        self.cam_idxs = np.array(motion_cam_idxs)
        self.bones = np.array(motion_bones).astype(np.float32)
        self.root_locs = np.array(motion_root_locs).astype(np.float32)

        self.kp_idxs = np.arange(len(self.bones))
        self.bg_idxs = np.zeros((len(self.bones),), dtype=np.int64)

        self.kp_idxs = np.arange(len(self.bones))
        self.bg_idxs = np.zeros((len(self.bones),), dtype=np.int64)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.init_dataset()
        #idx = self.idxs[idx] # don't need this no more
        bone, root_loc = self.get_pose_data(idx) 
        c2w, K, focal, center, cam_idx = self.get_camera_data(idx)
        bg = self.get_bg(idx)

        if self.cam_overwrite is not None:
            cam_idx = cam_idx * 0 + self.cam_overwrite

        return {
            'c2ws': c2w,
            'K': K,
            'root_locs': root_loc,
            'bones': bone,
            'hwf': (*self.resolution, focal),
            'center': center,
            'cam_idxs': cam_idx,
            'bgs': bg,
        }


class HDRIDataset(BaseRenderDataset):

    def __init__(
        self,
        *args,
        angles: np.array = np.array([0]),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.angles = angles

    def init_meta(self):
        super().init_meta()

        # self.c2ws = np.array([
        #     [1.,  0.,  0.,  0.],
        #     [0., -1.,  0.,  -0.125],
        #     [0.,  0., -1., -4.],
        #     [0.,  0.,  0.,  1.],
        # ]).astype(np.float32)[None].repeat(len(self.idxs), 0)
        # self.focals = np.array([800]*len(self.idxs)).astype(np.float32)
    
    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        angle = self.angles[idx % len(self.angles)]
        ret['angle'] = angle
        x, y, w, h = 0, 0, self.resolution[1], self.resolution[0]
        ret['bboxes'] = np.array([x, y, w, h]).astype(np.int32)
        return ret


class MeshRenderDataset(BaseRenderDataset):

    def __init__(
        self,
        *args,
        mcube_resolution=256,
        mcube_pts_range=1.3,
        **kwargs,
    ):
        """
        Parameters
        ----------
        mcube_resolution: int, resolution for matching cube 
        mcube_pts_range: float, range of the coordinates of the matching cube
        """
        self.mcube_resolution = mcube_resolution
        self.mcube_pts_range = mcube_pts_range
        super(MeshRenderDataset, self).__init__(*args, **kwargs)
    
    def init_meta(self):
        super(MeshRenderDataset, self).init_meta()

        res = self.mcube_resolution
        radius = self.mcube_pts_range
        t = np.linspace(-radius, radius, res+1)
        grid_pts = np.stack(np.meshgrid(t, t, t), axis=-1).astype(np.float32)

        sh = grid_pts.shape
        self.grid_pts = grid_pts
    
    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        root_locs = ret['root_locs']
        ret['pts'] = root_locs + self.grid_pts
        return ret
    
def load_trained_model(config, ckpt_path=None):

    # TODO: config these in a different way?
    expname, basedir = config.expname, config.basedir
    log_path = os.path.join(basedir, expname)
    data_attrs = instantiate(config.dataset).get_meta()

    ckpt, _ = find_ckpts(config, log_path, ckpt_path) 
    model = build_model(config.model, data_attrs, ckpt)
    return model


def build_render_dataset(config):
    dataset = instantiate(config)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        num_workers=config.get('num_workers', 2), 
        shuffle=False
    )

    return dataloader


def render(config):

    forward_type = config.get('forward_type', 'render')
    raychunk = config.get('raychunk', 1024 * 4)

    render_dataloader = build_render_dataset(config.render_dataset)
    model_config = OmegaConf.load(config.model_config)
    # model_config.model.pts_config.block_irrel=False
    model = load_trained_model(model_config, ckpt_path=config.get('ckpt_path', None))
    save_gt = config.get('save_gt', False)
    output_path = config.output_path
    img_path, mesh_path = '', ''
    render_normal = False
    if forward_type == 'render':
        img_path = os.path.join(output_path, 'images')
        os.makedirs(img_path, exist_ok=True)
        gt_img_path = os.path.join(output_path, 'gt_images')
        os.makedirs(gt_img_path, exist_ok=True)
    elif forward_type == 'render_relight':
        img_path = os.path.join(output_path, 'images')
        os.makedirs(img_path, exist_ok=True)
        gt_img_path = os.path.join(output_path, 'gt_images')
        os.makedirs(gt_img_path, exist_ok=True)
        mode = config.get('relight_mode', 'directional')
        floor_point = torch.Tensor(config.get('floor_point', [0,0,0]))
        floor_normal = torch.Tensor(config.get('floor_normal', [0,-1,0]))
        if mode == 'hdri':
            assert config.hdri_path is not None
            intensity = config.get('intensity', 1.5)
            model.light = ImportanceHDRiLight(hdri_resolution=config.hdri_resolution, num_secondary_rays=config.num_secondary_rays, hdri_path=config.hdri_path, exposure=intensity)
        elif mode == 'directional':
            ambient = torch.Tensor(config.get('ambient', [0.75]))
            color = torch.Tensor(config.get('color', [1,1,1]))
            intensity = config.get('intensity', 1.5)
            model.light = DirectionalLightRelight()
            model.light.color = color
            model.light.ambient = ambient
            model.light.light_intensity = intensity

        model.light.floor_point = floor_point
        model.light.floor_normal = floor_normal
    elif forward_type == 'normal':
        render_normal = True
        forward_type = 'render'
        img_path = os.path.join(output_path, 'normal')
        os.makedirs(img_path, exist_ok=True)
    elif forward_type == 'geometry':
        mesh_path = os.path.join(output_path, 'mesh')
        os.makedirs(mesh_path, exist_ok=True)
    else:
        raise NotImplementedError(f'Unknow query {forward_type}')

    model.eval()
    img_cnt = 0

    """
    # This belong elsewhere
    if  config.get('floor_point', None) is not None:
        model.light.floor_point = torch.Tensor(config.get('floor_point', None))
        print('floor point', model.light.floor_point)
    if  config.get('floor_normal', None) is not None:
        model.light.floor_normal = torch.Tensor(config.get('floor_normal', None))
    """
    for i, data in enumerate(tqdm(render_dataloader)):
        data = to_device(data, device='cuda')
        gt_img = None
        if 'img' in data:
            gt_img = data.pop('img').cpu().numpy()[0]

        if forward_type == 'render_relight':
            if mode == 'hdri':
                model.light.hdri_angle = data['angle'].item()
            else:
                model.light.new_direction(data['angle'].item())

        preds = model(data, forward_type=forward_type, render_normal=render_normal, raychunk=raychunk)
        
        if forward_type in ['render', 'normal', 'render_relight']:
            saved_keys = config.get('saved_keys', ['rgb_imgs'])
            for save_key in saved_keys:
                imgs = preds[save_key]
                
                for j, img in enumerate(imgs):
                    write_path = os.path.join(img_path, f'{save_key}_{img_cnt:05d}.png')
                    #if save_key == "depth_imgs":
                        #np.save(os.path.join(img_path, f'{save_key}_{img_cnt:05d}.npy'), img)
                    img = (np.clip(img.cpu().numpy(),0.0,1.0) * 255).astype(np.uint8)
                    if img.shape[2] == 1:
                        img = np.repeat(img, 3, 2)
                    imageio.imwrite(write_path, img)

                    if gt_img is not None and save_gt:
                        assert gt_img_path is not None
                        write_path = os.path.join(gt_img_path, f'{img_cnt:05d}.png')
                        imageio.imwrite(write_path, gt_img)
            
        else:
            densities = preds['density']
            #densities = preds['sigma']

            for j, density in enumerate(densities):
                write_path = os.path.join(mesh_path, f'{img_cnt:05d}.ply')
                res = density.shape[1] - 1
                density = density.cpu().numpy()
                density = np.maximum(density, 0)
                vertices, triangles = mcubes.marching_cubes(
                    density.reshape(res + 1, res + 1, res + 1), 
                    config.get('density_threshold', 5.)
                )
                mesh = trimesh.Trimesh(vertices / res - .5, triangles)
                mesh.export(write_path)
                img_cnt += 1

        img_cnt += 1

@hydra.main(version_base='1.3', config_path=CONFIG_BASE, config_name='h36m_zju.yaml')
def cli(config):
    return render(config)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.multiprocessing.set_start_method('spawn')
    cli()