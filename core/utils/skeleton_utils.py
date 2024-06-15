import io
import cv2
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from collections import deque
from copy import copy, deepcopy
from collections import namedtuple
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
import matplotlib.pyplot as plt
import core.utils.rotation_conversions as p3dr

"""
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds
"""

#################################
#         Skeleton Helpers      #
#################################

Skeleton = namedtuple("Skeleton", ["joint_names", "joint_trees", "root_id", "nonroot_id", "cutoffs", "end_effectors"])

def mat_to_hom_torch(mat):
    """
    To homogeneous coordinates
    """
    last_row = torch.tensor([[0., 0., 0., 1.]]).to(mat.device)

    if mat.dim() == 3:
        last_row = last_row.expand(mat.size(0), 1, 4)
    return torch.cat([mat, last_row], dim=-2)

def rotate_x(phi):
    cos = np.cos(phi)
    sin = np.sin(phi)
    return np.array([[1,   0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0,   0,    0, 1]], dtype=np.float32)

def rotate_z(psi):
    cos = np.cos(psi)
    sin = np.sin(psi)
    return np.array([[cos, -sin, 0, 0],
                     [sin,  cos, 0, 0],
                     [0,      0, 1, 0],
                     [0,      0, 0, 1]], dtype=np.float32)
def rotate_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,   0,  sin, 0],
                     [0,     1,    0, 0],
                     [-sin,  0,  cos, 0],
                     [0,   0,      0, 1]], dtype=np.float32)


def translate(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def arccos_safe(a):
    clipped = np.clip(a, -1.+1e-8, 1.-1e-8)
    return np.arccos(clipped)

def hmm(A, B):
    R_A, T_A = A[..., :3, :3], A[..., :3, -1:]
    R_B, T_B = B[..., :3, :3], B[..., :3, -1:]
    R = R_A @ R_B
    T = R_A @ T_B + T_A
    return torch.cat([R, T], dim=-1)

CanonicalSkeleton = Skeleton(
    joint_names=[
        # 0-4
        'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        # 5-9
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
        # 10-14
        'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
        # 15-16
        'spine', 'head',
    ],
    joint_trees=np.array([
        1, 15, 1, 2, 3,
        1, 5, 6, 14, 8,
        9, 14, 11, 12, 14,
        14, 1]),
    root_id=14,
    nonroot_id=[i for i in range(16) if i != 14],
    cutoffs={},
    end_effectors=None,
)

SMPLSkeleton = Skeleton(
    joint_names=[
        # 0-3
        'pelvis', 'left_hip', 'right_hip', 'spine1',
        # 4-7
        'left_knee', 'right_knee', 'spine2', 'left_ankle',
        # 8-11
        'right_ankle', 'spine3', 'left_foot', 'right_foot',
        # 12-15
        'neck', 'left_collar', 'right_collar', 'head',
        # 16-19
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        # 20-23,
        'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ],
    joint_trees=np.array(
                [0, 0, 0, 0,
                 1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 9, 9, 12,
                 13, 14, 16, 17,
                 18, 19, 20, 21]),
    root_id=0,
    nonroot_id=[i for i in range(24) if i != 0],
    cutoffs={'hip': 200, 'spine': 300, 'knee': 70, 'ankle': 70, 'foot': 40, 'collar': 100,
            'neck': 100, 'head': 120, 'shoulder': 70, 'elbow': 70, 'wrist': 60, 'hand': 60},
    end_effectors=[10, 11, 15, 22, 23],
)

# for backward compatibility
CMUSkeleton = SMPLSkeleton

MixamoSkeleton = Skeleton(
    joint_names=[
        # 0-3
        'mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 
        # 4-7
        'mixamorig:Neck', 'mixamorig:Head', 'mixamorig:LeftUpLeg', 'mixamorig:LeftLeg', 
        # 8-11
        'mixamorig:LeftFoot', 'mixamorig:LeftToeBase', 'mixamorig:RightUpLeg', 'mixamorig:RightLeg', 
        # 12-15
        'mixamorig:RightFoot', 'mixamorig:RightToeBase', 'mixamorig:LeftShoulder', 'mixamorig:LeftArm', 
        # 16-19
        'mixamorig:LeftForeArm', 'mixamorig:LeftHand', 'mixamorig:RightShoulder', 'mixamorig:RightArm', 
        # 20-21
        'mixamorig:RightForeArm', 'mixamorig:RightHand'],
    joint_trees=np.array([
        0,  0,  1,  2,  
        3,  4,  0,  6,  
        7,  8,  0, 10,
        11, 12,  3, 14,
        15, 16,  3, 18,
        19, 20]
    ),
    root_id=0,
    nonroot_id=[i for i in range(22) if i != 0],
    cutoffs={},
    end_effectors=np.array([5, 9, 13, 17, 21]),
)

ViTPoseSkeleton = Skeleton(
    joint_names=['spine0', # 0
                 'spine1', # 1
                 'neck', # 2
                 'head', # 3
                 'thigh_L', # 4
                 'shin_L', # 5
                 'foot_L', # 6
                 'thigh_R', # 7
                 'shin_R', # 8
                 'foot_R', # 9
                 'shoulder_L', # 10
                 'upperarm_L', # 11
                 'forearm_L', # 12 
                 'hand_L', # 13
                 'shoulder_R', # 14
                 'upperarm_R', # 15
                 'forearm_R', # 16
                 'hand_R',], # 17
    joint_trees=np.array([
        0,  0,  1,  2,  # spine - head
        0,  4,  5, # left leg
        0,  7, 8, # right leg
        2, 10, 11, 12, # left arm
        2, 14, 15, 16]  # right arm
    ),
    root_id=0,
    nonroot_id=[i for i in range(18) if i != 0],
    cutoffs={},
    end_effectors=np.array([3, 6, 9, 13, 17]),
)

ViTPoseSkeletonv2 = Skeleton(
    joint_names=['spine0', # 0
                 'spine1', # 1
                 'neck', # 2
                 'head', # 3
                 'hip_L', # 4
                 'thigh_L', # 5
                 'shin_L', # 6
                 'foot_L', # 7
                 'hip_R', # 8
                 'thigh_R', # 9
                 'shin_R', # 10
                 'foot_R', # 11
                 'shoulder_L', # 12
                 'upperarm_L', # 13
                 'forearm_L', # 14
                 'hand_L', # 15
                 'shoulder_R', # 16
                 'upperarm_R', # 17
                 'forearm_R', # 18
                 'hand_R',], # 19
    joint_trees=np.array([
        0,  0,  1,  2,  # spine - head
        0,  4,  5, 6, # left leg
        0,  8, 9, 10, # right leg
        1, 12, 13, 14, # left arm
        1, 16, 17, 18]  # right arm
    ),
    root_id=0,
    nonroot_id=[i for i in range(20) if i != 0],
    cutoffs={},
    end_effectors=np.array([3, 7, 11, 15, 19]),
)

Mpi3dhpSkeleton = Skeleton(
    joint_names=[
    # 0-3
    'spine3', 'spine4', 'spine2', 'spine',
    # 4-7
    'pelvis', 'neck', 'head', 'head_top',
    # 8-11
    'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
    # 12-15
    'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow',
    # 16-19
    'right_wrist', 'right_hand', 'left_hip', 'left_knee',
    # 20-23
    'left_ankle', 'left_foot', 'left_toe', 'right_hip',
    # 24-27
    'right_knee', 'right_ankle', 'right_foot', 'right_toe'
    ],
    joint_trees=np.array([
        2, 0, 3, 4,
        4, 1, 5, 6,
        5, 8, 9, 10,
        11, 5, 13, 14,
        15, 16, 4, 18,
        19, 20, 21, 4,
        23, 24, 25, 26
    ]),
    root_id=4,
    nonroot_id=[i for i in range(27) if i != 4],
    cutoffs={},
    end_effectors=None,
)

HARESkeleton = Skeleton(
        joint_names=[
            'Spine_04', 'Spine_05', 'Neck', 'Head', 'Mouth', 'Ear_1.L',
            'Ear_2.L', 'Eye.L', 'Ear_1.R', 'Ear_2.R', 'Eye.R', 'Hip_f.L',
            'Though_f.L', 'Leg_1_f.L', 'Leg_2_f.L', 'Feet_f.L', 'Claws_f.L',
            'Hip_f.R', 'Though_f.R', 'Leg_1_f.R', 'Leg_2_f.R', 'Feet_f.R',
            'Claws_f.R', 'Spine_03', 'Spine_02', 'Though_b.L', 'Leg_1_b.L',
            'Feet_b.L', 'Claws_b.L', 'Though_b.R', 'Leg_1_b.R', 'Feet_b.R',
            'Claws_b.R', 'Spine_01', 'Tail'
        ], 
       joint_trees=[ 
            0,  0,  1,  2,  3,  3,  5,  3,  3,  8,  3,  1, 11, 12, 13, 14, 15,
            1, 17, 18, 19, 20, 21,  0, 23, 24, 25, 26, 27, 24, 29, 30, 31, 24,
            33
        ], 
        root_id=0, 
        nonroot_id=[i for i in range(35) if i != 0], 
        cutoffs={}, 
        end_effectors=None
    )

WOLFSkeleton = Skeleton(
        joint_names=[
            'Spine_03', 'Spine_02', 'Spine_01', 'tail_01', 'tail_02',
            'tail_03', 'tail_04', 'tail_05', 'hip_b.L', 'thigh_b.L', 'leg_b.L',
            'shin_b.L', 'foot_b.L', 'claws_b.L', 'hip_b.R', 'thigh_b.R',
            'leg_b.R', 'shin_b.R', 'foot_b.R', 'claws_b.R', 'Spine_04',
            'Spine_05', 'neck', 'head', 'mouth', 'nose', 'eye.L', 'Ear.L',
            'eye.R', 'Ear.R', 'hip_f.L', 'thigh_f.L', 'leg_f.L', 'shin_f.L',
            'foot_f.L', 'claws_f.L', 'hip_f.R', 'thigh_f.R', 'leg_f.R',
            'shin_f.R', 'foot_f.R', 'claws_f.R'], 
        joint_trees=[ 
            0,  0,  1,  2,  3,  4,  5,  6,  1,  8,  9, 10, 11, 12,  1, 14, 15,
            16, 17, 18,  0, 20, 21, 22, 23, 23, 23, 23, 23, 23, 21, 30, 31, 32,
            33, 34, 21, 36, 37, 38, 39, 40
        ],
       root_id=0, 
       nonroot_id=[i for i in range(35) if i != 0], 
       cutoffs={}, 
       end_effectors=None
    )

def get_skeleton_type(kps):

    if kps.shape[-2] == 17:
        skel_type = CanonicalSkeleton
    elif kps.shape[-2] == 28:
        skel_type = Mpi3dhpSkeleton
    else:
        skel_type = SMPLSkeleton
    return skel_type

def get_index_mapping_to_canon(skel_def):

    canon_joint_names = CanonicalSkeleton.joint_names
    try:
        idx_map = [skel_def.joint_names.index(j) for j in canon_joint_names]
    except:
        raise ValueError("Skeleton joint does not match the canonical one.")
    return idx_map

def canonicalize(skel, skel_type="3dhp"):

    if skel_type == "3dhp":
        skel_def = Mpi3dhpSkeleton
        idx_map = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

    if idx_map is None:
        idx_map = get_index_mapping_to_canon(skel_def)

    if len(skel.shape) == 3:
        canon_skel = deepcopy(skel[:, idx_map, :])
    elif len(skel.shape) == 2:
        canon_skel = deepcopy(skel[idx_map, :])
    else:
        raise ValueError(f"Skeleton should have either 3 or 2 dimensions, but got {len(skel)} instead.")

    return canon_skel

def coord_to_homogeneous(c):
    assert c.shape[-1] == 3

    if len(c.shape) == 2:
        h = np.ones((c.shape[0], 1)).astype(c.dtype)
        return np.concatenate([c, h], axis=1)
    elif len(c.shape) == 1:
        h = np.array([0, 0, 0, 1]).astype(c.dtype)
        h[:3] = c
        return h
    else:
        raise NotImplementedError(f"Input must be a 2-d or 1-d array, got {len(c.shape)}")

#################################
#       smpl processing         #
#################################

smpl_rest_pose = np.array([[ 0.00000000e+00,  2.30003661e-09, -9.86228770e-08],
                           [ 1.63832515e-01, -2.17391014e-01, -2.89178602e-02],
                           [-1.57855421e-01, -2.14761734e-01, -2.09642015e-02],
                           [-7.04505108e-03,  2.50450850e-01, -4.11837511e-02],
                           [ 2.42021069e-01, -1.08830070e+00, -3.14962119e-02],
                           [-2.47206554e-01, -1.10715497e+00, -3.06970738e-02],
                           [ 3.95125849e-03,  5.94849110e-01, -4.03754264e-02],
                           [ 2.12680623e-01, -1.99382353e+00, -1.29327580e-01],
                           [-2.10857525e-01, -2.01218796e+00, -1.23002514e-01],
                           [ 9.39484313e-03,  7.19204426e-01,  2.06931755e-02],
                           [ 2.63385147e-01, -2.12222481e+00,  1.46775618e-01],
                           [-2.51970559e-01, -2.12153077e+00,  1.60450473e-01],
                           [ 3.83779174e-03,  1.22592449e+00, -9.78838727e-02],
                           [ 1.91201791e-01,  1.00385976e+00, -6.21964522e-02],
                           [-1.77145526e-01,  9.96228695e-01, -7.55542740e-02],
                           [ 1.68482102e-02,  1.38698268e+00,  2.44048554e-02],
                           [ 4.01985168e-01,  1.07928419e+00, -7.47655183e-02],
                           [-3.98825467e-01,  1.07523870e+00, -9.96334553e-02],
                           [ 1.00236952e+00,  1.05217218e+00, -1.35129794e-01],
                           [-9.86728609e-01,  1.04515052e+00, -1.40235111e-01],
                           [ 1.56646240e+00,  1.06961894e+00, -1.37338534e-01],
                           [-1.56946480e+00,  1.05935931e+00, -1.53905824e-01],
                           [ 1.75282109e+00,  1.04682994e+00, -1.68231070e-01],
                           [-1.75758195e+00,  1.04255080e+00, -1.77773550e-01]], dtype=np.float32)

def get_noisy_joints(kp3d, ext_scale, noise_mm):
    noise = np.random.normal(scale=noise_mm * ext_scale, size=kp3d.shape)
    print(f"noise of scale {noise_mm * ext_scale}")
    return kp3d + noise

def get_noisy_bones(bones, noise_degree):
    noise_scale = np.pi / 180. * noise_degree
    mask = (np.random.random(bones.shape) > 0.5).astype(np.float32)
    noise = np.random.normal(0, noise_scale, bones.shape) * mask
    noisy_bones = bones + noise
    print(np.abs(noise).max())
    return noisy_bones

def perturb_poses(bone_poses, kp_3d, ext_scale, noise_degree=0.1,
                  noise_mm=None, dataset_ext_scale=0.25 / 0.00035,
                  noise_pelvis=None,
                  skel_type=SMPLSkeleton):

    noisy_bones = bone_poses if noise_degree is None else get_noisy_bones(bone_poses, noise_degree)
    rest_poses = smpl_rest_pose.copy()[None]
    # scale rest poses accordingly
    rest_poses = rest_poses.repeat(kp_3d.shape[0], 0) * ext_scale
    if noise_mm is not None:
        rest_poses = get_noisy_joints(rest_poses,
                                      ext_scale / dataset_ext_scale,
                                      noise_mm)

    pelvis_loc = kp_3d[:, skel_type.root_id].copy()
    if noise_pelvis is not None:
        noise = np.random.normal(scale=noise_pelvis * ext_scale / dataset_ext_scale, size=pelvis_loc.shape)
        pelvis_loc += noise
    l2ws = np.array([get_smpl_l2ws(bone, pose, scale=1.) for bone, pose in zip(noisy_bones, rest_poses)])
    l2ws[:, :, :3, -1] += pelvis_loc[:, None]

    noisy_skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])
    noisy_kp = l2ws[:, :, :3, -1]

    return noisy_bones, noisy_skts, noisy_kp

def skt_from_smpl(bone_poses, scale, kp_3d=None,
                  pelvis_loc=None, skel_type=SMPLSkeleton):
    l2ws = np.array([get_smpl_l2ws(bone_pose, scale=scale) for bone_pose in bone_poses])

    if kp_3d is not None:
        l2ws[:, :, :3, -1] = kp_3d
    if pelvis_loc is not None:
        l2ws[:, :, :3, -1] += pelvis_loc[:, None]
    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])
    return skts, l2ws

def get_smpl_l2ws(pose, rest_pose=None, scale=1., skel_type=SMPLSkeleton, coord="xxx"):
    # TODO: take root as well

    def mat_to_homo(mat):
        last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.concatenate([mat, last_row], axis=0)

    joint_trees = skel_type.joint_trees
    if rest_pose is None:
        # original bone parameters is in (x,-z,y), while rest_pose is in (x, y, z)
        rest_pose = smpl_rest_pose


    # apply scale
    rest_kp = rest_pose * scale
    mrots = [Rotation.from_rotvec(p).as_matrix()  for p in pose]
    mrots = np.array(mrots)

    l2ws = []
    # TODO: assume root id = 0
    # local-to-world transformation
    l2ws.append(mat_to_homo(np.concatenate([mrots[0], rest_kp[0, :, None]], axis=-1)))
    mrots = mrots[1:]
    for i in range(rest_kp.shape[0] - 1):
        idx = i + 1
        # rotation relative to parent
        joint_rot = mrots[idx-1]
        joint_j = rest_kp[idx][:, None]

        parent = joint_trees[idx]
        parent_j = rest_kp[parent][:, None]

        # transfer from local to parent coord
        joint_rel_transform = mat_to_homo(
            np.concatenate([joint_rot, joint_j - parent_j], axis=-1)
        )

        # calculate kinematic chain by applying parent transform (to global)
        l2ws.append(l2ws[parent] @ joint_rel_transform)

    l2ws = np.array(l2ws)

    return l2ws

def get_rest_pose_from_l2ws(l2ws, skel_type=SMPLSkeleton):
    """
    extract rest pose from local to world matrices for each joint
    """

    joint_trees = skel_type.joint_trees

    rest_pose = [l2ws[skel_type.root_id, :3, -1]]
    kp = l2ws[:, :3, -1]
    for i in range(len(l2ws[1:])):
        idx = i + 1
        parent = joint_trees[idx]
        # rotate from world to the parent rest pose!
        joint_rel_pos =  l2ws[parent, :3, :3].T @ (kp[idx] - kp[parent])
        joint_pos = rest_pose[parent] + joint_rel_pos
        rest_pose.append(joint_pos)

    return np.array(rest_pose)

def bones_to_rot(bones):
    if bones.shape[-1] == 3:
        return axisang_to_rot(bones)
    elif bones.shape[-1] == 6:
        return rot6d_to_rotmat(bones)
    else:
        raise NotImplementedError

def rot_to_axisang(rot):
    return p3dr.matrix_to_axis_angle(rot)

def rot_to_rot6d(rot):
    return rot[..., :3, :2].flatten(start_dim=-2)

def axisang_to_rot(axisang):
    return p3dr.axis_angle_to_matrix(axisang)

def axisang_to_quat(axisang):
    return p3dr.axis_angle_to_quaternion(axisang)

def axisang_to_rot6d(axisang):
    return rot_to_rot6d(axisang_to_rot(axisang))

def rot6d_to_axisang(rot6d):
    return rot_to_axisang(rot6d_to_rotmat(rot6d))

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (*,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x_shape = x.shape[:-1]
    x = x.reshape(-1,3,2)

    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1).reshape(*x_shape, 3, 3)

#################################
#       Coordinate System       #
#################################

def nerf_c2w_to_extrinsic(c2w):
    return np.linalg.inv(swap_mat(c2w))

def nerf_c2w_to_extrinsic_th(c2w):
    return torch.inverse(swap_mat_th(c2w))

## LEGACY BELOW ##
def nerf_bones_to_smpl(bones):
    # undo local transformation
    bones = torch.cat([bones[..., 0:1], -bones[..., 2:3], bones[..., 1:2]], dim=-1)
    rots = axisang_to_rot(bones.view(-1, 3)).view(*bones.shape[:2], 3, 3)
    # undo global transformation
    root_rot = torch.tensor([[1., 0., 0.],
                             [0., 0.,-1.],
                             [0., 1., 0.]]).to(bones.device)
    root_rot = root_rot.expand(len(rots), 3, 3)
    rots[:, 0] = root_rot @ rots[:, 0]
    return rots

def smpl_pts_to_surreal(pts):
    return np.concatenate([
        pts[..., 0:1],
        pts[..., 2:3],
       -pts[..., 1:2],
    ], axis=-1)

def smpl_rot_to_surreal(rot, np=False):
    R = [[1, 0, 0],
         [0, 0, 1],
         [0,-1, 0]]
    R = torch.FloatTensor(R).to(rot.device) if not np else np.array(R, dtype=np.float32)
    if len(rot.shape) == 3:
        return R[None] @ rot
    return R @ rot
## LEGACY ABOVE ##

def skeleton3d_to_2d(kps, c2ws, H, W, focals, centers=None):

    exts = np.array([nerf_c2w_to_extrinsic(c2w) for c2w in c2ws])

    kp2ds = []
    for i, (kp, ext) in enumerate(zip(kps, exts)):
        f = focals[i] if not isinstance(focals, float) else focals
        h = H if isinstance(H, int) else H[i]
        w = W if isinstance(W, int) else W[i]
        center = centers[i] if centers is not None else None
        kp2d = world_to_cam(kp, ext, h, w, f, center)
        kp2ds.append(kp2d)

    return np.array(kp2ds)

#################################
#       Keypoint processing     #
#################################
def get_axis_aligned_rotation(vec):
    '''
    Find a rotation that, when applying the vec (as rot @ vec),
    the rotated vector will align with z-axis.
    '''
    # TODO: codes are mostly redundant with create_local_coord
    # first, find rotation around y-axis
    vec_xz = vec[[0, 2]] / (np.linalg.norm(vec[[0, 2]]) + 1e-8)
    # note: we care only about the projection on z-axis
    # note: reference point is on z, so if x is negative, then we are in 3 or 4th quadrant
    theta = arccos_safe(vec_xz[-1]) * np.sign(vec_xz[0])
    rot_y = rotate_y(theta).T
    rotated_y = rot_y[:3, :3] @ vec

    # then, find rotation around x-axis
    vec_yz = rotated_y[1:3] / (np.linalg.norm(rotated_y[1:3]) + 1e-8)
    # similarly, make sign correct
    psi = arccos_safe(vec_yz[-1]) * np.sign(vec_yz[0])
    rot_x = rotate_x(psi)
    rotated_x = rot_x[:3, :3] @ rotated_y
    # rotate the coordinate system to so that z_axis == vec
    rot = np.linalg.inv(rot_x @ rot_y)
    return rot.T

def get_kp_bounding_cylinder(kp, skel_type=None, ext_scale=0.001,
                             extend_mm=250, top_expand_ratio=1.,
                             bot_expand_ratio=0.25, head=None, verbose=False):
    '''
    head: -y for most dataset (SPIN estimated), z for SURREAL
    '''

    # g_axes: axes that define the ground plane
    # h_axis: axis that is perpendicular to the ground
    # flip: to flip the height (if the sky is on the negative part)
    assert head is not None, 'need to specify the direction of ground plane (i.e., the direction when the person stand up straight)!'
    if verbose:
        print(f'Head direction: {head}')
    if head.endswith('z'):
        g_axes = [0, 1]
        h_axis = 2
    elif head.endswith('y'):
        g_axes = [0, 2]
        h_axis = 1
    else:
        raise NotImplementedError(f'Head orientation {head} is not implemented!')
    flip = 1 if not head.startswith('-') else -1

    if skel_type is None:
        skel_type = get_skeleton_type(kp)

    n_dim = len(kp.shape)
    # find root location
    root_id = skel_type.root_id
    if not isinstance(root_id, int):
        root_id = root_id[0] # use the first root
    root_loc = kp[..., root_id, :]

    # calculate distance to center line
    if n_dim == 2:
        dist = np.linalg.norm(kp[:, g_axes] - root_loc[g_axes], axis=-1)
    elif n_dim == 3: # batch
        dist = np.linalg.norm(kp[..., g_axes] - root_loc[:, None, g_axes], axis=-1)
        max_height = (flip * kp[..., h_axis]).max()
        min_height = (flip * kp[..., h_axis]).min()

    # find the maximum distance to center line (in mm*ext_scale)
    max_dist = dist.max(-1)
    max_height = (flip * kp[..., h_axis]).max(-1)
    min_height = (flip * kp[..., h_axis]).min(-1)

    # set the radius of cylinder to be a bit larger
    # so that every part of the human is covered
    extension = extend_mm * ext_scale
    radius = max_dist + extension
    top = flip * (max_height + extension * top_expand_ratio) # extend head a bit more
    bot = flip * (min_height - extension * bot_expand_ratio) # don't need that much for foot
    cylinder_params = np.stack([root_loc[..., g_axes[0]], root_loc[..., g_axes[1]],
                               radius, top, bot], axis=-1)
    return cylinder_params

def calculate_angle(a, b=None):
    if b is None:
        b = torch.Tensor([0., 0., 1.]).view(1, 1, -1)
    dot_product = (a * b).sum(-1)
    norm_a = torch.norm(a, p=2, dim=-1)
    norm_b = torch.norm(b, p=2, dim=-1)
    cos = dot_product / (norm_a * norm_b)
    cos = torch.clamp(cos, -1. + 1e-6, 1. - 1e-6)
    angle = torch.acos(cos)
    assert not torch.isnan(angle).any()

    return angle - 0.5 * np.pi

def cylinder_to_box_2d(cylinder_params, hwf, w2c=None, scale=1.0,
                       center=None, make_int=True):

    H, W, focal = hwf

    root_loc, radius = cylinder_params[..., :2], cylinder_params[..., 2:3]
    top, bot = cylinder_params[..., 3:4], cylinder_params[..., 4:5]

    rads = np.linspace(0., 2 * np.pi, 50)

    if len(root_loc.shape) == 1:
        root_loc = root_loc[None]
        radius = radius[None]
        top = top[None]
        bot = bot[None]
    N = root_loc.shape[0]

    x = root_loc[..., 0:1] + np.cos(rads)[None] * radius
    z = root_loc[..., 1:2] + np.sin(rads)[None] * radius

    y_top = top * np.ones_like(x)
    y_bot = bot * np.ones_like(x)
    w = np.ones_like(x) # to make homogenous coord

    top_cap = np.stack([x, y_top, z, w], axis=-1)
    bot_cap = np.stack([x, y_bot, z, w], axis=-1)

    cap_pts = np.concatenate([top_cap, bot_cap], axis=-2)
    cap_pts = cap_pts.reshape(-1, 4)

    intrinsic = focal_to_intrinsic_np(focal)

    if w2c is not None:
        cap_pts = cap_pts @ w2c.T
    cap_pts = cap_pts @ intrinsic.T
    cap_pts = cap_pts.reshape(N, -1, 3)
    pts_2d = cap_pts[..., :2] / cap_pts[..., 2:3]

    max_x = pts_2d[..., 0].max(-1)
    min_x = pts_2d[..., 0].min(-1)
    max_y = pts_2d[..., 1].max(-1)
    min_y = pts_2d[..., 1].min(-1)

    if make_int:
        max_x = np.ceil(max_x).astype(np.int32)
        min_x = np.floor(min_x).astype(np.int32)
        max_y = np.ceil(max_y).astype(np.int32)
        min_y = np.floor(min_y).astype(np.int32)

    tl = np.stack([min_x, min_y], axis=-1)
    br = np.stack([max_x, max_y], axis=-1)

    if center is None:
        offset_x = int(W * .5)
        offset_y = int(H * .5)
    else:
        offset_x, offset_y = int(center[0]), int(center[1])


    tl[:, 0] += offset_x
    tl[:, 1] += offset_y

    br[:, 0] += offset_x
    br[:, 1] += offset_y

    # scale the box
    if scale != 1.0:
        box_width = (max_x - min_x) * 0.5 * scale
        box_height = (max_y - min_y) * 0.5 * scale
        center_x = (br[:, 0] + tl[:, 0]).copy() * 0.5
        center_y = (br[:, 1] + tl[:, 1]).copy() * 0.5

        tl[:, 0] = center_x - box_width
        br[:, 0] = center_x + box_width
        tl[:, 1] = center_y - box_height
        br[:, 1] = center_y + box_height

    tl[:, 0] = np.clip(tl[:, 0], 0, W-1)
    br[:, 0] = np.clip(br[:, 0], 0, W-1)
    tl[:, 1] = np.clip(tl[:, 1], 0, H-1)
    br[:, 1] = np.clip(br[:, 1], 0, H-1)

    if N == 1:
        tl = tl[0]
        br = br[0]
        pts_2d = pts_2d[0]

    return tl, br, pts_2d

def get_skeleton_transformation(kps, skel_type=None, align_z=True):


    if skel_type is None:
        skel_type = get_skeleton_type(kps)

    skt_builder = HierarchicalTransformation(skel_type.joint_trees, skel_type.joint_names,
                                             align_z=align_z)
    skts = []
    transformed_kps = []

    for kp in kps:
        skt_builder.build_transformation(kp)
        skts.append(np.array(skt_builder.get_all_w2ls()))
        transformed_kps.append(skt_builder.transform_w2ls(kp))
    skts = np.stack(skts)
    transformed_kps = np.array(transformed_kps)[..., :3]
    return skts, transformed_kps

def extend_cmu_skeleton(kps):
    original_joints = SMPLSkeleton.joint_names
    extended_joints = SMPLSkeletonExtended.joint_names

    mapping = [extended_joints.index(name) for name in original_joints]
    name_to_idx = {name: i for i, name in enumerate(extended_joints)}

    extended = np.zeros((kps.shape[0], 28, 3)).astype(np.float32)
    extended[:, mapping] = kps

    # get the indices we want to set
    left_upper = name_to_idx["left_upper_arm"]
    left_lower = name_to_idx["left_lower_arm"]
    right_upper = name_to_idx["right_upper_arm"]
    right_lower = name_to_idx["right_lower_arm"]

    # get the indices for calculation
    left_shoulder = name_to_idx["left_shoulder"]
    left_elbow = name_to_idx["left_elbow"]

    right_shoulder = name_to_idx["right_shoulder"]
    right_elbow = name_to_idx["right_elbow"]

    left_wrist = name_to_idx["left_wrist"]
    right_wrist = name_to_idx["right_wrist"]

    extended[:, left_upper] = (extended[:, left_shoulder] + extended[:, left_elbow]) * 0.5
    extended[:, right_upper] = (extended[:, right_shoulder] + extended[:, right_elbow]) * 0.5

    extended[:, left_lower] = (extended[:, left_wrist] + extended[:, left_elbow]) * 0.5
    extended[:, right_lower] = (extended[:, right_wrist] + extended[:, right_elbow]) * 0.5

    return extended

def get_geodesic_dists(kp, in_hops=False, skel_type=SMPLSkeleton):
    """
    Run Floyd-Warshall algo to get geodesic distances
    """

    N_J = kp.shape[-2]
    dists = np.ones((N_J, N_J)) * 100000.
    joint_trees = skel_type.joint_trees

    for i, parent in enumerate(joint_trees):
        if not in_hops:
            dist = ((kp[i] - kp[parent])**2.0).sum()**0.5
        else:
            dist = 1.
        dists[i, parent] = dist
        dists[parent, i] = dist
        dists[i, i] = 0.

    for k in range(N_J):
        for i in range(N_J):
            for j in range(N_J):
                new_dist = dists[i][k] + dists[k][j]
                if dists[i][j] > new_dist:
                    dists[i][j] = new_dist

    return dists


def create_kp_masks(masks, kp2ds=None, kp3ds=None, c2ws=None, hwf=None,
                   extend_iter=3, skel_type=SMPLSkeleton):
    '''
    Create extra foreground region for the (estimated) keypoints.
    ----
    masks: (N, H, W, 1) original masks
    kp2ds: (N, NJ, 2) batch of 2d keypoints
    kp3ds: (N, NJ, 3) batch of 3d keypoints
    c2ws: (N, 4, 4) batch of camera-to-world matrices
    hwf: height, width and focal(s)
    extend_iter: number of times to expand the keypoint masks. Each iter create around 2 pixels.
    '''

    if kp2ds is None:
        assert kp3ds is not None
        H, W, focal = hwf
        kp2ds = skeleton3d_to_2d(kp3d, c2ws, H, W, focal)

    H, W, C = masks[0].shape

    kp_masks = []

    for i, kp in enumerate(kp2ds):
        kp_mask = create_kp_mask(masks[i], kp, H, W, extend_iter, skel_type)
        kp_masks.append(kp_mask)

    return kp_masks

def create_kp_mask(mask, kp, H, W, extend_iter=3, skel_type=SMPLSkeleton):

    # draw skeleton on blank image
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    drawn = draw_skeleton2d(blank, kp, skel_type)
    drawn = drawn.mean(axis=-1)
    drawn[drawn > 0] = 1

    # extend the skeleton a bit more so it covers some space
    # and merge it with mask
    d_kernel = np.ones((5, 5))
    dilated = cv2.dilate(drawn, kernel=d_kernel, iterations=extend_iter)[..., None]
    kp_mask = np.logical_or(mask, dilated).astype(np.uint8)

    return kp_mask

#################################
#       Plotting Helpers        #
#################################


# TODO: move plotting to visualization

def create_plane(x=None, y=None, z=None, length_x=10., length_y=5,
                 length_z=10, n_sample=50):

    x_r = length_x / 2.
    y_r = length_y / 2.
    z_r = length_z / 2.
    if z is not None:
        plane ='xy'
        x, y = np.meshgrid(np.linspace(-x_r, x_r, n_sample), np.linspace(-y_r, y_r, n_sample))
        plane = np.stack([x, y, np.ones_like(x) * z], axis=-1)
    elif y is not None:
        plane = 'xz'
        x, z = np.meshgrid(np.linspace(-x_r, x_r, n_sample), np.linspace(-z_r, z_r, n_sample))
        plane = np.stack([x, np.ones_like(x) * y, z], axis=-1)
    else:
        plane = 'yz'
        y, z = np.meshgrid(np.linspace(-y_r, y_r, n_sample), np.linspace(-z_r, z_r, n_sample))
        plane = np.stack([np.ones_like(y) * x, y, z], axis=-1)

    return plane.astype(np.float32)

def dist_to_joints(kp3d, pts):
    return np.linalg.norm(pts[:, None, :] - kp3d[None], axis=-1)

def plot_bounding_cylinder(kp, cylinder_params=None, fig=None, g_axes=[0, -1], **kwargs):
    '''
    g_axes: ground-plane axes, by default is x-z
    '''
    if cylinder_params is None:
        cylinder_params = get_kp_bounding_cylinder(kp, **kwargs)

    root_loc, radius = cylinder_params[:2], cylinder_params[2]
    top, bot = cylinder_params[3], cylinder_params[4]

    # create circle
    rads = np.linspace(0., 2 * np.pi, 50)
    x = root_loc[0] + np.cos(rads) * radius
    z = root_loc[1] + np.sin(rads) * radius

    y_top = top * np.ones_like(x)
    y_bot = bot * np.ones_like(x)

    #cap_x = x.tolist() + [None] + x.tolist()
    #cap_y = y_top.tolist() + [None] + y_bot.tolist()
    #cap_z = z.tolist() + [None] + z.tolist()

    top_x = x.tolist() + [None]
    top_y = y_top.tolist() + [None]
    top_z = z.tolist() + [None]

    bot_x = x.tolist() + [None]
    bot_y = y_bot.tolist() + [None]
    bot_z = z.tolist() + [None]

    top_cap = go.Scatter3d(x=top_x, y=top_y, z=top_z, mode="lines",
                        line=dict(color="red", width=2))
    bot_cap = go.Scatter3d(x=bot_x, y=bot_y, z=bot_z, mode="lines",
                        line=dict(color="blue", width=2))
    data = [top_cap, bot_cap]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    return fig

def get_head_tail_bone_align_transforms(rest_pose, rest_heads, skel_type):
    assert skel_type in [HARESkeleton, WOLFSkeleton, MixamoSkeleton, ViTPoseSkeleton, ViTPoseSkeletonv2]
    """
    For Blender-based skeletons
    """

    joint_trees = skel_type.joint_trees
    joint_names = skel_type.joint_names

    # initialize bone-align transformation
    children = [[] for j in joint_trees]

    # search for children
    for i, parent in enumerate(joint_trees):
        children[parent].append(i)

    transforms = torch.eye(4).reshape(1, 4, 4).repeat(len(joint_trees), 1, 1)
    for child_idx in range(len(joint_names)):
        parent_idx = joint_trees[child_idx]
        if child_idx == parent_idx:
            continue

        dir_vec = rest_heads[child_idx] - rest_pose[child_idx]
        rot = get_axis_aligned_rotation(dir_vec)
        #rot = np.eye(4)
        #trans = -0.5 * np.linalg.norm(dir_vec, axis=-1)[..., None] * np.array([[0., 1., 0.]], dtype=np.float32)
        trans = -0.5 * np.linalg.norm(dir_vec, axis=-1)[..., None] * np.array([[0., 0., 1.]], dtype=np.float32)
        
        """ 
        if len(children[parent_idx])>1:
            rot = np.eye(4).astype(rot.dtype)
            trans = trans * 0.0
        """
        rot[:3, -1] = trans
        transforms[child_idx] = torch.tensor(rot.copy())
    return transforms, []

def get_bone_align_transforms(rest_pose, skel_type, rest_heads=None):
    if skel_type in [HARESkeleton, WOLFSkeleton, MixamoSkeleton, ViTPoseSkeleton, ViTPoseSkeletonv2]:
        assert rest_heads is not None, f'Skeleton type {skel_type} requires rest_heads'
        return get_head_tail_bone_align_transforms(rest_pose, rest_heads=rest_heads, skel_type=skel_type)
    
    assert skel_type in [SMPLSkeleton]

    joint_trees = skel_type.joint_trees
    joint_names = skel_type.joint_names
    skel_profile = get_skel_profile_from_rest_pose(rest_pose, skel_type)
    no_trans_names = skel_profile['no_trans_in_align_names']

    # initialize bone-align transformation
    children = [[] for j in joint_trees]

    # search for children
    for i, parent in enumerate(joint_trees):
        children[parent].append(i)
    
    transforms = torch.eye(4).reshape(1, 4, 4).repeat(len(joint_trees), 1, 1)
    child_idxs = []
    dir_vecs = []
    for parent_idx, c in enumerate(children):
        # has no child or has multiple child:
        # no needs to align
        if len(c) < 1 or len(c) > 1:
            child_idxs.append(parent_idx)
            continue
        child_idx = c[0]
        # from parent to child
        dir_vec = rest_pose[child_idx] - rest_pose[parent_idx]
        rot = get_axis_aligned_rotation(dir_vec)

        # translation to center of the bone (shift it along z-axis)
        trans = -0.5 * np.linalg.norm(dir_vec, axis=-1)[..., None] * np.array([[0., 0., 1.]], dtype=np.float32)

        if (not any([n in joint_names[child_idx] for n in no_trans_names])) or skel_type == SMPLSkeleton:
            rot[:3, -1] = trans

        transforms[parent_idx] = torch.tensor(rot.copy())
        child_idxs.append(child_idx)
        dir_vecs.append(dir_vec)
    dir_vecs = np.stack(dir_vecs)
    
    return transforms, child_idxs

def create_axis_boxes(axis_scale):
    # each part has 8 coords
    coords = np.zeros((axis_scale.shape[0], 8, 3))

    for i, scale in enumerate(axis_scale):
        x, y, z = scale.copy()
        coords[i, 0, :] = ( x, y, z)
        coords[i, 1, :] = (-x, y, z)
        coords[i, 2, :] = ( x,-y, z)
        coords[i, 3, :] = ( x, y,-z)
        coords[i, 4, :] = (-x,-y, z)
        coords[i, 5, :] = ( x,-y,-z)
        coords[i, 6, :] = (-x, y,-z)
        coords[i, 7, :] = (-x,-y,-z)
    return coords.copy()

def boxes_to_pose(l2ws, corners, rest_pose, skel_type=SMPLSkeleton, rest_heads=None):
    align_T = get_bone_align_transforms(rest_pose, skel_type=skel_type, rest_heads=rest_heads)[0].inverse().cpu().numpy()
    ones = np.ones((*corners.shape[:2], 1))
    corners_h = np.concatenate([corners, ones], axis=-1)
    corners_w = l2ws[:, None ]@ align_T[:, None] @ corners_h[..., None]
    return corners_w[..., :3, 0]


def swap_mat(mat):
    # NOTE: this is OpenCV <-> OpenGL
    # [right, -up, -forward]
    # equivalent to right multiply by:
    # [1, 0, 0, 0]
    # [0,-1, 0, 0]
    # [0, 0,-1, 0]
    # [0, 0, 0, 1]
    return np.concatenate([
        mat[..., 0:1], -mat[..., 1:2], -mat[..., 2:3], mat[..., 3:]
        ], axis=-1)

def swap_mat_th(mat):
    # [right, -up, -forward]
    # equivalent to right multiply by:
    # [1, 0, 0, 0]
    # [0,-1, 0, 0]
    # [0, 0,-1, 0]
    # [0, 0, 0, 1]
    return torch.cat([
        mat[..., 0:1], -mat[..., 1:2], -mat[..., 2:3], mat[..., 3:]
        ], dim=-1)

def focal_to_intrinsic_np(focal):
    if isinstance(focal, float) or (len(focal.reshape(-1)) < 2):
        focal_x = focal_y = focal
    else:
        focal_x, focal_y = focal
    return np.array([[focal_x,      0, 0, 0],
                     [     0, focal_y, 0, 0],
                     [     0,       0, 1, 0]],
                    dtype=np.float32)

def build_intrinsic(center_x, center_y, focal):
    if isinstance(focal, float) or (len(focal.reshape(-1)) < 2):
        focal_x = focal_y = focal
    else:
        focal_x, focal_y = focal
    return np.array([[focal_x,      0, center_x, 0],
                     [     0, focal_y, center_y, 0],
                     [     0,       0,        1, 0]],
                    dtype=np.float32)

def world_to_cam(pts, extrinsic, H, W, focal, center=None):

    if center is None:
        offset_x = W * .5
        offset_y = H * .5
    else:
        offset_x, offset_y = center

    if pts.shape[-1] < 4:
        pts = coord_to_homogeneous(pts)

    intrinsic = focal_to_intrinsic_np(focal)

    cam_pts = pts @ extrinsic.T @ intrinsic.T
    cam_pts = cam_pts[..., :2] / (cam_pts[..., 2:3] + 1e-8)
    cam_pts[cam_pts == np.inf] = 0.
    cam_pts[..., 0] += offset_x
    cam_pts[..., 1] += offset_y
    return cam_pts

def world_to_cam_K(pts, extrinsic, intrinsic):
    if pts.shape[-1] < 4:
        pts = coord_to_homogeneous(pts)
    print(pts.shape)
    print(extrinsic.shape)
    print(intrinsic.shape)
    cam_pts = pts @ extrinsic.T @ intrinsic.T
    cam_pts = cam_pts[..., :2] / (cam_pts[..., 2:3] + 1e-8)
    cam_pts[cam_pts == np.inf] = 0.
    return cam_pts

def calculate_bone_length(kp, skel_type=SMPLSkeleton, to_child=False):
    assert skel_type.root_id == 0
    joint_trees = skel_type.joint_trees
    bone_lens = []

    if not to_child:
        for i in range(1, kp.shape[0]):
            parent = joint_trees[i]
            bone_lens.append((((kp[i] - kp[parent])**2).sum()**0.5))
    else:
        # figure out the children
        children = get_children_joints(skel_type=skel_type)
        bone_lens = []

        for i, c in enumerate(children):
            
            if len(c) < 1:
                # if they have no child
                bone_lens.append(-1.)
                continue
            if i == 0:
                child_idxs = c[1:]
            else:
                child_idxs = c
            lens = (((kp[i:i+1] - kp[child_idxs])**2).sum(-1)**0.5).mean()

            child_idx = c[0]
            bone_lens.append(lens)
            

    return np.array(bone_lens)

def get_children_joints(skel_type):

    joint_trees = skel_type.joint_trees
    
    children = [[] for j in joint_trees]
    for joint, parent in enumerate(joint_trees):
        children[parent].append(joint)
    return children

def get_skel_profile_from_rest_pose(rest_pose, skel_type):
    if len(rest_pose.shape) == 2:
        # expand to (1, N_joints, 3)
        rest_pose = rest_pose[None]

    joint_names = skel_type.joint_names
    if skel_type == SMPLSkeleton:
        width_names = ['shoulder', 'hip', 'collar', 'knee']
    elif skel_type == HARESkeleton:
        width_names = ['hip_f', 'though_f', 'though_b', 'eye', 'ear_1']
    elif skel_type == MixamoSkeleton:
        width_names = ['upleg', 'shoulder']
    elif skel_type == ViTPoseSkeleton or skel_type == ViTPoseSkeletonv2:
        width_names = ['thigh', 'shoulder']
    elif skel_type == WOLFSkeleton:
        width_names = ['hip_f', 'thigh_f', 'thigh_b', 'eye', 'ear']
    else:
        raise NotImplementedError
    

    part_widths = {}
    for width_name in width_names:
        parts = [i for i, n in enumerate(joint_names) if width_name in n.lower()] 
        part_widths[f'{width_name}_width'] = np.linalg.norm(rest_pose[:, parts[0], :] - rest_pose[:, parts[1], :], axis=-1) 

    bone_lens, bone_lens_to_child = [], []

    for r in rest_pose:
        bone_lens.append(calculate_bone_length(r, skel_type=skel_type))
        bone_lens_to_child.append(calculate_bone_length(r, skel_type=skel_type, to_child=True))

    # root has no parent, concatenate 0
    bone_lens = np.concatenate([np.zeros((len(rest_pose), 1)), np.array(bone_lens)], axis=-1)
    bone_lens_to_child = np.array(bone_lens_to_child)

    # set different body parts
    if skel_type == SMPLSkeleton:
        head = ['head', 'neck']
        #torso = ['shoulder', 'spine', 'collar', 'neck', 'pelvis']
        #torso = ['spine', 'collar', 'neck', 'pelvis']
        torso = ['spine', 'pelvis']
        collars = ['collar']
        arms = ['shoulder', 'elbow', 'wrist', 'hand']
        legs = ['hip', 'knee', 'ankle', 'foot']
        tail = None
        ear = None
    elif skel_type == MixamoSkeleton:
        head = ['head', 'neck']
        torso = ['spine', 'hips']
        arms = ['arm', 'shoulder', 'hand']
        legs = ['leg', 'foot', 'toe']
        collars = None
        tail = None
        ear = None
    elif skel_type == ViTPoseSkeleton:
        head = ['neck', 'head']
        torso = ['spine0', 'spine1']
        arms = ['shoulder', 'upperarm', 'forearm', 'hand']
        legs = ['thigh', 'shin', 'foot']
        collars = None
        tail = None
        ear = None
    elif skel_type == ViTPoseSkeletonv2:
        head = ['neck', 'head']
        torso = ['spine0', 'spine1']
        arms = ['shoulder', 'upperarm', 'forearm', 'hand']
        legs = ['hip', 'thigh', 'shin', 'foot']
        collars = None
        tail = None
        ear = None
    else:
        head = ['mouth', 'eye', 'head', 'ear', 'neck', 'nose']
        torso = ['spine_02', 'spine_03', 'spine_04', 'spine_05']
        tail = ['spine_01', 'tail']
        ear = ['ear']
        arms = ['_f']
        legs = ['_b']
        collars = None

    # find different parts
    torso_idxs, arm_idxs, leg_idxs, head_idxs, collar_idxs = [], [], [], [], []
    ear_idxs, tail_idxs = [], []
    for i, name in enumerate(joint_names):
        if any([n.lower() in name.lower() for n in torso]):
            torso_idxs.append(i)
        if any([n.lower() in name.lower() for n in arms]):
            arm_idxs.append(i)
        if any([n.lower() in name.lower() for n in legs]):
            leg_idxs.append(i)
        if any([n.lower() in name.lower() for n in head]):
            head_idxs.append(i)
        if collars is not None and any([n.lower() in name.lower() for n in collars]):
            collar_idxs.append(i)
        if tail is not None and any([n.lower() in name.lower() for n in tail]):
            tail_idxs.append(i)
        if ear is not None and any([n.lower() in name.lower() for n in ear]):
            ear_idxs.append(i)

    torso_idxs = np.array(torso_idxs)
    arm_idxs = np.array(arm_idxs)
    leg_idxs = np.array(leg_idxs)
    head_idxs = np.array(head_idxs)
    collar_idxs = np.array(collar_idxs)
    tail_idxs = np.array(tail_idxs)
    ear_idxs = np.array(ear_idxs)

    # find end-effectors and joints with multiple children
    children = [[] for i in range(len(skel_type.joint_trees))]
    for i, j in enumerate(skel_type.joint_trees):
        children[j].append(i)

    child_idxs = np.zeros((len(children),))
    end_effectors, multiple_children = [], []
    for i, c in enumerate(children):
        if len(c) == 0:
            end_effectors.append(i)
            continue
        if len(c) > 1:
            multiple_children.append(i)
        elif len(c) == 1:
            child_idxs[i] = c[0]
    end_effectors = np.array(end_effectors)
    multiple_children = np.array(multiple_children)

    # TODO: hard-coded for now
    # 10, 11: feet (end effector)
    # 22, 23: hands (end effector)
    #  6, 12: not end effectors, but probably can be remove because of overlapping
    no_trans_in_align_names = []
    if skel_type == SMPLSkeleton:
        rigid_ignore = np.array([10, 11, 12, 22, 23])
    elif skel_type == HARESkeleton:
        #ignore_name = ['claw', 'tail', 'eye', 'mouth']
        ignore_name = []
        rigid_ignore = []
        for i, name in enumerate(joint_names):
            if any([n.lower() in name.lower() for n in ignore_name]):
                rigid_ignore.append(i)
        rigid_ignore = np.array(rigid_ignore)
    elif skel_type == WOLFSkeleton:
        #ignore_name = ['claw', 'tail_05', 'eye', 'mouth', 'nose']
        ignore_name = []
        rigid_ignore = []
        for i, name in enumerate(joint_names):
            if any([n.lower() in name.lower() for n in ignore_name]):
                rigid_ignore.append(i)
        rigid_ignore = np.array(rigid_ignore)
    elif skel_type == MixamoSkeleton:
        ignore_name = []
        rigid_ignore = []
    elif skel_type == ViTPoseSkeleton or skel_type == ViTPoseSkeletonv2:
        ignore_name = []
        rigid_ignore = []
    else:
        raise NotImplementedError
    rigid_cluster = np.array([i for i in range(len(skel_type.joint_names)) if i not in rigid_ignore])

    return {'bone_lens': bone_lens,
            'bone_lens_to_child': bone_lens_to_child,
            'torso_idxs': torso_idxs,
            'arm_idxs': arm_idxs,
            'leg_idxs': leg_idxs,
            'head_idxs': head_idxs,
            'collar_idxs': collar_idxs,
            'tail_idxs': tail_idxs,
            'ear_idxs': ear_idxs,
            'end_effectors': end_effectors,
            'multiple_children': multiple_children,
            'children': child_idxs,
            'rigid_idxs': rigid_cluster,
            'rigid_ignore': rigid_ignore,
            'full_children': children,
            'no_trans_in_align_names': no_trans_in_align_names,
            'rest_pose': rest_pose,
            'skel_type': skel_type,
            **part_widths}

def find_n_hops_joint_neighbors(skel_profile, n_hops=2):

    skel_type = skel_profile['skel_type']
    joint_trees = skel_type.joint_trees
    full_children = deepcopy(skel_profile['full_children'])

    rigid_idxs = skel_profile['rigid_idxs']

    nb_joints = []
    for i, joint_idx in enumerate(rigid_idxs):
        joint_idx = int(joint_idx)
        children = list(deepcopy(full_children[joint_idx]))
        parent = joint_trees[joint_idx]

        nb_joints_ = []
        queue = [joint_idx] + children + [parent] 
        for h in range(n_hops):
            queue_ = []
            for q in queue:
                q = int(q)
                if q in nb_joints_:
                    continue
                nb_joints_.append(q)
                
                # find children and apraents
                q_children = deepcopy(full_children[q])
                for child in q_children:
                    if child not in queue and child not in nb_joints_:
                        queue_.append(child)

                q_parent = joint_trees[q]
                if q_parent not in queue and q_parent not in nb_joints_:
                    queue_.append(q_parent)
            queue = deepcopy(queue_)

        # special case for head when we collapse the neck and head together
        if n_hops >= 2 and joint_idx == 15 and skel_type == SMPLSkeleton:
                nb_joints_.append(skel_type.joint_names.index('left_collar'))
                nb_joints_.append(skel_type.joint_names.index('right_collar'))


        nb_joints.append(np.unique(nb_joints_))
    nb_joints = deepcopy(nb_joints)

    hop_mask = torch.zeros(len(rigid_idxs), len(joint_trees))

    for i, r_joints in enumerate(nb_joints):
        hop_mask[i, r_joints] = 1.
    
    hop_mask = hop_mask[..., rigid_idxs]
    """ 
    for i, joint_idx in enumerate(rigid_idxs):
        print(f'joint name {np.array(skel_type.joint_names)[joint_idx]}')
        print(f'LBS hops: {np.array(skel_type.joint_names)[rigid_idxs][hop_mask[i].cpu().numpy() > 0]}')
    """

    return hop_mask, nb_joints

def calculate_kinematic(
        rest_pose, 
        bones, 
        root_locs=None, 
        skel_type=SMPLSkeleton,
        unroll_kinematic_chain=True,
    ):
    """ Turn rest_pose into a particular body pose specified by bones (in axis-ang rotation)
    """

    # the indices could be redundant, only calculate the unique ones

    #bone = self.bones[unique_indices]
    N, N_J, _ = bones.shape

    rest_pose = rest_pose.reshape(1, N_J, 3)
    rest_pose = rest_pose.expand(N, N_J, 3)
    root_id = skel_type.root_id
    joint_trees = skel_type.joint_trees

    rots = axisang_to_rot(bones)

    l2ws = []
    root_l2w = mat_to_hom_torch(torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1))
    l2ws.append(root_l2w)

    # create joint-to-joint transformation
    children_rots = torch.cat([rots[:, :root_id], rots[:, root_id+1:]], dim=1)
    children_locs = torch.cat([rest_pose[:, :root_id], rest_pose[:, root_id+1:]], dim=1)[..., None]
    parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
    parent_locs = rest_pose[:, parent_ids, :, None]

    joint_rel_transforms = mat_to_hom_torch(
            torch.cat([children_rots, children_locs - parent_locs], dim=-1).view(-1, 3, 4)
    ).view(-1, N_J-1, 4, 4)

    if unroll_kinematic_chain:
        l2ws = unrolled_kinematic_chain(root_l2w, joint_rel_transforms)
        l2ws = torch.cat(l2ws, dim=-3)
    else:
        for i, parent in enumerate(parent_ids):
            l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
        l2ws = torch.stack(l2ws, dim=-3)

    # can't do inplace, so do this instead
    if root_locs is None:
        root_locs = torch.zeros(N, 4, 4).to(l2ws.device)
    else:
        zeros = torch.zeros(N, 4, 3).to(l2ws.device)
        # pad to obtain (N, 4)
        root_locs = torch.cat([root_locs, torch.zeros(N, 1)], dim=-1)[..., None]

        # pad to (N, 4, 4)
        root_locs = torch.cat([zeros, root_locs], dim=-1)

    # add pelvis shift
    l2ws = l2ws + root_locs[:, None]

    # inverse l2ws to get skts (w2ls)
    skts = torch.inverse(l2ws)

    # reconstruct the originally requested sequence
    kp = l2ws[..., :3, -1]

    return kp, skts

def unrolled_kinematic_chain(root_l2w, joint_rel_transforms):
    """
    unrolled kinematic chain for SMPL skeleton. Should run faster..
    """

    root_l2w = root_l2w[:, None]
    # parallelize left_hip (1), right_hip (2) and spine1 (3)
    # Note: that joint indices are substracted by 1 for joint_rel_transforms
    # because root is excluded
    chain_1 = root_l2w.expand(-1, 3, 4, 4) @ joint_rel_transforms[:, 0:3]

    # parallelize left_knee (4), right_knee (5) and spine2 (6)
    chain_2 = chain_1 @ joint_rel_transforms[:, 3:6]

    # parallelize left_ankle (7), right_angle (8) and spine3 (9)
    chain_3 = chain_2 @ joint_rel_transforms[:, 6:9]

    # parallelize left_foot (10), right_foot (11),
    # neck (12), left_collar (13) and right_collar (14)
    # Note: that last 3 are children of spine3
    chain_4 = chain_3[:, [0, 1, 2, 2, 2]] @ joint_rel_transforms[:, 9:14]

    # parallelize head (15), left_shoulder (16), right_shoulder (17)
    # Note: they connect to neck, left_collar, and right_collar respectively
    # i.e., the last 3 elements in chain_4
    chain_5 = chain_4[:, -3:] @ joint_rel_transforms[:, 14:17]

    # parallelize left_elbow (18) and right_elbow (19)
    # Note: connect to left_collar and right_collar respectively,
    # i.e., the last 2 elelments in chain_5
    chain_6 = chain_5[:, -2:] @ joint_rel_transforms[:, 17:19]

    # parallelize left_wrist (20) and right_wrist (21)
    chain_7 = chain_6 @ joint_rel_transforms[:, 19:21]

    # parallelize left_wrist (22) and right_wrist (23)
    chain_8 = chain_7 @ joint_rel_transforms[:, 21:23]

    return [root_l2w, chain_1, chain_2, chain_3,
            chain_4, chain_5, chain_6, chain_7, chain_8]

def farthest_point_sampling(
    p, 
    n=1600, 
    init_idx=0, 
    chunk=100, 
    selected=None,
    compute_dist=lambda a, b, *args, **kwargs: ((a-b)**2).mean(-1),
):
    '''
    My favorite algorithm
    '''
    
    indices = torch.zeros(n).long()

    if selected is None:
        indices[0] = init_idx
    
        # initialize distance to all point from the starting center
        dists = compute_dist(p[indices[0]:indices[0]+1], p, chunk=chunk)
    else:
        # first compute the distance from p to all the selected points
        dists = compute_dist(p, selected, chunk=chunk).min(dim=-1).values
        indices[0] = dists.argmax(dim=0)
        dists = torch.minimum(
                    compute_dist(p[indices[0]:indices[0]+1], p, chunk=chunk)[0],
                    dists,
                )

    if len(dists.shape) > 1:
        assert dists.shape[0] == 1
        dists = dists[0]

    for i in range(1, n):
        new_idx = dists.argmax(dim=0)
        indices[i] = new_idx
        
        new_dists = compute_dist(p[new_idx:new_idx+1], p, chunk=chunk)

        if len(new_dists.shape) == 2:
            assert new_dists.shape[0] == 1
            new_dists = new_dists[0]

        dists = torch.minimum(dists, new_dists)
    return indices

def rasterize_points(
        points, 
        point_feats,
        c2ws, 
        K, 
        img_size=(1000, 1000),
        radius=0.005,
        points_per_pixel=1,
        background_color=torch.tensor([1.0, 1.0, 1.0]),
        render=False,
        **kwargs,
    ):
    # TODO: we have a mix of torch and numpy tensors..
    """ Run p3d rasterizer, assuming same image size for all batches
    points: [B, :, 3] 3d point coordinates
    point_feats: [B, :, C] point features
    c2ws: [B, 4, 4] camera-to-world matrices
    K: [B, 3, 3] camera intrinsics
    img_size: [H, W] image size
    """
    # Step 1.: create camera
    w2cs = nerf_c2w_to_extrinsic_th(c2ws)
    R = w2cs[:, :3, :3]
    T = w2cs[:, :3, 3]
    K = K[..., :3, :3]

    img_size_tensor = torch.tensor(img_size)[None].expand(points.shape[0], -1)
    p3d_cam = cameras_from_opencv_projection(
        R, T, K,
        image_size=img_size_tensor,
    )
    # Step 2.: create pointclouds
    point_clouds = Pointclouds(points=points, features=point_feats)

    # Step 3.: create rasterizer
    raster_settings = PointsRasterizationSettings(
        image_size=img_size, 
        radius=radius,
        points_per_pixel=points_per_pixel,
    )
    rasterizer = PointsRasterizer(cameras=p3d_cam, raster_settings=raster_settings)

    # Step 4.: rasterize
    point_frag = rasterizer(point_clouds, **kwargs)


    outputs = {'point_frags': point_frag}
    # Step opt.: create renderer
    if render:
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(
                background_color=background_color,
            )
        )
        rendering = renderer(point_clouds, **kwargs)
        outputs['rendering'] = rendering
    return outputs

def k_medoids(similarity_matrix, k):
    """ From https://github.com/jiachangliu/k_medoids
    """
    
    # Step 1: Select initial medoids
    num = len(similarity_matrix)
    row_sums = torch.sum(similarity_matrix, dim=1)
    normalized_sim = similarity_matrix.T / row_sums
    normalized_sim = normalized_sim.T
    priority_scores = -torch.sum(normalized_sim, dim=0)
    values, indices = priority_scores.topk(k)
    
    tmp = -similarity_matrix[:, indices]
    tmp_values, tmp_indices = tmp.topk(1, dim=1)
    min_distance = -torch.sum(tmp_values)
    cluster_assignment = tmp_indices.resize_(num)
    
    # Step 2: Update medoids
    for i in range(k):
        sub_indices = (cluster_assignment == i).nonzero()
        sub_num = len(sub_indices)
        sub_indices = sub_indices.resize_(sub_num)
        sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
        sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
        sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
        sub_medoid_index = torch.argmin(sub_row_sums)
        # update the cluster medoid index
        indices[i] = sub_indices[sub_medoid_index]
        
    # Step 3: Assign objects to medoids
    tmp = -similarity_matrix[:, indices]
    tmp_values, tmp_indices = tmp.topk(1, dim=1)
    total_distance = -torch.sum(tmp_values)
    cluster_assignment = tmp_indices.resize_(num)
        
    while (total_distance < min_distance):
        min_distance = total_distance
        # Step 2: Update medoids
        for i in range(k):
            sub_indices = (cluster_assignment == i).nonzero()
            sub_num = len(sub_indices)
            sub_indices = sub_indices.resize_(sub_num)
            sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
            sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
            sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
            sub_medoid_index = torch.argmin(sub_row_sums)
            # update the cluster medoid index
            indices[i] = sub_indices[sub_medoid_index]

        # Step 3: Assign objects to medoids
        tmp = -similarity_matrix[:, indices]
        tmp_values, tmp_indices = tmp.topk(1, dim=1)
        total_distance = -torch.sum(tmp_values)
        cluster_assignment = tmp_indices.resize_(num)
        
    return indices

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 = vec1
    vec2 = vec2

    norm_a = np.linalg.norm(vec1, axis=-1) + 1e-8 
    norm_b = np.linalg.norm(vec2, axis=-1) + 1e-8
    a = (vec1 / norm_a[..., None]).reshape(-1, 3)
    b = (vec2 / norm_b[..., None]).reshape(-1, 3)
    v = np.cross(a, b, axis=-1)

    # do product
    c = (a * b).sum(axis=-1)
    s = np.linalg.norm(v, axis=-1) + 1e-8
    I = np.eye(3).reshape(1, 3, 3).repeat(len(v), axis=0)
    k_mat = np.eye(3).reshape(1, 3, 3).repeat(len(v), axis=0)

    k_mat[:, 0, 1] = -v[:, 2]
    k_mat[:, 0, 2] =  v[:, 1]
    k_mat[:, 1, 0] =  v[:, 2]
    k_mat[:, 1, 2] = -v[:, 0]
    k_mat[:, 2, 0] = -v[:, 1]
    k_mat[:, 2, 1] =  v[:, 0]

    rot =  I + k_mat + k_mat @ k_mat * ((1 - c) / (s ** 2))[:, None, None]

    v_check = np.abs(v).sum(axis=-1)
    mask = (v_check > 1e-6)[:, None, None]
    signed_I = np.sign(c)[..., None, None] * np.eye(3).reshape(1, 3, 3).repeat(len(v), axis=0)
    rot = rot * mask + signed_I * (1 - mask)

    return rot


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