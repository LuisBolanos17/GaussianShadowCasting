import torch
import torch.nn as nn
import numpy as np
from core.utils.skeleton_utils import (
    Skeleton, 
    SMPLSkeleton, 
    calculate_kinematic,
)
from typing import Optional, Dict, Any
from einops import rearrange

class PoseOpt(nn.Module):
    """
    Pose optimization module

    """

    def __init__(
        self,
        rest_pose: torch.Tensor,
        kp3d: np.ndarray,
        bones: np.ndarray, # FIXME: shouldn't call it bones anymore
        skel_type: Skeleton,
        emb_dim: int = 16, # unique-identifier for the per-pose code
        depth: int = 4,
        width: int = 128,
        residual_scale: float = 0.1, # scale the output
        rot_6d: bool = False,
        n_embs: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert skel_type == SMPLSkeleton, 'Only SMPLSkeleton is supported for now'
        assert depth >= 2, 'Depth must be at least 2'
        self.skel_type = skel_type
        self.n_embs = n_embs if n_embs is not None else len(kp3d)
        self.residual_scale = residual_scale

        rvecs = torch.tensor(bones)
        # NOTE: this is different from original A-NeRF implementation, but should 
        # work better
        pelvis = torch.tensor(kp3d[:, skel_type.root_id])

        self.register_buffer('rest_pose', rest_pose, persistent=False)
        self.register_buffer('pelvis', pelvis, persistent=False)
        self.register_buffer('rvecs', rvecs, persistent=False)

        self.pose_embs = nn.Embedding(self.n_embs, emb_dim)

        # initialize refinement network
        rvec_dim = 3 if not rot_6d else 6
        N_joints = kp3d.shape[1]
        rvec_dim = rvec_dim * N_joints
        net = [nn.Linear(rvec_dim + emb_dim, width), nn.ReLU()]
        for _ in range(depth - 2):
            net.extend([nn.Linear(width, width), nn.ReLU()])
        net.append(nn.Linear(width, rvec_dim + 3))
        self.refine_net = nn.Sequential(*net)

    def forward(
        self,
        network_inputs: Dict[str, Any],
        kp3d: torch.Tensor,
        bones: torch.Tensor, # FIXME: rvec
        kp_idxs: torch.LongTensor,
        N_unique: int = 1,
        unroll: bool = True,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        kp3d: torch.Tensor (B, J, 3) -- joint locations
        bonees: torch.Tensor (B, J, 3 or 6) -- rotation vector for each joint/bone
        kp_idxs: torch.LongTensor -- index of the poses 
        unroll: bool -- unroll the for loop for kinematic calculation, this gives faster forward pass
        """
        N_samples = len(kp3d)
        skip = N_samples // N_unique
        N, J, _ = bones.shape
        # The body poses are organized in (N_images * N_rays_per_image, ...)
        # meaning that there will be N_rays_per_image redundant poses (the same image has the same pose)
        # -> skip the redundant one
        rvecs = rearrange(bones[::skip], 'b j d -> b (j d)')
        kp_unique = kp_idxs[::skip]
        kp3d_unique = kp3d[::skip]

        embs = self.pose_embs(kp_unique)
        pelvis = self.pelvis[kp_unique].clone()
        assert torch.allclose(pelvis, kp3d_unique[:, self.skel_type.root_id], atol=1e-5), 'Pelvis should be the same as root joint'

        x = torch.cat([rvecs, embs], dim=-1)
        residual = self.refine_net(x) * self.residual_scale
        residual_rvecs, residual_pelvis = torch.split(residual, [rvecs.shape[-1], 3], dim=-1)
        rvecs = rearrange(rvecs + residual_rvecs, 'b (j d) -> b j d', j=J)
        pelvis = pelvis + residual_pelvis
        rest_pose = self.rest_pose.clone()

        kp, skts = calculate_kinematic(
            rest_pose[None],
            rvecs,
            root_locs=pelvis,
            skel_type=self.skel_type,
            unroll_kinematic_chain=unroll,
        )

        # to update
        # 1. kp3d, 2. skts, 3. bones
        # -> expand the shape
        kp = rearrange(kp[:, None].expand(-1, skip, -1, -1), 'i s j d -> (i s) j d')
        skts = rearrange(skts[:, None].expand(-1, skip, -1, -1, -1), 'i s j k d -> (i s) j k d')
        rvecs = rearrange(rvecs[:, None].expand(-1, skip, -1, -1), 'i s j d -> (i s) j d')
        network_inputs.update(
            kp3d=kp,
            skts=skts,
            bones=rvecs, # FIXIT: nameing
        )
        return network_inputs

    @torch.no_grad()
    def export_optimized_pose(self, unroll: bool = True):
        pelvis = self.pelvis.clone()
        rvecs = self.rvecs.clone()
        B, J, _ = rvecs.shape
        rest_pose = self.rest_pose.clone()

        embs = self.pose_embs(torch.arange(len(rvecs)))

        x = torch.cat([rvecs, embs], dim=-1)
        residual = self.refine_net(x) * self.residual_scale

        residual_rvecs, residual_pelvis = torch.split(residual, [rvecs.shape[-1], 3], dim=-1)
        rvecs = rearrange(rvecs + residual_rvecs, 'b (j d) -> b j d', j=J)
        pelvis = pelvis + residual_pelvis

        kp, skts = calculate_kinematic(
            rest_pose[None],
            rvecs,
            root_locs=pelvis,
            skel_type=self.skel_type,
            unroll_kinematic_chain=unroll,
        )

        return {
            'kp3d': kp,
            'skts': skts,
            'bones': rvecs,
        }

# TODO: fixing legacy code
# ----- belows are all legacy code -----
def create_popt(args, data_attrs, ckpt=None, device=None):


    # get attributes from dict
    skel_type = data_attrs['skel_type']
    rest_pose = data_attrs['rest_pose'].reshape(-1, len(skel_type.joint_names), 3)
    rest_pose = torch.tensor(rest_pose, requires_grad=False)
    beta = torch.tensor(data_attrs['betas'])
    init_kps = torch.tensor(data_attrs['kp3d'])
    init_bones = torch.tensor(data_attrs['bones'])
    kp_map = data_attrs.get('kp_map', None)
    kp_uidxs = data_attrs.get('kp_uidxs', None)
    rest_pose_idxs = data_attrs.get('rest_pose_idxs', None)

    poseopt_kwargs = {}
    poseopt_class = PoseOptLayer

    popt_layer = poseopt_class(init_kps.clone().to(device),
                               init_bones.clone().to(device),
                               rest_pose.to(device),
                               beta=beta,
                               skel_type=skel_type,
                               kp_map=kp_map,
                               kp_uidxs=kp_uidxs,
                               rest_pose_idxs=rest_pose_idxs,
                               use_cache=args.opt_pose_cache,
                               use_rot6d=args.opt_rot6d,
                               **poseopt_kwargs)

    grad_opt_vars = list(popt_layer.parameters())
    pose_optimizer = torch.optim.Adam(params=grad_opt_vars,
                                      lr=args.opt_pose_lrate,
                                      betas=(0.9, 0.999))

    # initalize anchor (for regularization loss)
    anchor_kps, anchor_bones, anchor_beta = init_kps, init_bones, beta

    if (ckpt is not None or args.init_poseopt is not None) and not (args.no_poseopt_reload):
        # load from checkpoint
        pose_ckpt = torch.load(args.init_poseopt) if args.init_poseopt is not None else ckpt
        popt_layer.load_state_dict(pose_ckpt["poseopt_layer_state_dict"])
        print("WARNING: pose-opt statedict logging is temporarily disabled for exp purpose!")
        #pose_optimizer.load_state_dict(pose_ckpt["pose_optimizer_state_dict"])

        if "poseopt_anchors" in pose_ckpt:
            anchor_kps = pose_ckpt["poseopt_anchors"]["kps"]
            anchor_bones = pose_ckpt["poseopt_anchors"]["bones"]
            anchor_beta = pose_ckpt["poseopt_anchors"]["beta"]

        if args.use_ckpt_anchor:
            # use the poses data from ckpt as optimization constraint
            with torch.no_grad():
                anchor_kps, anchor_bones, _, anchor_rots = popt_layer(torch.arange(anchor_bones.shape[0]))
            anchor_kps, anchor_bones = anchor_kps.cpu().clone(), anchor_bones.cpu().clone()
            anchor_beta = popt_layer.get_beta()

    # recompute anchor_rots to ensure it's consistent with bones
    anchor_rots = axisang_to_rot(anchor_bones.view(-1, 3)).view(*anchor_kps.shape[:2], 3, 3)
    popt_anchors = {"kps": anchor_kps, "bones": anchor_bones, "rots": anchor_rots, "beta": anchor_beta}

    if popt_layer.use_cache:
        print("update cache for faster forward")
        popt_layer.update_cache()

    pose_optimizer.zero_grad() # to remove gradients from ckpt
    popt_kwargs = {'popt_anchors': popt_anchors,
                   'popt_layer': popt_layer,
                   'skel_type': skel_type}

    return pose_optimizer, popt_kwargs

def mat_to_hom(mat):
    """
    To homogeneous coordinates
    """
    last_row = torch.tensor([[0., 0., 0., 1.]]).to(mat.device)

    if mat.dim() == 3:
        last_row = last_row.expand(mat.size(0), 1, 4)
    return torch.cat([mat, last_row], dim=-2)

def load_bones_from_state_dict(state_dict, device='cpu'):
    bones = state_dict['poseopt_layer_state_dict']['bones']
    if bones.shape[-1] == 6:
        N, N_J, _ = bones.shape
        bones = bones.view(-1, 6)
        rots = rot6d_to_rotmat(bones)
        bones = rot_to_axisang(rots).view(N, N_J, -1)
    return bones.to(device)

def load_poseopt_from_state_dict(state_dict):
    '''
    Assume no kp_map for now...
    '''
    print("Loading pose opt state dict")
    pelvis = state_dict['poseopt_layer_state_dict']['pelvis']
    # bones do not necessarily have the right shape
    bones = state_dict['poseopt_layer_state_dict']['bones']

    # prob if it is multiview
    kp_map = kp_uidxs = None
    if 'kp_map' in state_dict['poseopt_layer_state_dict']:
        kp_map = state_dict['poseopt_layer_state_dict']['kp_map'].cpu().numpy()
        kp_uidxs = state_dict['poseopt_layer_state_dict']['kp_uidxs'].cpu().numpy()

    N, N_J, N_D = pelvis.shape[0], *bones.shape[1:]
    # multiview setting has root bone removed and stored separately,
    # so need to add 1 back to the bone dimension
    if kp_map is not None:
        N_J += 1
    dummy_kp = torch.zeros(N, N_J, 3)
    dummy_bone = torch.zeros(N, N_J, 3)

    poseopt = PoseOptLayer(dummy_kp, dummy_bone, dummy_kp[0:1],
                           use_rot6d= N_D==6, kp_map=kp_map, kp_uidxs=kp_uidxs)
    poseopt.load_state_dict(state_dict['poseopt_layer_state_dict'])
    return poseopt


def pose_ckpt_to_pose_data(path=None, popt_sd=None, ext_scale=0.001, legacy=False, skel_type=SMPLSkeleton):
    # TODO: ext_scale is hard-coded for now. Fix it later
    if popt_sd is None:
        popt_sd = torch.load(path)['poseopt_layer_state_dict']
    poseopt = load_poseopt_from_state_dict({'poseopt_layer_state_dict': popt_sd})

    with torch.no_grad():
        pelvis = poseopt.get_pelvis().cpu().numpy()
        bones = poseopt.get_bones().cpu().numpy()
        rest_pose = poseopt.get_rest_pose()[0].cpu().numpy()

    if legacy:
        pelvis[..., 1:] *= -1
        rest_pose = np.concatenate([ rest_pose[..., :1],
                                    -rest_pose[..., 2:3],
                                     rest_pose[..., 1:2],], axis=-1)
        bones = np.concatenate([ bones[..., :1],
                                -bones[..., 2:3],
                                 bones[..., 1:2],], axis=-1)
        root_rot = axisang_to_rot(torch.tensor(bones[..., :1, :].reshape(-1, 3)))
        rot_on_root = torch.tensor([[1., 0., 0.],
                                    [0., 0.,-1.],
                                    [0., 1., 0.]]).float()
        root_rot = rot_to_axisang(rot_on_root[None, :3, :3] @ root_rot).reshape(-1, 3).cpu().numpy()
        bones[..., 0, :] = root_rot
        print(f'{path} - legacy')

    # create local-to-world matrices and add global translation
    l2ws = np.array([get_smpl_l2ws(b, rest_pose=rest_pose, scale=1.) for b in bones])
    l2ws[..., :3, -1] += pelvis[:, None]
    kp3d = l2ws[..., :3, -1].copy().astype(np.float32)
    skts = np.linalg.inv(l2ws).astype(np.float32)
    l2ws = l2ws.astype(np.float32)
    # get cylinders
    cyls = get_kp_bounding_cylinder(kp3d, ext_scale=ext_scale, skel_type=skel_type,
                                    extend_mm=250, head='-y').astype(np.float32)
    return kp3d, bones, skts, cyls, rest_pose, pelvis

