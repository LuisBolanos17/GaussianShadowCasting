import os
import torch 
import mcubes
import trimesh
import imageio
import argparse

import numpy as np

from hydra.utils import instantiate
from omegaconf import OmegaConf

from train import build_model, find_ckpts
from core.utils.skeleton_utils import *
from core.utils.visualization import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.multiprocessing.set_start_method('spawn')


def get_dist_pts_to_lineseg(pts, p0, p1):
    """
    pts: query pts
    p0: the 1st endpoint of the lineseg
    p1: the 2nd endpoint of the lineseg
    """

    seg = p1 - p0
    seg_len = torch.norm(seg, dim=-1, p=2)
    seg_norm = seg / (seg_len[:, None] + 1e-6)

    vec = pts - p0

    # determine if the pts is in-between p0 and p1.
    # if so, the dist is dist(pts, seg)
    # otherwise it should be dist(p0, pts) or dist(p1, pts)
    dist_p0 = torch.norm(vec, dim=-1, p=2)
    dist_p1 = torch.norm(pts - p1, dim=-1, p=2)

    # unroll it here to save some computes..
    # dist_line = get_dist_pts_to_line(pts, p0, p1)
    cross = torch.cross(vec, seg.expand(*vec.shape), dim=-1)
    dist_line = torch.norm(cross, dim=-1, p=2) / (torch.norm(seg, dim=-1, p=2) + 1e-6)

    # we can check if it's in-between by projecting vec to seg and check the length/dir
    proj = (vec * seg_norm).sum(-1)

    dist = torch.where(
        proj < 0, # case1
        dist_p0,
        torch.where(
            proj > 1, # case2
            dist_p1,
            dist_line # case3
        )
    )

    return dist


def get_bone_dist(pts, kps, skel_type=SMPLSkeleton):
    """ Compute the distance from a 3d points to a vector (bone).
    """

    joint_trees = torch.tensor(skel_type.joint_trees)
    nonroot_id = torch.tensor(skel_type.nonroot_id)

    if pts.dim() == 3:
        pts = pts[:, :, None, :].expand(-1, -1, len(nonroot_id), -1)

    kps_parent = kps[:, joint_trees, :]
    kps_parent = kps_parent[:, None, nonroot_id, :]
    kps_nonroot = kps[:, None, nonroot_id, :]

    dist_to_bone = get_dist_pts_to_lineseg(pts, kps_nonroot, kps_parent)

    if skel_type == HARESkeleton:
        pass
    return dist_to_bone


def farthest_point_sampling(p, n=1600, init_idx=0, chunk=100):
    """ Grab points that are farthest from each other, this is a clustering algorithm
    that minimize the maximum distance from a point to its nearest cluster center.

    Args:
        p: ndarray (N, K), N points with feature dimension K
        n: int, number of points to sample
        init_idx: int, the index of the first point to start with
        chunk: int, batchify the computation to avoid OOM

    Return:
        indices: ndarray (n), indices of the sampled points
    """
    
    indices = torch.zeros(n).long()
    indices[0] = init_idx
    
    # initialize distance to all point from the starting center
    dists = compute_dist(p[indices[0]:indices[0]+1], p, chunk=chunk)
    
    for i in range(1, n):
        new_idx = dists.argmax(dim=0)
        
        indices[i] = new_idx
        new_dists = compute_dist(p[new_idx:new_idx+1], p, chunk=chunk)
        dists = torch.minimum(dists, new_dists)
    return indices


def compute_dist(a, b, dist_fn=lambda a, b: ((a-b)**2).mean(-1), chunk=100):
    """ Batchify distance computation to avoid OOM.

    Args:
        a: tensor (N, D), N points with feature dimension D
        b: tensor (M, D), M points with feature dimension D
        dist_fn: function, distance function
        chunk: int, batchify the computation to avoid OOM
    
    Return:
        dists: tensor (N, M), distance between each point in a and b
    """
    dists = []
    for i in range(0, len(b), chunk):
        d = dist_fn(a, b[i:i+chunk])
        dists.append(d)
        
    return torch.cat(dists)


def label_surface_points(points, model, n_iters=5):
    """ Assign surface points to different body parts based on DANBO volumes.

    Args:
        points: tensor (N, 3), surface points
        model: a pretrained model
        n_iters: int, number of iterations before assigning the points to the closest box
    
    Return:
        labels: tensor (N), the label of each point
    """
    skel_profile = model.skel_profile
    N_J = len(model.skel_type.joint_names)
    rigid_idxs = np.array([
            i for i in range(N_J) if i not in skel_profile['rigid_ignore']
    ])

    box_centers = torch.zeros(1, len(rigid_idxs), 3)
    box_centers = model.pts_embedder.unalign_pts(box_centers,rigid_idxs=rigid_idxs)
    box_centers = (box_centers + model.rest_pose[None, rigid_idxs])
    dist_to_boxes = ((points[:, None] - box_centers).pow(2.)).sum(dim=-1)
    axis_scale = model.get_axis_scale().detach()
    rest_pose = model.rest_pose

    # initialize the labels to the closest box
    # Note: this does not guarantee the points to be valid in the box!
    dist_sorted = dist_to_boxes.sort().indices
    label_ptrs = torch.zeros(len(points)).long()

    # now, go through the points to check if they are valid.
    # if not, assign them to the next closest boxes    
    all_valid = False
    iter_cnt = 0
    while not all_valid and iter_cnt < n_iters:
        iter_cnt += 1 
        labels = dist_sorted[torch.arange(len(points)), label_ptrs]

        invalid_idxs = []
        for i, ri in enumerate(rigid_idxs):
            label_idxs = torch.where(labels==i)[0].clone()
            label_pts = points[label_idxs]
            r_loc = rest_pose[ri:ri+1]
            label_pts_j = (label_pts - r_loc)
            T = model.pts_embedder.transforms[ri:ri+1]
            Rpts = (T[..., :3, :3] @ label_pts_j[..., None])[..., 0]
            apts = Rpts + T[..., :3, -1]
            apts =  apts / axis_scale[ri:ri+1]
            
            invalid = ((apts.abs() > 1.).sum(-1) > 0).float()
            invalid_idxs.append(label_idxs[invalid==1])
        invalid_idxs = torch.cat(invalid_idxs)
        if len(invalid_idxs) > 0:
            label_ptrs[invalid_idxs] += 1
        else:
            all_valid = True
    labels = dist_sorted[torch.arange(len(points)), label_ptrs]
    return labels


def extract_mcubes(configs):
    """ Extract surface points using marching cubes
    """

    pretrained_path = configs.pretrained_path
    pretrained_ckpt = os.path.join(pretrained_path, f'{configs.ckpt:07d}.th')
    pretrained_config = OmegaConf.load(os.path.join(pretrained_path, 'config.yaml'))

    data_attrs = instantiate(pretrained_config.dataset).get_meta()
    pretrained_config.ckpt_path = pretrained_ckpt
    ckpt, _ = find_ckpts(pretrained_config, pretrained_path)
    pretrained_model = build_model(pretrained_config.model, data_attrs, ckpt).cuda()

    # obtain all the surface points in each of the volume box
    skel_type = pretrained_model.skel_type
    skel_profile = pretrained_model.skel_profile
    rest_pose = pretrained_model.rest_pose
    rigid_idxs = np.array([
        i for i in range(len(skel_type.joint_names))
        if i not in skel_profile['rigid_ignore']
    ])
    N_J = len(skel_type.joint_names)

    res = configs.res
    radius = configs.radius
    threshold = configs.threshold
    axis_scale = pretrained_model.get_axis_scale().detach()

    t = np.linspace(-radius, radius, res)
    grid_pts = torch.tensor(np.stack(np.meshgrid(t, t, t), axis=-1).astype(np.float32))
    geom_inputs = {
        'pts': grid_pts.reshape(-1, 3),
        'bones': torch.zeros(1, N_J, 3)
    }

    point_shift = 0
    if skel_type != SMPLSkeleton:
        assert skel_type in [HARESkeleton, WOLFSkeleton, MixamoSkeleton]
        point_shift = rest_pose[:1]
        geom_inputs['kp3d'] = rest_pose[None]  - point_shift[None]
        rest_skts = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, N_J, 1, 1)
        rest_skts[..., :3, -1] = -(rest_pose - point_shift)
        geom_inputs['skts'] = rest_skts

    pretrained_model.eval()
    preds = pretrained_model(geom_inputs, forward_type='geometry', chunk=1024*64)
    density = preds['density'].reshape(res, res, res).cpu().numpy()
    density[density < threshold] = 0.
    density = np.maximum(density, 0)

    vertices, triangles = mcubes.marching_cubes(
                                density.reshape(res, res, res), 
                                threshold)
            
    # scale the vertices back to the original size
    # Note: in mcubes, the range is [0, res-1]
    #mesh = trimesh.Trimesh(vertices / res - .5, triangles)
    #mesh.export('hare_test.ply')
    vertices = radius * 2 * (vertices / (res-1) - 0.5)

    # differ by a rotate along z axis
    rot = np.array([
        [ 0., 1., 0.],
        [-1., 0., 0.],
        [ 0., 0., 1.],
    ]).astype(np.float32)
    surface_pts = torch.tensor(vertices @ rot).float()

    # label each vertex with the closest box center
    if not isinstance(point_shift, int):
        assert skel_type in [HARESkeleton, WOLFSkeleton, MixamoSkeleton]
        surface_pts = surface_pts + point_shift
    labels = label_surface_points(surface_pts, pretrained_model)

    # extract points
    val_range = [-1., 1.]
    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'lightblue']
    N_fps = configs.n_points
    fig = None
    extracted = {
        'anchor_pts': [],
        'canon_pts': [],
        'axis_scale': axis_scale.cpu(),
    }

    for i, ri in enumerate(rigid_idxs):
        cnt = (labels == i).sum()
        label_pts = surface_pts[labels==i].clone()
        
        r_loc = rest_pose[ri:ri+1]
        label_pts_j = (label_pts - r_loc)
        T = pretrained_model.pts_embedder.transforms[ri:ri+1]
        Rpts = (T[..., :3, :3] @ label_pts_j[..., None])[..., 0]
        apts = Rpts + T[..., :3, -1]
        apts =  apts / axis_scale[ri:ri+1]
        
        invalid = ((apts.abs() > 1.).sum(-1) > 0).float()
        valid = 1 - invalid
        valid_pts = label_pts[valid > 0]

        if len(valid_pts) > N_fps:
            fps_idx = farthest_point_sampling(valid_pts, n=N_fps).sort().values
        else:
            fps_idx = torch.arange(len(valid_pts))
        color = colors[i % len(colors)]

        # for visualization
        fig = plot_points3d(
            (valid_pts[fps_idx].reshape(-1, 3) - point_shift).cpu().numpy(), 
            color=color, 
            fig=fig, 
            x_range=val_range, 
            y_range=val_range, 
            z_range=val_range
        )
        print(f'joint {skel_type.joint_names[ri]}: {cnt-invalid.sum()}/{cnt} (invalid count: {invalid.sum()}) ')

        valid_anchors = apts[valid > 0][fps_idx]
        extracted['anchor_pts'].append(valid_anchors)
        extracted['canon_pts'].append(valid_pts[fps_idx])

    # save the extracted point cloud image
    img = byte2array(fig.to_image(format='png'))
    torch.save(extracted, configs.output_name)
    imageio.imwrite(configs.output_name.replace('.th', '.png'), img)
    print(f'Done extraction. Output saved to {configs.output_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract points from a lightly trained model')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='path to the pretrained model')
    parser.add_argument('--ckpt', type=int, default=10000,
                        help='checkpoints to extract from')
    parser.add_argument('--n_points', type=int, default=1000,
                        help='number of points to extract for each body part')
    parser.add_argument('--threshold', type=float, default=20.,
                        help='threshold for determining if a point is inside or outside the body part')
    parser.add_argument('--trunk_shift', type=float, default=-0.01,
                        help='shift the trunk points by this amount because it is usually bloated')
    parser.add_argument('--res', type=int, default=512,
                        help='resolution of the grid to sample from')
    parser.add_argument('--radius', type=float, default=1.3,
                        help='bound for the grids')
    parser.add_argument('--output_name', type=str, default='extracted_points.th',)
    configs = parser.parse_args()

    extract_mcubes(configs)

