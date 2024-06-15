import io
import cv2
import numpy as np
import plotly
import plotly.graph_objects as go

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .skeleton_utils import (
    get_skeleton_type,
    create_axis_boxes, 
    create_plane,
    boxes_to_pose, 
    SMPLSkeleton, 
    swap_mat,
    dist_to_joints,
    skeleton3d_to_2d,
)

from PIL import Image

def byte2array(byte):
    buf = io.BytesIO(byte)
    img = Image.open(buf)
    return np.asarray(img)

def plot_points3d_mpl(points, val_range=1.2, marker_size=1.):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*points, c='red', s=marker_size)
    ax.set_xlim([-val_range, val_range])
    ax.set_ylim([-val_range, val_range])
    ax.set_zlim([-val_range, val_range])
    ax.view_init(elev=40, azim=-70, roll=10)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clear()
    plt.close(fig)
    return image

def plot_boxes_mpl(corners, val_range=1.2, marker_size=1., alpha=0.3, line_color='teal'):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for box in corners:
        ax.scatter(*box.transpose(1, 0), c='red',s=marker_size)
        ax.plot(*box[[0, 1, 4, 2, 0]].transpose(1, 0), c='teal', alpha=alpha)
        ax.plot(*box[[0, 3, 5, 2]].transpose(1, 0), c='teal', alpha=alpha)
        ax.plot(*box[[4, 7, 6, 1]].transpose(1, 0), c='teal', alpha=alpha)
        ax.plot(*box[[7, 5]].transpose(1, 0), c='teal', alpha=alpha)
        ax.plot(*box[[6, 3]].transpose(1, 0), c='teal', alpha=alpha)
        
    ax.set_xlim([-val_range, val_range])
    ax.set_ylim([-val_range, val_range])
    ax.set_zlim([-val_range, val_range])
    ax.view_init(elev=40, azim=-70, roll=10)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clear()
    plt.close(fig)

    return image

def plot_boxes(corners, fig=None, label=False, marker_size=3,
               color="orange", line_width=2, line_color="green"):

    box_corners = []
    box_contours = []
    for corner in corners:
        # plot the 8 corners
        box_corners.append(go.Scatter3d(x=corner[..., 0],
                                        y=corner[..., 1],
                                        z=corner[..., 2],
                                        mode="markers", marker=dict(size=marker_size),
                                        line=dict(color=color)
                                       ))
        xs, ys, zs = corner[..., 0], corner[..., 1], corner[..., 2]
        # connect corners
        lx, ly, lz = [], [], []
        connections = [0, 1, 6, 3, 0, 2, 5, 3, 6, 7, 5, 2, 4, 7, 4, 1]
        for j in connections:
            lx += [xs[j]]
            ly += [ys[j]]
            lz += [zs[j]]

        box_contours.append(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                                 line=dict(color=line_color, width=line_width),
                                 hoverinfo="none"))
    data = [*box_contours]

    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)
    return fig

def plot_skeleton3d(skel, fig=None, skel_id=None, skel_type=None,
                    cam_loc=None, layout_kwargs=None, line_width=2,
                    marker_size=3, blank=False, axis_range=[-1.0, 1.0],
                    color='blue'):
    """
    Plotting function for canonicalized skeleton
    """

    # x, y, z of lines in the center/left/right body part
    clx, cly, clz = [], [], []
    llx, lly, llz = [], [], []
    rlx, rly, rlz = [], [], []

    x, y, z = list(skel[..., 0]), list(skel[..., 1]), list(skel[..., 2])

    if skel_type is None:
        skel_type = get_skeleton_type(skel)

    joint_names = skel_type.joint_names
    joint_tree = skel_type.joint_trees
    root_id = skel_type.root_id

    for i, (j, name) in enumerate(zip(joint_tree, joint_names)):
        if "left" in name:
            lx, ly, lz = llx, lly, llz
        elif "right" in name:
            lx, ly, lz = rlx, rly, rlz
        else:
            lx, ly, lz = clx, cly, clz

        lx += [x[i], x[j], None]
        ly += [y[i], y[j], None]
        lz += [z[i], z[j], None]

    joint_names = [f"{i}_{name}" for i, name in enumerate(joint_names)]
    if skel_id is not None:
        joint_names = [f"{skel_id}_{name}" for name in joint_names]

    points = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=marker_size),
                          line=dict(color=color),
                          text=joint_names)
    center_lines = go.Scatter3d(x=clx, y=cly, z=clz, mode="lines",
                                line=dict(color="black", width=line_width),
                                hoverinfo="none")
    left_lines = go.Scatter3d(x=llx, y=lly, z=llz, mode="lines",
                              line=dict(color="blue", width=line_width),
                              hoverinfo="none")
    right_lines = go.Scatter3d(x=rlx, y=rly, z=rlz, mode="lines",
                               line=dict(color="red", width=line_width),
                               hoverinfo="none")
    data = [points, center_lines, left_lines, right_lines]

    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    camera = dict(up=dict(x=0, y=1, z=0))
    if cam_loc is not None:
        camera['eye'] = dict(x=cam_loc[0], y=cam_loc[1], z=cam_loc[2])
    

    root_id = skel_type.root_id
    root_loc = skel[root_id]
    x_range = root_loc[0] + np.array(axis_range)
    y_range = root_loc[1] + np.array(axis_range)
    z_range = root_loc[2] + np.array(axis_range)
    
    scene = dict(
                aspectmode='cube',
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
            )

    fig.update_layout(scene=scene, scene_camera=camera)
    if blank:
        if layout_kwargs is not None:
            print('WARNING! layout_kwargs is overwritten by blank=True')
        fig.update_layout(scene=dict(
            xaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
            yaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
            zaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
        ))
    elif layout_kwargs is not None:
        fig.update_layout(**layout_kwargs)
    return fig


def plot_vectors(
    vectors,
    fig=None, 
    layout_kwargs=None, 
    line_width=2,
    marker_size=3, 
    blank=False, 
    color='blue'
):
    """
    vectors: [B, 2, 3] (batch size, [start point, end point], [x, y, z])
    """
    lx, ly, lz = [], [], []

    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]


    for x_, y_, z_ in zip(x, y, z):
        lx += [x_[0], x_[1], None]
        ly += [y_[0], y_[1], None]
        lz += [z_[0], z_[1], None]


    # put markers on starting point
    points = go.Scatter3d(
        x=np.array(x[:, 0]), 
        y=np.array(y[:, 0]), 
        z=np.array(z[:, 0]), 
        mode="markers", 
        marker=dict(size=marker_size),
        line=dict(color=color),
    )
    points_e = go.Scatter3d(
        x=np.array(x[:, 1]), 
        y=np.array(y[:, 1]), 
        z=np.array(z[:, 1]), 
        mode="markers", 
        marker=dict(size=marker_size),
        line=dict(color=color),
    )
    lines = go.Scatter3d(
        x=lx, 
        y=ly, 
        z=lz, 
        mode="lines",
        line=dict(color="black", width=line_width),
        hoverinfo="none"
    )

    data = [points, lines, points_e]

    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    camera = dict(up=dict(x=0, y=1, z=0))

    
    scene = dict(
        aspectmode='cube',
    )

    fig.update_layout(scene=scene, scene_camera=camera)
    if blank:
        if layout_kwargs is not None:
            print('WARNING! layout_kwargs is overwritten by blank=True')
        fig.update_layout(scene=dict(
            xaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
            yaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
            zaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
        ))
    elif layout_kwargs is not None:
        fig.update_layout(**layout_kwargs)
    return fig

def plot_skeleton2d(skel, skel_type=None, img=None):

    if skel_type is None:
        skel_type = get_skeleton_type(skel)

    joint_names = skel_type.joint_names
    joint_tree = skel_type.joint_trees

    if img is not None:
        plt.imshow(img)

    for i, j in enumerate(skel):
        name = joint_names[i]
        parent = skel[joint_tree[i]]
        offset = parent - j

        if "left" in name:
            color = "red"
        elif "right" in name:
            color = "blue"
        else:
            color = "green"
        plt.arrow(j[0], j[1], offset[0], offset[1], color=color)

def plot_points3d(pts, fig=None, label=False, marker_size=3, color="orange", opacity=1.0,
                  x_range=None, y_range=None, z_range=None):

    if label:
        names = [f'{i}' for i in range(len(pts))]
    else:
        names = None
    pts = go.Scatter3d(x=pts[..., 0], y=pts[..., 1], z=pts[..., 2],
                       mode='markers', marker=dict(size=marker_size), text=names,
                       line=dict(color=color), opacity=opacity)


    data = [pts]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)
    scene = None
    if x_range is not None:
        scene = dict(
                    aspectmode='cube',
                    xaxis=dict(range=x_range),
                    yaxis=dict(range=y_range),
                    zaxis=dict(range=z_range),
                )
        fig.update_layout(scene=scene)
    return fig

def plot_cameras(extrinsics=None, viewmats=None, fig=None):
    if extrinsics is not None:
        viewmats = np.array([np.linalg.inv(ext) for ext in extrinsics])
    cam_pos = viewmats[:, :3, 3]
    rights = viewmats[:, :3, 0] * 0.5
    ups = viewmats[:, :3, 1] * 0.5
    fwds = viewmats[:, :3, 2] * 0.5

    rlx, rly, rlz = [], [], []
    ulx, uly, ulz = [], [], []
    flx, fly, flz = [], [], []

    for cp, r, u, f in zip(cam_pos, rights, ups, fwds):
        rlx += [cp[0], cp[0] + r[0], None]
        rly += [cp[1], cp[1] + r[1], None]
        rlz += [cp[2], cp[2] + r[2], None]

        ulx += [cp[0], cp[0] + u[0], None]
        uly += [cp[1], cp[1] + u[1], None]
        ulz += [cp[2], cp[2] + u[2], None]

        flx += [cp[0], cp[0] + f[0], None]
        fly += [cp[1], cp[1] + f[1], None]
        flz += [cp[2], cp[2] + f[2], None]

    points = go.Scatter3d(x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
                          mode="markers",
                          line=dict(color="orange"),
                          text=[f"cam {i}" for i in range(len(cam_pos))])

    up_lines = go.Scatter3d(x=ulx, y=uly, z=ulz, mode="lines",
                                line=dict(color="magenta", width=2),
                                hoverinfo="none")
    right_lines = go.Scatter3d(x=rlx, y=rly, z=rlz, mode="lines",
                                line=dict(color="green", width=2),
                                hoverinfo="none")
    forward_lines = go.Scatter3d(x=flx, y=fly, z=flz, mode="lines",
                                line=dict(color="orange", width=2),
                                hoverinfo="none")

    data = [points, up_lines, right_lines, forward_lines]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    camera = dict(up=dict(x=0, y=1, z=0))
    fig.update_layout(scene_camera=camera)

    return fig

def get_plotly_plane(tl, tr, br, bl, N=3, color='blue',
                     showscale=False, opacity=0.3, **kwargs):
    x = np.linspace(tl[0], tr[0], N)
    y = np.linspace(tl[1], bl[1], N)
    z = np.array([[tl[2], tr[2]], [bl[2], br[2]]])
    colorscale = [[0, color], [1, color]]

    return go.Surface(x=x, y=y, z=z, colorscale=colorscale,
                      showscale=showscale, opacity=opacity, **kwargs)

def plot_frustum(os, ds, near=1.0, far=6.0, color="red", fig=None, near_only=False):
    lx, ly, lz = [], [], []
    near_planes = []
    far_planes = []

    for o, d in zip(os, ds):
        nf = np.array(d) * far
        nd = np.array(d) * near
        for i in range(4):
            if near_only:
                line = nd
            else:
                line = nf
            lx += [o[0], o[0] + line[i][0], None]
            ly += [o[1], o[1] + line[i][1], None]
            lz += [o[2], o[2] + line[i][2], None]


        # plot near plane
        nx = [o[0] + nd[0][0], o[0] + nd[1][0],
              o[0] + nd[2][0], o[0] + nd[3][0],
              o[0] + nd[0][0], None]
        ny = [o[1] + nd[0][1], o[1] + nd[1][1],
              o[1] + nd[2][1], o[1] + nd[3][1],
              o[1] + nd[0][1], None]
        nz = [o[2] + nd[0][2], o[2] + nd[1][2],
              o[2] + nd[2][2], o[2] + nd[3][2],
              o[2] + nd[0][2], None]
        near_plane = go.Scatter3d(x=nx, y=ny, z=nz, mode="lines",
                                  surfaceaxis=0, opacity=0.3,
                                  line=dict(color="blue", width=2)
                                 )
        near_planes.append(near_plane)

        if not near_only:
            fx = [o[0] + nf[0][0], o[0] + nf[1][0],
                  o[0] + nf[2][0], o[0] + nf[3][0],
                  o[0] + nf[0][0]]
            fy = [o[1] + nf[0][1], o[1] + nf[1][1],
                  o[1] + nf[2][1], o[1] + nf[3][1],
                  o[1] + nf[0][1]]
            fz = [o[2] + nf[0][2], o[2] + nf[1][2],
                  o[2] + nf[2][2], o[2] + nf[3][2],
                  o[2] + nf[0][2]]
            far_plane = go.Scatter3d(x=fx, y=fy, z=fz, mode="lines",
                                      line=dict(color="green", width=2)
                                     )
            far_planes.append(far_plane)

    ray_lines = go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                             line=dict(color=color, width=2),
                             hoverinfo="none")

    data = [ray_lines, *near_planes, *far_planes]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    camera = dict(up=dict(x=0, y=1, z=0))
    fig.update_layout(scene_camera=camera)

    return fig

def plot_3d_bounding_box(vertices, fig=None):
    v0, v1, v2, v3, v4, v5, v6, v7 = vertices

    # v0->v1->v2->v3->v0
    lower_cap = np.concatenate([vertices[:4], vertices[:1]], axis=0)
    lower_cap = go.Scatter3d(x=lower_cap[:, 0], y=lower_cap[:, 1], z=lower_cap[:, 2],
                             mode="lines",
                             line=dict(color="red"))
    # v4->v5->v6->v7->v4
    upper_cap = np.concatenate([vertices[4:], vertices[4:5]], axis=0)
    upper_cap = go.Scatter3d(x=upper_cap[:, 0], y=upper_cap[:, 1], z=upper_cap[:, 2],
                             mode="lines",
                             line=dict(color="blue"))
    lx, ly, lz = [], [], []
    # v0->v4, v1->v5, v2->v6, v3->v7
    for i in range(4):
        lx += [vertices[i, 0], vertices[i+4, 0], None]
        ly += [vertices[i, 1], vertices[i+4, 1], None]
        lz += [vertices[i, 2], vertices[i+4, 2], None]

    lines = go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                         line=dict(color="black"))

    data = [lower_cap, upper_cap, lines]

    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)
    camera = dict(up=dict(x=0, y=1, z=0))
    fig.update_layout(scene_camera=camera)

    return fig

def plot_surface(coords, fig=None, amp=None):
    surface = go.Surface(x=coords[..., 0], y=coords[..., 1], z=coords[...,2],
                         surfacecolor=amp, colorscale="Spectral", opacity=0.5,
                         cmax=1., cmin=-1.)

    if fig is not None:
        for trace in fig['data']:
            trace['showlegend'] = False
        fig.add_trace(surface)
    else:
        fig = go.Figure(data=[surface])
    return fig

def get_surface_fig(kp, embedder, joint_names, n_sample=640, joint_idx=0,
                    freq=0, freq_skip=2, fig=None, sin=True,
                    x_offset=0, y_offset=0, z_offset=0, n_split=2,
                    x_only=False, y_only=False, z_only=False, v_range=2):
    #xy_plane = create_plane(z=kp[joint_idx,2]+z_offset, n_sample=n_sample)
    #yz_plane = create_plane(x=kp[joint_idx,0]+x_offset, n_sample=n_sample)
    #xz_plane = create_plane(y=kp[joint_idx,1]+y_offset, n_sample=n_sample)
    xy_plane = create_plane(z=z_offset, n_sample=n_sample)
    yz_plane = create_plane(x=x_offset, n_sample=n_sample)
    xz_plane = create_plane(y=y_offset, n_sample=n_sample)
    if x_only:
        planes = [yz_plane]
    elif y_only:
        planes = [xz_plane]
    elif z_only:
        planes = [xy_plane]
    else:
        planes = [xy_plane, yz_plane, xz_plane]

    embeds = []
    for plane in planes:

        dist = dist_to_joints(kp, plane.reshape(-1, 3))
        embed = [embedder.embed(torch.FloatTensor(dist[:, i:i+1])).numpy().reshape(n_sample, n_sample, -1)
                for i in range(dist.shape[1])]
        embeds.append(embed)
    embeds = np.array(embeds)

    embed_sins = embeds[..., 1::2][..., ::freq_skip]
    embed_coss = embeds[..., 2::2][..., ::freq_skip]
    p_fn = "Sin" if sin else "Cos"
    #title = f"{p_fn} - Joint: {joint_names[joint_idx]} - Freq: 2^{freq_skip * freq}"
    fig = plot_skeleton3d(kp, fig=fig)
    embeds = embed_sins if sin else embed_coss
    for i, (p, e) in enumerate(zip(planes, embeds)):
        fig = plot_surface(p, fig=fig, amp=e[joint_idx, ..., freq])

    fig.update_layout(#title_text=title,
                      scene=dict(
                          xaxis = dict(range=[-v_range, v_range]),
                          yaxis = dict(range=[-v_range, v_range]),
                          zaxis = dict(range=[-v_range, v_range]),

                      ),
                      scene_aspectmode='cube',
                      margin=dict(r=10, l=10, b=10, t=10),)
    return fig

def generate_sweep_vid(embedder, path, cam_pose, kp, axis="x",
                       joint_idx=SMPLSkeleton.joint_names.index("right_hand"),
                       offset=5, n_sample=80, freq=0, ext_scale=0.001, H=512, W=512, focal=100):
    import imageio
    from ray_utils import get_corner_rays
    offsets = np.linspace(-offset, offset, 28)
    imgs = []

    x_only = True if axis == "x" else False
    y_only = True if axis == "y" else False
    z_only = True if axis == "z" else False

    #x_only, y_only, z_only = False, False, False

    for offset in offsets:
        fig = plot_cameras(viewmats=swap_mat(cam_pose))
        #rays_o, rays_d = get_corner_rays(H, W, focal, cam_pose)
        #fig = plot_frustum(np.array(rays_o), np.array(rays_d),
        #                  near=0.5, far=4.2,
        #                  fig=fig, near_only=False)
        #fig = plot_bounding_cylinder(kp, ext_scale=ext_scale, fig=fig,
        #                             extend_mm=500)

        fig = get_surface_fig(kp, embedder, joint_names=SMPLSkeleton.joint_names, n_sample=n_sample, fig=fig,
                              joint_idx=joint_idx, freq=freq, z_offset=offset, y_offset=offset, x_offset=offset,
                              n_split=4, x_only=x_only, y_only=y_only, z_only=z_only, v_range=np.max(offsets)+0.3)
        img = byte2array(fig.to_image(format="png"))
        imgs.append(img)
    imgs = np.array(imgs)
    imageio.mimwrite(path, (imgs).astype(np.uint8), fps=14, quality=8)
    return imgs


def plot_joint_axis(kp, l2ws=None, fig=None, scale=0.1):

    rights = np.array([1., 0., 0., 1. / scale]) * scale
    ups    = np.array([0., 1., 0., 1. / scale]) * scale
    fwds   = np.array([0., 0., 1., 1. / scale]) * scale

    rights = np.array([l2w @ rights for l2w in l2ws])
    ups = np.array([l2w @ ups for l2w in l2ws])
    fwds = np.array([l2w @ fwds for l2w in l2ws])

    rlx, rly, rlz = [], [], []
    ulx, uly, ulz = [], [], []
    flx, fly, flz = [], [], []

    for k, r, u, f in zip(kp, rights, ups, fwds):
        #import pdb; pdb.set_trace()
        rlx += [k[0], r[0], None]
        rly += [k[1], r[1], None]
        rlz += [k[2], r[2], None]

        ulx += [k[0], u[0], None]
        uly += [k[1], u[1], None]
        ulz += [k[2], u[2], None]

        flx += [k[0], f[0], None]
        fly += [k[1], f[1], None]
        flz += [k[2], f[2], None]

    up_lines = go.Scatter3d(x=ulx, y=uly, z=ulz, mode="lines",
                            line=dict(color="magenta", width=2),
                            hoverinfo="none")
    right_lines = go.Scatter3d(x=rlx, y=rly, z=rlz, mode="lines",
                               line=dict(color="green", width=2),
                               hoverinfo="none")
    forward_lines = go.Scatter3d(x=flx, y=fly, z=flz, mode="lines",
                                 line=dict(color="orange", width=2),
                                 hoverinfo="none")
    data = [up_lines, right_lines, forward_lines]
    if fig is not None:
        for d in data:
            fig.add_trace(d)
    else:
        fig = go.Figure(data=data)
    return fig

#################################
#          Draw Helpers         #
#################################
def draw_skeletons_3d(imgs, skels, c2ws, H, W, focals, centers=None,
                      width=3, flip=False, skel_type=SMPLSkeleton):
    skels_2d = skeleton3d_to_2d(skels, c2ws, H, W, focals, centers)
    return draw_skeletons(imgs, skels_2d, skel_type, width, flip)

def draw_skeletons(imgs, skels, skel_type=SMPLSkeleton, width=3, flip=False):
    skel_imgs = []
    for img, skel in zip(imgs, skels):
        skel_img = draw_skeleton2d(img, skel, skel_type, width, flip)
        skel_imgs.append(skel_img)
    return np.array(skel_imgs)

def draw_skeleton2d(img, skel, skel_type=None, width=3, flip=False, radius=4, joint_color=(0, 0, 0)):
    if skel_type is None:
        skel_type = get_skeleton_type(skel)

    joint_names = skel_type.joint_names
    joint_tree = skel_type.joint_trees

    img = img.copy()
    C = img.shape[-1]
    if C == 4:
        joint_color = joint_color + (255,)

    for i, j in enumerate(skel):
        name = joint_names[i]
        parent = skel[joint_tree[i]]
        offset = parent - j

        #""" 
        if "left" in name:
            color = (100, 149, 237) if not flip else (3, 31, 255)
            joint_color = (255, 0, 0)
            #color = (0, 0, 255)
        elif "right" in name:
            #color = (255, 0, 0)
            color = (255, 0, 0) if not flip else (249, 33, 36)
            joint_color = (0, 0, 255)
        else:
            color = (46, 204, 13) if not flip else (0, 0, 0)
            joint_color = (0, 255, 0)
        #"""
        #color = (0,0,0)
        if C == 4:
            color = color + (255,)
            joint_color = joint_color + (255,)
        cv2.line(img, (int(round(j[0])), int(round(j[1]))),
                      (int(round(j[0]+offset[0])), int(round(j[1]+offset[1]))),
                 color, thickness=width,
                 lineType=cv2.LINE_AA)
        
        """ 
        cv2.circle(img, (int(round(j[0])), int(round(j[1]))), radius, joint_color, -1,
                   lineType=cv2.LINE_AA)

        cv2.circle(img, (int(round(j[0]+offset[0])), int(round(j[1]+offset[1]))), radius, joint_color, -1,
                   lineType=cv2.LINE_AA)
        """
    return img

def draw_points2d(img, pts, color=(0, 255, 0), radius=3):
    for pt in pts:
        cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), radius, color, -1,
                    lineType=cv2.LINE_AA)
    return img

def draw_opt_poses(c2ws, hwf, anchor_kps, opt_kps, gt_kps=None,
                   base_imgs=None, centers=None, res=0.1, skel_type=SMPLSkeleton,
                   thickness=4, marker_size=30):
    '''
    Plot both anchor and optimized keypoints
    '''
    root_id = skel_type.root_id
    h, w, focals = hwf

    anchors_2d = skeleton3d_to_2d(anchor_kps, c2ws, h, w, focals, centers)
    opts_2d = skeleton3d_to_2d(opt_kps, c2ws, h, w, focals, centers)

    shift_h = anchors_2d[:, root_id, 0] - opts_2d[:, root_id, 0]
    shift_w = anchors_2d[:, root_id, 1] - opts_2d[:, root_id, 1]
    anchors_2d[..., 0] -= shift_h[:, None]
    anchors_2d[..., 1] -= shift_w[:, None]

    # Step 2: draw the image
    imgs = []
    for i, (anchor_2d, opt_2d) in enumerate(zip(anchors_2d, opts_2d)):
        img = base_imgs[i]
        img = draw_markers(opt_2d, img=img, H=h, W=w, res=res, thickness=thickness, marker_size=marker_size, color=(255, 149, 0, 32))
        img = draw_markers(anchor_2d, img=img, H=h, W=w, res=res, thickness=thickness,
                           marker_size=marker_size, color=(0, 200, 0, 32),
                           marker_type=cv2.MARKER_TILTED_CROSS)
        imgs.append(img)
    return np.array(imgs)

def draw_markers(pts, img=None, H=1000, W=1000, res=0.1, marker_type=cv2.MARKER_CROSS,
                 thickness=3, color=(255, 0, 0), marker_size=5):
    '''
    Draw 2d points on the input image
    '''
    if img is None:
        img = np.ones((int(H * res), int(W * res), 3), dtype=np.uint8) * 255
    elif res != 1.0:
        img = cv2.resize(img, (int(W * res), int(H * res)))
    for p in pts:
        img = cv2.drawMarker(img, (int(p[0] * res), int(p[1] * res)),
                             color=color, markerType=marker_type,
                             thickness=thickness, markerSize=marker_size)
    return img

