import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from hydra.utils import instantiate
from core.networks.danbo import DANBO
from core.networks.anerf import NORMAL_TRANSFORM
from core.positional_enc import PositionalEncoding

from core.networks.embedding import Optcodes
from core.networks.anerf import merge_encodings
from core.utils.skeleton_utils import Skeleton
from einops import rearrange

from omegaconf import DictConfig
from typing import Mapping, Any, List, Optional


class NPC(DANBO):

    def __init__(
        self,
        *args, 
        pts_config: DictConfig = DictConfig({}),
        deform_config: DictConfig = DictConfig({}),
        constraint_pts: int = 100,
        sdf_B_init: float = 0.1,
        use_global_view: bool = False,
        add_film: bool = False,
        use_f_r: bool = True,
        full_sdf: bool = False,
        **kwargs,
    ):
        self.constraint_pts = constraint_pts
        self.sdf_B_init = sdf_B_init
        self.deform_config = deform_config
        self.pts_config = pts_config
        self.use_global_view = use_global_view
        self.add_film = add_film
        self.use_f_r = use_f_r
        self.full_sdf = full_sdf

        super(NPC, self).__init__(*args, **kwargs)
        self.init_pts(pts_config, deform_config)
        assert self.pred_sdf

    @property
    def vnet_input(self):
        if self.use_framecodes:
            return self.input_ch_view + self.framecode_ch
        return self.input_ch_view
    
    def init_skeleton(self, skel_type: Skeleton, rest_pose: np.ndarray):
        super(NPC, self).init_skeleton(skel_type, rest_pose)
        self.N_joints = len(self.skel_type.joint_names)
        self.part_ignore = self.skel_profile['rigid_ignore']
        self.rigid_idxs = self.skel_profile['rigid_idxs'].copy()

    def init_embedder(
        self,
        *args,
        view_posi_enc: DictConfig,
        voxel_posi_enc: DictConfig,
        **kwargs,
    ):
        super(NPC, self).init_embedder(
            *args,
            view_posi_enc=view_posi_enc, 
            voxel_posi_enc=voxel_posi_enc, 
            **kwargs
        )
        pts_feat_dims = self.pts_config.feat_config.n_out
        bone_feat_dims = self.pts_config.bone_config.n_out
        pose_feat_dims = self.deform_config.n_pose_feat
        self.voxel_posi_enc = instantiate(voxel_posi_enc, input_dims=pts_feat_dims)
        self.input_ch = self.voxel_posi_enc.dims + bone_feat_dims + pose_feat_dims

        # plus one for bone-to-surface vector projeciton
        if self.use_f_r:
            self.input_ch += 1

        if self.add_film:
            self.input_ch = self.input_ch + 32
        
        view_posi_enc_inputs = 1
        self.view_posi_enc = instantiate(view_posi_enc, input_dims=view_posi_enc_inputs, dist_inputs=True)
        self.input_ch_view = self.view_posi_enc.dims

        if self.use_global_view:
            self.global_view_posi_enc = PositionalEncoding(3, num_freqs=4)
            self.input_ch_view += self.global_view_posi_enc.dims
        
        if self.full_sdf:
            self.input_ch_view += 3


    def init_density_net(self):

        W = self.W

        self.mlp = nn.Sequential(
            nn.Linear(self.dnet_input, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
        )

        self.sigma_linear = nn.Linear(W, 1)

        if self.pred_sdf:
            # TODO: jump here
            self.register_parameter(
                'B',
                nn.Parameter(torch.tensor(self.sdf_B_init), requires_grad=True)
            )

    def init_radiance_net(self):

        W, view_W = self.W, self.view_W
        self.feature_linear = nn.Linear(W, W)
        self.color_pred = nn.Sequential(
            nn.Linear(W + self.vnet_input, view_W),
            nn.ReLU(inplace=True),
            nn.Linear(view_W, 3)
        )
        if self.use_framecodes:
            self.framecodes = Optcodes(self.n_framecodes, self.framecode_ch)

    def init_pts(self, pts_config: DictConfig, deform_config: DictConfig):
        """ Initialize point cloud network
        """
        N_joints = len(self.rigid_idxs)
        
        bone_centers = self.rest_pose[None, self.rigid_idxs]

        deform_net = instantiate(
            deform_config,
            per_node_input=self.input_ch_graph,
            skel_profile=self.skel_profile,
            output_ch=None,
            pts_per_volume=pts_config.pts_per_volume,
        )
        
        self.pc = instantiate(
            pts_config,
            deform_net=deform_net,
            pts_embedder=self.pts_embedder,
            pose_embedder=self.pose_embedder,
            skel_profile=self.skel_profile,
            bone_centers=bone_centers,
            _recursive_=False,
        )

    def init_raycast(
        self,
        raycast: DictConfig,
        **kwargs,
    ):
        self.raycast = instantiate(
            raycast, 
            **kwargs, 
            vol_scale_fn=self.get_axis_scale,
            #anchor_fn=self.get_T_pose_anchor,
            rest_pose=self.rest_pose,
            skel_type=self.skel_type,
            skel_profile=self.skel_profile,
            rest_heads=self.rest_heads,
            rigid_idxs=self.rigid_idxs,
            _recursive_=False
        )

    def reset_rgb_head(self):
        # TODO: is there a better way to find these network?
        for l in self.color_pred:
            if isinstance(l, nn.Linear):
                l.reset_parameters()
        self.feature_linear.reset_parameters()
        #self.rgb_linear.reset_parameters()

    def forward_rays(self, batch: Mapping[str, Any], render_normal: bool = False, pose_opt: bool = False, **kwargs):

        if 'rays_o' not in batch and 'rays_d' not in batch:
            raise NotImplementedError('Rays are not provided as input. '
                                      'For rendering image with automatically detected rays, call render(...)')
        
        # Step 1. cast ray
        bgs = batch.get('bgs', None) # background color
        sample_info = self.raycast(batch)
        pts, z_vals = sample_info['pts'], sample_info['z_vals']

        # Step 2. model evaluation
        # TODO: do we need a get_nerf_inputs function?
        network_inputs = self.get_network_inputs(batch, pts, z_vals)
        if pose_opt and self.pose_opt is not None:
            # TODO: is this a good way?
            network_inputs = self.pose_opt(
                network_inputs=network_inputs,
                kp3d=network_inputs['kp3d'],
                bones=network_inputs['bones'],
                kp_idxs=network_inputs['real_kp_idx'],
                N_unique=network_inputs['N_unique'],
            )

        raw, encoded = self.evaluate_pts(
            network_inputs, 
            coarse=True,
            render_normal=render_normal,
        )
        
        if raw is None:
            return self._empty_outputs(batch)

        bgnet_ret = None
        if self.training and self.bkgd_net is not None:
            bgnet_ret = self.bkgd_net(batch)
            bgs = bgnet_ret['bgs']

        # Step 3. coarse rendering
        rendered = self.raw2outputs(
            raw, 
            z_vals, 
            batch['rays_d'],
            bgs=bgs,
            encoded=encoded, 
            batch=batch,
        )

        if self.raycast.N_importance == 0:
            return self.collect_outputs(rendered, None, encoded, None)

        # Step 4. importance sampling (if importance sampling enabled)
        rendered_coarse = rendered
        encoded_coarse = encoded

        weights = rendered_coarse['weights']
        is_sample_info = self.raycast(
            batch, 
            pts=pts, 
            z_vals=z_vals,
            weights=weights,
            importance=True,
        )
        pts_is = is_sample_info['pts'] # only importance samples
        z_vals = is_sample_info['z_vals'] # include both coarse and importance samples

        # Step 5. model evaluation (if importance sampling enabled)
        network_inputs = self.get_network_inputs(batch, pts_is, is_sample_info['z_vals_is'])
        raw_is, encoded_is = self.evaluate_pts(
            network_inputs, 
            encoded_coarse=encoded_coarse,
            render_normal=render_normal,
        )

        # Step 6. merge coarse and importance prediction for rendering
        sorted_idxs = is_sample_info['sorted_idxs']
        N_rays = len(batch['rays_o'])
        N_total_samples = pts.shape[1] + pts_is.shape[1]
        encoded_is = merge_encodings(encoded_coarse, encoded_is, sorted_idxs, N_rays, N_total_samples)

        # Step 7. fine rendering
        if raw_is is not None:
            raw = merge_encodings({'raw': raw}, {'raw': raw_is}, sorted_idxs, N_rays, N_total_samples)['raw']
            rendered = self.raw2outputs(
                raw, 
                z_vals, 
                batch['rays_d'],
                bgs=bgs if self.light is None else None,
                batch=batch,
                encoded=encoded_is, 
            )

            if self.light is not None:

                pts_merged = merge_encodings({'pts': pts}, {'pts': pts_is}, sorted_idxs, N_rays, N_total_samples)['pts']

                rendered, extras = self.light.compute_light(
                    rendered,
                    rays_o=batch['rays_o'],
                    rays_d=batch['rays_d'],
                    w2ls=batch['skts'],
                    visibility=self.visibility,
                    N_unique=batch['N_unique'],
                    density=raw[..., 3:4],
                    pts=pts_merged,
                    bgs=bgs,
                    valid_idxs=encoded_is.get('valid_idxs', None),
                    global_iter=batch.get('global_iter', None),
                )
                encoded_is.update(**extras)
        else:
            ref = bgs if bgs is not None else torch.zeros_like(rendered_coarse['rgb_map'])
            if self.light is not None:
                #rendered_coarse.update(**self._empty_light_outputs(ref))
                rendered_coarse.update(**self._empty_light_outputs(batch, ref, ref))
            if self.pred_sdf and self.full_sdf:
                rendered_coarse.update(**self._empty_sdf_outputs(ref))
            rendered = rendered_coarse

        if bgnet_ret is not None:
            rendered.update(**bgnet_ret)

        return self.collect_outputs(rendered, rendered_coarse, encoded_is, encoded_coarse)

    def evaluate_pts(
        self, 
        inputs: Mapping[str, Any],
        geometry_only: bool = False, 
        geom_extra_rets: List[str] = [], 
        encoded_coarse: Optional[Mapping[str, torch.Tensor]] = None, 
        coarse: bool = False, 
        render_normal: bool = False,
        **kwargs
    ):

        if self.full_sdf:
            inputs['pts'].requires_grad = True

        # Step 1. encode all pts feature
        density_inputs, encoded_pts = self.encode_pts(
            inputs, 
            encoded_coarse=encoded_coarse,
        )
        if self.training and coarse and self.constraint_pts > 0 and encoded_pts is not None:
            pc_constraints = self.get_pc_constraints(
                inputs=inputs,
                pc_info=encoded_pts['pc_info'],
            )
            encoded_pts.update(pc_constraints=pc_constraints)

        
        if density_inputs is None:
            # terminate because not valid points 
            return None, None
        
        # Step 2. density prediction
        sigma, density_feature = self.inference_sigma(density_inputs)

        if self.pred_sdf and self.full_sdf:
            # surface_gradient = torch.autograd.grad(
            #     outputs=sigma,
            #     inputs=inputs['pts'],
            #     grad_outputs=torch.ones_like(sigma),
            #     create_graph=self.training,
            # )[0]
            surface_gradient = self.compute_surface_gradient(
                inputs['pts'],
                sigma,
                create_graph=self.training,
                global_iter=inputs.get('global_iter', None),
            )

            encoded_pts.update(surface_gradient=surface_gradient)


        density = self.to_density(sigma)

        if geometry_only:
            geom_ret = {'density': density, 'sigma': sigma}
            if 'valid_idxs' in encoded_pts:
                geom_ret['valid_idxs'] = encoded_pts['valid_idxs']
            if 'surface_gradient' in encoded_pts:
                geom_ret['surface_gradient'] = encoded_pts['surface_gradient']
            for k in geom_extra_rets:
                if k in encoded_pts:
                    geom_ret[k] = encoded_pts[k]
            return None, geom_ret

        # Step 3. encode all ray feature
        view_inputs, encoded_views = self.encode_views(
            inputs, 
            refs=encoded_pts['pts_t'],
            encoded_pts=encoded_pts
        )
        
        # Step 4. rgb prediction
        rgb = self.inference_rgb(view_inputs, density_feature)

        # Step 5: create final outputs
        shape = inputs['pts'].shape[:2] # (N_rays, N_samples)
        output_list = [rgb, density]
        outputs = self.fill_prediction(torch.cat(output_list, dim=-1), shape, encoded_pts)

        return outputs, self.collect_encoded(encoded_pts, encoded_views)
    
    def encode_pts(
        self, 
        inputs: Mapping[str, Any], 
        pc_info: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded_coarse: Optional[Mapping[str, torch.Tensor]] = None, 
        is_pc: bool = False
    ):

        q_w = inputs['pts']
        N_rays, N_samples = q_w.shape[:2]
        N_joints = len(self.rigid_idxs)
        N_unique = inputs['N_unique']
        rays_per_pose = N_rays // N_unique
        unique_idxs = torch.arange(N_rays) // rays_per_pose
        unique_idxs = unique_idxs.reshape(N_rays, 1).expand(-1, N_samples)
        unique_idxs = unique_idxs.reshape(-1)

        # get pts_t (3d points in local space)
        encoded_pts = self.pts_embedder(**inputs, rigid_idxs=self.rigid_idxs)
        encoded_pose = self.pose_embedder(**inputs)

        if self.training and self.constraint_pts > 0:
            encoded_pose['pose'].requires_grad = True
        pose_pe = self.pose_posi_enc(encoded_pose['pose'])[0]

        q_b = encoded_pts['pts_t']

        # move points to scaled per-joint space to check validity
        vol_scale = self.get_axis_scale(rigid_idxs=self.rigid_idxs)
        q_bs = q_b / vol_scale.reshape(1, 1, N_joints ,3)

        invalid = ((q_bs.abs() > 1).sum(-1) > 0).float()

        valid = (1 - invalid).reshape(N_rays * N_samples, N_joints)

        # all point clouds are valid
        valid_idxs = torch.arange(len(valid)) if is_pc else torch.where(valid.sum(-1) > 0)[0]
        unique_idxs = unique_idxs[valid_idxs]
        N_valid = len(valid_idxs)

        if N_valid == 0:
            return None, None

        # step 1. find k nearest volumes
        q_b = q_b.reshape(N_rays * N_samples, N_joints, 3)
        if not is_pc:
            q_b = q_b[valid_idxs]
            valid = valid[valid_idxs]

        q_w = q_w.reshape(N_rays * N_samples, 3)[valid_idxs]

        N_unique = inputs['N_unique']

        # N_expand = N_rays // N_unique, ray_idxs = valid_idxs // N_samples, pose_idxs = ray_idxs // N_expand
        # simplified to the line below
        pose_idxs = (valid_idxs * N_unique).div(N_samples * N_rays, rounding_mode='trunc')

        # get view direction
        vw = inputs['rays_d'][:, None].expand(-1, N_samples, -1).reshape(-1, 3)[valid_idxs]
        vd = self.view_embedder(
            **inputs, 
            valid_idxs=valid_idxs, 
            rigid_idxs=self.rigid_idxs,
            refs=encoded_pts['pts_t'],
        )['d']
        
        # query point clouds to get all the rest information
        # need q_v to get bone relative feature
        # but can use q_w for finding NN points
        inputs.update(
            q_w=q_w, 
            q_b=q_b, 
            vd=vd,
            vw=vw,
            valid_q_idxs=valid_idxs, 
            valid_pose_idxs=pose_idxs,
            pose_pe=pose_pe
        )
        encoded_q = self.pc.query_feature(
            inputs,
            pc_info=pc_info,
            is_pc=is_pc,
            need_hessian=self.training and self.full_sdf,
        )

        f_p_s = encoded_q['f_p_s']
        f_theta = encoded_q['f_theta']
        f_d = encoded_q['f_d']
        f_r = encoded_q['f_r']
        f_p_s = self.voxel_posi_enc(f_p_s, weights=encoded_q['a_sum'])[0]

        
        # TODO: omit further triming points for now. 
        feat_list = [f_p_s, f_theta, f_d]
        if self.use_f_r:
            feat_list.append(f_r)
        density_inputs = torch.cat(feat_list, dim=-1)
        if self.add_film:
            pc_info = encoded_q['pc_info']
            film = pc_info['t_film'][unique_idxs]
            density_inputs = torch.cat([density_inputs, film], dim=-1)

        encoded_pts.update(**encoded_q, valid_idxs=valid_idxs, unique_idxs=unique_idxs)
        return density_inputs, encoded_pts
    
    def get_pc_constraints(
        self,
        inputs: Mapping[str, Any],
        pc_info: Mapping[str, torch.Tensor],
        eik_noise: float = 0.03,
    ):

        p_w = pc_info['p_w']
        # remove redundant info -> keep only the unique poses
        skip = len(inputs['skts']) // inputs['N_unique']
        skts = inputs['skts'][::skip]
        bones = inputs['bones'][::skip]
        kp3d = inputs['kp3d'][::skip]
        rays_o = inputs['rays_o'][::skip]
        rays_d = inputs['rays_d'][::skip]

        N_graphs, N_joints, pts_per_volume = p_w.shape[:3]

        constraint_idxs = np.stack([sorted(np.random.choice(
            pts_per_volume,
            self.constraint_pts,
            replace=False,
        )) for _ in range(N_joints)])
        constraint_idxs = constraint_idxs + np.arange(N_joints)[:, None] * pts_per_volume

        # note: after indexing, the shape is (N_graphs, N_joints, constraint_pts, 3)
        p_cts = rearrange(p_w, 'g j p d -> g (j p) d')[:, constraint_idxs]
        # we reshape to (g, *, d) as if we have sample points (N_rays [N_graphs], N_samples [N_joints x N_constraint_pts])
        p_cts = rearrange(p_cts, 'g j p d -> g (j p) d', j=N_joints)

        encode_inputs = {
            'pts': p_cts, 
            'rays_o': rays_o,
            'rays_d': rays_d,
            'skts': skts,
            'kp3d': kp3d,
            'bones': bones,
            'N_unique': N_graphs,
        }

        density_inputs, encoded = self.encode_pts(encode_inputs, pc_info=pc_info, is_pc=True)
        # note: sigma is sdf
        sigma, pc_feat = self.inference_sigma(density_inputs)
        sigma = rearrange(sigma, '(g j p) d-> g j (p d)', g=N_graphs, j=N_joints)

        # eikonal / surface gradient
        noise = (torch.rand_like(p_cts) * 2. - 1.) * eik_noise
        p_ncts = p_cts + noise
        encode_inputs.update(pts=p_ncts)

        density_inputs, encoded = self.encode_pts(encode_inputs, pc_info=pc_info, is_pc=True)
        # note: sigma is sdf at perturbed locations
        sigma_n, pc_feat = self.inference_sigma(density_inputs)
        sigma_n = rearrange(sigma_n, '(g j p) d-> g j (p d)', g=N_graphs, j=N_joints)

        pc_grad = torch.autograd.grad(
            outputs=sigma_n,
            inputs=[p_ncts],
            grad_outputs=torch.ones_like(sigma_n),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        ret = {
            'pc_sigma': sigma,
            'pc_grad': pc_grad,
        }

        return ret
    
    def encode_views(self, inputs: Mapping[str, Any], refs: torch.Tensor, encoded_pts: Mapping[str, torch.Tensor]):

        valid_idxs = encoded_pts['valid_idxs']
        f_v = encoded_pts['f_v']
        a = encoded_pts.get('a_sum', None)
        view_inputs = self.view_posi_enc(f_v, a)[0]

        if self.full_sdf:
            normal = F.normalize(encoded_pts['surface_gradient'], p=2, dim=-1)
            normal = rearrange(normal, 'r s c -> (r s) c')[valid_idxs]
            view_inputs = torch.cat([view_inputs, normal], dim=-1)

        if self.use_framecodes:
            N_rays, N_samples = refs.shape[:2]
            # expand from (N_rays, ...) to (N_rays, N_samples, ...)
            cam_idxs = inputs['cam_idxs']
            cam_idxs = cam_idxs.reshape(N_rays, 1, -1).expand(-1, N_samples, -1)
            framecodes = self.framecodes(cam_idxs.reshape(N_rays * N_samples, -1))
            framecodes = framecodes[valid_idxs]

            view_inputs = torch.cat([view_inputs, framecodes], dim=-1)

        if self.use_global_view:
            N_rays, N_samples = refs.shape[:2]
            rays_d = inputs['rays_d'].reshape(N_rays, 1, 3).expand(-1, N_samples, 3)
            rays_d = rays_d.reshape(N_rays * N_samples, 3)[valid_idxs]
            rays_d = self.global_view_posi_enc(rays_d)[0]
            view_inputs = torch.cat([view_inputs, rays_d], dim=-1)

        return view_inputs, {}

    def get_axis_scale(self, rigid_idxs: Optional[torch.Tensor] = None):
        return self.pc.get_axis_scale(rigid_idxs)

    def inference_sigma(self, density_inputs: torch.Tensor):
        x = self.mlp(density_inputs)
        sigma = self.sigma_linear(x)
        return sigma, x

    def inference_rgb(self, view_inputs: torch.Tensor, density_feature: torch.Tensor, rgb_eps: float = 0.001):

        density_feature = self.feature_linear(density_feature)
        view_inputs = torch.cat([view_inputs, density_feature], dim=-1)
        raw_rgb = self.color_pred(view_inputs)
        rgb = torch.sigmoid(raw_rgb) * (1 + 2 * rgb_eps) - rgb_eps
        return rgb
    
    def collect_encoded(self, encoded_pts: Mapping[str, torch.Tensor], encoded_views: Mapping[str, torch.Tensor]):
        ret = super(NPC, self).collect_encoded(encoded_pts, encoded_views)
        ret['pc_info'] = encoded_pts['pc_info']
        if 'pc_constraints' in encoded_pts:
            ret['pc_constraints'] = encoded_pts['pc_constraints']
        return ret

    def collect_outputs(
        self, 
        ret: Mapping[str, torch.Tensor],
        ret0: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded0: Optional[Mapping[str, torch.Tensor]] = None
    ):
        ret = super(NPC, self).collect_outputs(ret, ret0, encoded, encoded0)
        if encoded is not None:
            pc_info = encoded['pc_info']
            pc_constraints = encoded.get('pc_constraints', None)
            to_collect = ['dp', 'dp_uc', 'p_w', 'p_w_uc', 'pc_sigma', 'pc_grad']
            for k in to_collect:
                if k in pc_info: 
                    ret[k] = pc_info[k]
                if pc_constraints is not None and k in pc_constraints:
                    ret[k] = pc_constraints[k]
        if self.training:
            ret.update(nb_idxs=self.pc.nb_idxs, nb_diffs=self.pc.nb_diffs, pts_beta=self.pc.get_pts_beta())
        
        return ret

    def raw2sdfoutputs(
        self, 
        raw: torch.Tensor, 
        z_vals: torch.Tensor, 
        rays_d: torch.Tensor, 
        bgs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """ Sligtly different from usual case
        """

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        # following Lior's paper
        density = raw[..., 3]
        if not self.training:
            density[density < 2.] = 0

        # note: this is in logspace, originally we do exp(-density*dists)
        free_energy = density * dists
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)  # shift one step

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        # sum_{i=1 to N samples} prob_of_already_hit_particles * alpha_for_i * color_for_i
        # C(r) = sum [T_i * (1 - exp(-sigma_i * delta_i)) * c_i] = sum [T_i * alpha_i * c_i]
        # alpha_i = 1 - exp(-sigma_i * delta_i)
        # T_i = exp(sum_{j=1 to i-1} -sigma_j * delta_j) = torch.cumprod(1 - alpha_i)
        # standard NeRF
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        T = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * T # probability of the ray hits something here
        #weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

        # get rgb (as usual)
        rgb = raw[..., :3] # [N_rays, N_samples, 3]

        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        eps = 1e-10
        inv_eps = 1. / eps
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = (weights.sum(dim=-1) / depth_map).double()
        disp_map = torch.where(
            (disp_map > 0) & (disp_map < inv_eps) & (weights.sum(dim=-1) > eps), disp_map, inv_eps
        ).float()

        acc_map = torch.minimum(torch.sum(weights, -1), torch.tensor(1.))

        ret = {
            'disp_map': disp_map, 
            'acc_map': acc_map,
            'weights': weights, 
            'alpha': alpha,
            'depth_map': depth_map,
        }

        encoded = kwargs['encoded']
        surface_gradient = encoded.get('surface_gradient', None)
        if surface_gradient is not None:
            normal_map = torch.mean(weights[...,None] * encoded['surface_gradient'], -2)  # [N_rays, 3]
            normal_map = F.normalize(normal_map, p=2, dim=-1)
            normal_map[depth_map < 0.5] = 1

            normal_map_viz = (((NORMAL_TRANSFORM@normal_map.detach().T).T*0.5)+0.5)
            normal_map_viz[depth_map < 0.5] = 1

            ret.update(normal_map=normal_map, normal_map_viz=normal_map_viz)

        if bgs is not None:
            rgb_map = rgb_map + (1. - acc_map)[..., None] * bgs
        ret.update(rgb_map=rgb_map)

        return ret

    def forward_render(self, *args, **kwargs):
        with torch.set_grad_enabled(self.full_sdf):
            return super().forward_render(*args, **kwargs)

    def get_summaries(self, *args, **kwargs):
        return self.pc.get_summaries(*args, **kwargs)

    def compute_surface_gradient(
        self,
        pts: torch.Tensor,
        outputs: torch.Tensor,
        create_graph: bool = False,
        global_iter: Optional[int] = None,
    ):
        #compute_grad = False
        detach_grad = False
        # can do this because RGB net doesn't use normal
        if self.training and self.lazy_sdf is not None and global_iter is not None:
            compute_grad = eikonal_iter = (global_iter % self.lazy_sdf) == 0
            detach_grad = global_iter is not None and not eikonal_iter
        else:
            compute_grad = True
            detach_grad = not self.training
        compute_grad = True

        surface_gradient = torch.zeros_like(pts)
        if compute_grad:
            surface_gradient = torch.autograd.grad(
                outputs=outputs,
                inputs=pts,
                grad_outputs=torch.ones_like(outputs),
                create_graph=create_graph,
            )[0]


        if detach_grad:
            surface_gradient = surface_gradient.detach()

        return surface_gradient