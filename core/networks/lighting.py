import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Any, Optional, Dict
from .gaussian_light import GaussianVisibility
import cv2

def safe_arccos(a):
    clipped = torch.clip(a, -1.+1e-8, 1.-1e-8)
    return torch.arccos(clipped)

def direction_to_equirectangular_range(dir, range = torch.Tensor([2*np.pi, -np.pi, np.pi, 0])):
    dir = F.normalize(dir, dim=1)[..., None]
    us = (torch.arctan2(dir[:,1], dir[:,0]) - range[1]) / range[0]
    vs = (safe_arccos(dir[:,2]) - range[3]) / range[2]
    return us.detach().cpu().numpy(), vs.detach().cpu().numpy()

def equirectangular_range_to_direction(u, v, range = np.array([2*np.pi, -np.pi, np.pi, 0])):
    phi = range[0] * u + range[1]
    theta = range[2] * v + range[3]
    sin_theta = np.sin(theta)
    return np.array([sin_theta * np.cos(phi), sin_theta*np.sin(phi), np.cos(theta)])

def equirectangular_range_to_direction_torch(u, v, range = torch.Tensor([2*np.pi, -np.pi, np.pi, 0])):
    phi = range[0] * u + range[1]
    theta = range[2] * v + range[3]
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta*torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x,y,z])

class DirectionalLight(nn.Module):

    def __init__(
        self,
        direction: List = [[0.5773, 0.5773, 0.5773]],
        color: List = [[1.0, 1.0, 1.0]],
        ambient: List = [0.1],
        floor_point: List = [0,0,0],
        floor_normal: List = [0,-1,0],
        light_intensity: float = 1.5,
        env: Optional[torch.Tensor] = None,
        optimize_ambient: bool = True,
        optimize_direction: bool = True,
        use_diffuse: bool = True,
        use_gaussians: bool = True,
        clip_irradiance: bool = True,
        stop_opt: Optional[int] = None,
    ):
        super().__init__()
        self.optimize_ambient = optimize_ambient
        self.optimize_direction = optimize_direction
        self.light_intensity = light_intensity
        self.use_diffuse = use_diffuse
        self.use_gaussians = use_gaussians
        self.clip_irradiance = clip_irradiance
        self.stop_opt = stop_opt

        if env is None:
            env = torch.Tensor([
                [0.34,0.84,1.05],
                [-1.63,-1.19,-1.0],
                [1,0.73,0.6],
                [0.89,0.61,0.49],
                [0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ])
        
        direction = torch.Tensor(direction)
        self.register_parameter(
            'direction',
            nn.Parameter(direction, requires_grad=optimize_direction)
        )

        color = torch.Tensor(color)
        self.register_buffer('color', color, persistent=False)

        ambient = torch.Tensor(ambient)
        self.register_parameter(
            'ambient',
            nn.Parameter(ambient, requires_grad=optimize_ambient)
        )

        floor_point = torch.Tensor(floor_point)
        self.register_buffer('floor_point', floor_point, persistent=False)

        floor_normal = torch.Tensor(floor_normal)
        self.register_buffer('floor_normal', floor_normal, persistent=False)

        self.register_buffer('env', env, persistent=False)
    
    def get_light_params(self, global_iter: Optional[int] = None):

        light_dir = self.direction
        light_color = self.color
        ambient = self.ambient

        if (global_iter is not None) and (self.stop_opt is not None) and self.optimize_direction:
            self.optimize_direction = global_iter < self.stop_opt

        if (global_iter is not None) and (self.stop_opt is not None) and self.optimize_ambient:
            self.optimize_ambient = global_iter < self.stop_opt

        if not self.optimize_direction:
            light_dir = light_dir.detach()
            light_color = light_color.detach()
        if not self.optimize_ambient:
            ambient = ambient.detach()
        
        return light_dir, light_color, ambient
    
    def new_direction(self, angle):
        start_direction = F.normalize(torch.Tensor([[-1, 1, 1]]))
        rad = -angle*np.pi/180.0
        roty = torch.Tensor([[np.cos(rad), 0.0, np.sin(rad)],
                            [0.0, 1.0, 0.0],
                            [-np.sin(rad), 0.0, np.cos(rad)]])
        self.direction = nn.Parameter((roty@start_direction.T).T)

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        depth: torch.Tensor,
        t_near: float = 0.1,
        t_far: float = 1000.0,
        global_iter: Optional[int] = None,
        normal: Optional[torch.Tensor] = None
    ):

        light_dir, _, _ = self.get_light_params(global_iter)

        denom = rays_d @ self.floor_normal
        t = (self.floor_point - rays_o) @ self.floor_normal / denom
        cond = ((t > t_near) & (t < t_far))
        cond2 = ((cond) & (depth < 0.25))

        #depth[cond2] = t[cond2]
        depth = torch.where(cond2, t, depth)
        rays_o_2nd = rays_o + rays_d * (depth[..., None])
        rays_d_2nd = F.normalize(-light_dir).expand(len(rays_o), -1)

        if normal is not None:
            normal[cond2] = self.floor_normal
        return rays_o_2nd, rays_d_2nd
    
    def compute_light(
        self, 
        rendered: Dict[str, Any],
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        w2ls: torch.Tensor,
        visibility: GaussianVisibility,
        N_unique: int = 1,
        density: Optional[torch.Tensor] = None,
        pts: Optional[torch.Tensor] = None,
        bgs: Optional[torch.Tensor] = None,
        valid_idxs: Optional[torch.Tensor] = None,
        global_iter: Optional[int] = None,
    ):

        # 2nd ray casting
        # assert self.light is not None
        # assert self.visibility is not None
        depth = rendered['depth_map']
        normal = rendered['normal_map']
        rays_o_2nd, rays_d_2nd = self.forward(rays_o, rays_d, depth, global_iter=global_iter, normal=normal)

        if self.use_gaussians:
            shadow_map = visibility(w2ls, rays_o_2nd, rays_d_2nd, N_unique=N_unique)
        else:
            shadow_map = torch.ones(rays_o_2nd.shape[0])

        light_dir, light_color, ambient = self.get_light_params(global_iter)
            
        if 'normal_map' in rendered and self.use_diffuse:
            diffuse_map = torch.clip((rendered['normal_map'] @ (-F.normalize(light_dir, dim=-1).T))[:,0]*self.light_intensity, 0.0, self.light_intensity) # clip to remove negative
            diffuse_map[rendered['acc_map'] < 0.25] = 1.0
            #irradiance_map = torch.clip((shadow_map[..., None]*diffuse_map[..., None]) + self.light.ambient, 0.0, 1.0)
            irradiance_map = light_color * shadow_map[..., None] * diffuse_map[..., None] + ambient 
            if self.clip_irradiance:
                irradiance_map = torch.clip(irradiance_map, 0, self.light_intensity) # light intensity > 1.0 (so white shirts dont get messed up)
                irradiance_map[rendered['acc_map'] < 0.25] = torch.clip(irradiance_map[rendered['acc_map'] < 0.25], 0.0, 1.0) # bg should not get light > 1
        else:
            irradiance_map = torch.clip(light_color*shadow_map[..., None] + ambient, 0.0, 1.0)
            irradiance_map[rendered['acc_map'] < 0.25] = torch.clip(irradiance_map[rendered['acc_map'] < 0.25], 0.0, 1.0)


        #####

        if bgs is not None:
            rgb_map = rendered['rgb_map']
            acc_map = rendered['acc_map']
            if self.training:
                # during training time: shadow shouldn't be applied to background 
                #rgb_lit_map = (rgb_map * irradiance_map + (1. - acc_map)[..., None] * bgs)
                rgb_lit_map = (rgb_map + (1. - acc_map)[..., None] * bgs) * irradiance_map # override if you do lol
            else:
                rgb_lit_map = (rgb_map + (1. - acc_map)[..., None] * bgs) * irradiance_map
            rgb_map = rgb_map + (1. - acc_map)[..., None] * bgs
            rendered.update(
                shadow_map=shadow_map,
                irradiance_map=irradiance_map,
                rgb_map=rgb_map,
                rgb_lit_map=rgb_lit_map
            )
            if self.use_diffuse:
                rendered.update(
                    diffuse_map=diffuse_map,
                )

        extras = {}
        if self.training:
            #print('sampling')
            output = visibility.sample_gaussian_preds(
                density=density.detach(),
                pts=pts,
                w2ls=w2ls,
                N_samples=256,
                valid_idxs=valid_idxs,
                global_iter=global_iter,
            )
            extras.update(**output)
        return rendered, extras
    
class ImportanceHDRiLight(nn.Module):

    def __init__(
        self,
        num_envs: int = 1,
        hdri_resolution: List = [256,512], # height, width
        floor_point: List = [0,0,0],
        floor_normal: List = [0,-1,0],
        num_secondary_rays: int = 1,
        hdri_angle: float = 0.0,
        hdri_path: str = "",
        exposure: float = 1.5
    ):
        super().__init__()
        self.use_diffuse = False
        self.hdri_resolution = hdri_resolution
        self.num_envs = num_envs
        self.num_secondary_rays = num_secondary_rays
        self.hdri_angle = hdri_angle
        self.half_pixel_size = [0.5/hdri_resolution[0], 0.5/hdri_resolution[1]]

        self.floor_point = torch.Tensor(floor_point)
        self.floor_normal = torch.Tensor(floor_normal)
        self.exposure = exposure

        self.hdri = cv2.cvtColor(cv2.imread(hdri_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
        self.hdri_full_resolution = self.hdri.shape[:2]
        self.envs = cv2.cvtColor(cv2.imread(hdri_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
        self.envs = cv2.GaussianBlur(self.envs, (127,127), 0)
        self.envs = cv2.resize(self.envs, (hdri_resolution[1], hdri_resolution[0]), cv2.INTER_AREA)
        self.envs = self.exposure*torch.Tensor(self.envs)[None,...]        

        self.direction_map = np.zeros((*hdri_resolution, 3))
        for v in range(hdri_resolution[0]):
            for u in range(hdri_resolution[1]):
                direction = equirectangular_range_to_direction(u/hdri_resolution[1],v/hdri_resolution[0])
                self.direction_map[v,u] = direction
        self.direction_map = torch.Tensor(self.direction_map)
    
    def get_diffuse_importance_samples(
        self,
        env_idxs: torch.Tensor,
        normal: torch.Tensor,
    ):
        rot = torch.Tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, -1.0, 0.0]])
        rad = self.hdri_angle*np.pi/180.0
        rotz= torch.Tensor([[np.cos(rad), np.sin(rad), 0.0],
                            [-np.sin(rad), np.cos(rad), 0.0],
                            [0.0, 0.0, 1.0]])
        normal_batched = (rotz@rot@normal.T).T[:,None,:] # convert -y up to +z up and rotate based on hdri angle
        npr = normal.shape[0]
        assert npr == env_idxs.shape[0]
        nsr = self.num_secondary_rays
        npsr = npr*nsr

        # first compute q(x), our importance sampling distribution
        # this will be a combination of diffuse sampling and intensity to form a new PDF
        # we will sample from it using its CDF
        envs_np = self.envs
        envs_intensities = torch.linalg.norm(envs_np, dim=3).reshape(self.envs.shape[0], -1)
        envs_intensities_batch = envs_intensities[env_idxs]

        
        dir_map_p_batch = self.direction_map.reshape(-1,3)[None,...].expand(npr, -1,-1) # Primary x numPixels x 3
        diffuse_modulation = torch.sum(dir_map_p_batch * normal_batched, dim=-1).reshape(npr, -1) # primary x numPixels
        diffuse_modulation = torch.clip(diffuse_modulation,0.0,1.0)

        envs_intensities_batch_modulated = (envs_intensities_batch * diffuse_modulation) # primary x numPixels
        envs_intensities_batch_modulated_sum = torch.sum(envs_intensities_batch_modulated, dim=(1), keepdim=True)

        # q(x) which we use to sample
        envs_intensities_batch_modulated_normalized = envs_intensities_batch_modulated/envs_intensities_batch_modulated_sum # primary x numPixels
        envs_cumsum_ps_batch = envs_intensities_batch_modulated_normalized.cumsum(axis=1)[:, None, :].expand(-1,nsr,-1).reshape(npr, nsr, -1) # primary x secondary x numPixels

        # compute ratio p(x)/q(x)
        px = 2/(self.hdri_resolution[0] * self.hdri_resolution[1]) # true distribution p(x) is uniform over the hemisphere along normal ~1/(num_pixels_in_hdri/2) 
        
        # # # sample from a uniform distribution to pick a U,V coordinate from the environment map to shoot our seondary ray for that batch
        ps_samples = torch.rand(*envs_cumsum_ps_batch.shape[:2],1)
        choices = (ps_samples < envs_cumsum_ps_batch).long().argmax(axis=2)
        Vs = choices / self.hdri_resolution[1]
        Us = choices % self.hdri_resolution[1]
        sampled_qx =  torch.Tensor(np.array([envs_intensities_batch_modulated_normalized[i,choices[i]].cpu().numpy() for i in range(npr)])) + 1e-7
        modulated_px = torch.Tensor(np.array([px * diffuse_modulation[i,choices[i]].cpu().numpy() for i in range(npr)]))
        ratio = modulated_px / sampled_qx
        return Vs.long().detach(), Us.detach(), ratio.detach(), envs_intensities_batch.detach(), diffuse_modulation.detach(), envs_intensities_batch_modulated_normalized.detach()
    
    def get_diffuse_importance_samples_numpy(
        self,
        env_idxs: torch.Tensor,
        normal: torch.Tensor,
    ):
        rot = torch.Tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, -1.0, 0.0]])
        rad = self.hdri_angle*np.pi/180.0
        rotz= torch.Tensor([[np.cos(rad), np.sin(rad), 0.0],
                            [-np.sin(rad), np.cos(rad), 0.0],
                            [0.0, 0.0, 1.0]])
        normal_batched = (rotz@rot@normal.T).T[:,None,:] # convert -y up to +z up and rotate based on hdri angle
        npr = normal.shape[0]
        assert npr == env_idxs.shape[0]
        nsr = self.num_secondary_rays
        npsr = npr*nsr

        # first compute q(x), our importance sampling distribution
        # this will be a combination of diffuse sampling and intensity to form a new PDF
        # we will sample from it using its CDF
        envs_np = self.envs.detach().cpu().numpy()
        envs_intensities = np.linalg.norm(envs_np, axis=3).reshape(self.envs.shape[0], -1)
        envs_intensities_p_batch = envs_intensities[env_idxs.cpu().numpy()]
        envs_intensities_ps_batch = envs_intensities_p_batch # primary x numPixels
        
        dir_map_p_batch = self.direction_map.cpu().numpy().reshape(-1,3)[None,...].repeat(npr, 0) # Primary x numPixels x 3
        diffuse_modulation = np.sum(dir_map_p_batch * normal_batched.cpu().numpy(), axis=-1).reshape(npr, -1) # primary x numPixels
        diffuse_modulation = np.clip(diffuse_modulation,0.0,1.0)

        envs_intensities_ps_batch_modulated = (envs_intensities_ps_batch * diffuse_modulation) # primary x numPixels
        envs_intensities_norm_ps_batch = (envs_intensities_ps_batch_modulated/envs_intensities_ps_batch_modulated.sum(axis=1, keepdims=1)) # primary x numPixels THIS IS q(x)!!
        envs_cumsum_ps_batch = envs_intensities_norm_ps_batch.cumsum(axis=1)[:, None, :].repeat(nsr,1).reshape(npr, nsr, -1) # primary x secondary x numPixels

        # compute ratio p(x)/q(x)
        px = 2/(self.hdri_resolution[0] * self.hdri_resolution[1]) # true distribution p(x) is uniform over the hemisphere along normal ~1/(num_pixels_in_hdri/2) 
        
        # # # sample from a uniform distribution to pick a U,V coordinate from the environment map to shoot our seondary ray for that batch
        ps_samples = np.random.rand(*envs_cumsum_ps_batch.shape[:2],1)
        choices = (ps_samples < envs_cumsum_ps_batch).argmax(axis=2)
        Vs, Us = np.unravel_index(choices, self.envs.shape[1:3])
        ratio = px/np.array([envs_intensities_norm_ps_batch[i,choices[i]] for i in range(npr)])
        return Vs, Us, ratio, envs_intensities_ps_batch, diffuse_modulation, envs_intensities_norm_ps_batch


    def get_ray_o_2nd(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
        acc: torch.Tensor,
        t_near: float = 0.1,
        t_far: float = 10.0,
    ):
        # get intersection between primary rays and ground plane for  casting shadows on the ground plane
        denom = rays_d @ self.floor_normal
        t = (self.floor_point - rays_o) @ self.floor_normal / denom
        cond = ((t > t_near) & (t < t_far))
        floor_mask = ((cond) & (depth < 2))

        # overwrite depth and normal values to include floor
        depth[floor_mask] = t[floor_mask]
        normal[floor_mask] = self.floor_normal
        #acc[cond2] = 1.0
        mask = depth < 2
        # secondary ray origins are at along the primary ray at the depth value given to us by NeRF and ground plane
        rays_o_2nd = rays_o + rays_d * (depth[..., None])
        return rays_o_2nd, mask, floor_mask

    
    def compute_light(
        self, 
        rendered: Dict[str, Any],
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        w2ls: torch.Tensor,
        visibility: GaussianVisibility,
        env_idxs: torch.Tensor = None, # index per primary ray that corresponds to environment map used
        N_unique: int = 1,
        density: Optional[torch.Tensor] = None,
        pts: Optional[torch.Tensor] = None,
        bgs: Optional[torch.Tensor] = None,
        valid_idxs: Optional[torch.Tensor] = None,
        global_iter: Optional[int] = None,
    ):
        if env_idxs is None:
            env_idxs = torch.zeros((rays_o.shape[0])).long()
        
        for p in self.parameters():
            p = torch.abs(p)

        assert 'normal_map' in rendered

        depth = rendered['depth_map'].detach()
        normal = rendered['normal_map'].detach().clone()
        acc_map = rendered['acc_map'].detach()

        rays_o_2nd, mask, floor_mask = self.get_ray_o_2nd(rays_o, rays_d, depth, normal, acc_map, t_near=0.05, t_far=50) # (#Primary x 3)
        rot = torch.Tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, -1.0, 0.0]])
            
        rad = self.hdri_angle*np.pi/180.0
        rotz= torch.Tensor([[np.cos(rad), np.sin(rad), 0.0],
                            [-np.sin(rad), np.cos(rad), 0.0],
                            [0.0, 0.0, 1.0]])

        #mask = depth < 0.5
        if mask.all():
            irradiance_samples = torch.ones((rays_o.shape[0], 3))
            irradiance_map = irradiance_samples
        else:
            irradiance_samples = torch.zeros((rays_o.shape[0], 3))
                        

            # importance sampling weighted towards brighter regions
            Vs, Us, ratio,_,_,_ = self.get_diffuse_importance_samples(env_idxs, normal)
            Us_float = (Us/self.hdri_resolution[1]) + self.half_pixel_size[1] + torch.rand(*Us.shape)*self.half_pixel_size[1]
            Vs_float = (Vs/self.hdri_resolution[0]) + self.half_pixel_size[0] + torch.rand(*Vs.shape)*self.half_pixel_size[0]
            # transform u,v coordiniates to x,y,z direction vector using an equirectangular projection
            # convert from +z up to -y up
            rays_d_2nd_importance = (rot@rotz.T@torch.Tensor(equirectangular_range_to_direction_torch(Us_float.reshape(-1), Vs_float.reshape(-1)))).T
            rays_d_2nd_importance = rays_d_2nd_importance.reshape(-1,self.num_secondary_rays,3) # (#Primary x #Secondary x 3)
            

            # calculate occlusion and irradiance for each sample
            for i in range(self.num_secondary_rays):
                # importance samples
                occlusion = visibility(w2ls, rays_o_2nd, rays_d_2nd_importance[:, i], N_unique=N_unique).detach()
                hdri_values = self.envs[env_idxs, Vs[:, i], Us[:, i]] #f(x)
                
                # occlusion * f(x) * p(x)/q(x) | x~q(x)
                irradiance_samples += occlusion[..., None] * hdri_values * ratio[:, i, None]


            # average across all samples and apply transformation
            irradiance_map = irradiance_samples / self.num_secondary_rays
            
            if torch.isnan(irradiance_map).any():
                print("irradiance map contains nans!")

        # #make the bg the hdri
        us, vs = direction_to_equirectangular_range((rotz@rot@rays_d.T).T)
        us = (us*(self.hdri_full_resolution[1]-1)).flatten().astype(int)
        vs = (vs*(self.hdri_full_resolution[0]-1)).flatten().astype(int)
        gamma = 2.8
        #irradiance_map = irradiance_map + (1. - acc_map)[..., None] * torch.Tensor(self.hdri[vs, us])
        #irradiance_map[depth < 0.5] = torch.Tensor(self.hdri[vs, us])
        bgs2 = torch.Tensor(self.hdri[vs, us]**(1/gamma))
        bgs2[floor_mask] = torch.Tensor([1.0,1.0,1.0])

        irradiance_map = irradiance_map**(1/gamma) # apply gamma to linear irradiance
        irradiance_map[mask] = 1
        #####

        if bgs is not None:
            rgb_map = rendered['rgb_map']
            #acc_map = rendered['acc_map']
            rgb_lit_map = (rgb_map + (1. - acc_map)[..., None] * bgs2) * irradiance_map
            rgb_map = rgb_map + (1. - acc_map)[..., None] * bgs
            irradiance_map[mask] = 0.75 # makes the person pop out more in irradiance map :)
            rendered.update(
                shadow_map=irradiance_map.mean(dim=1), # dont want to deal with collected outputs etc.
                irradiance_map=irradiance_map,
                rgb_map=rgb_map,
                rgb_lit_map=rgb_lit_map
            )
        extras = {}
        return rendered, extras
    

class DirectionalLightRelight(nn.Module):

    def __init__(
        self,
        direction: List = [[0.5773, 0.5773, 0.5773]],
        color: List = [[1.0, 1.0, 1.0]],
        ambient: List = [0.1],
        floor_point: List = [0,0,0],
        floor_normal: List = [0,-1,0],
        light_intensity: float = 1.5,
        use_diffuse: bool = True,
        use_gaussians: bool = True,
        clip_irradiance: bool = True,
    ):
        super().__init__()
        self.light_intensity = light_intensity
        self.use_diffuse = use_diffuse
        self.use_gaussians = use_gaussians
        self.clip_irradiance = clip_irradiance

        
        self.direction = torch.Tensor(direction)
        self.color = torch.Tensor(color)
        self.ambient = torch.Tensor(ambient)


        self.floor_point = torch.Tensor(floor_point)
        self.floor_normal = torch.Tensor(floor_normal)
        
    def new_direction(self, angle):
        start_direction = F.normalize(torch.Tensor([[-1, 1, 1]]))
        rad = -angle*np.pi/180.0
        roty = torch.Tensor([[np.cos(rad), 0.0, np.sin(rad)],
                            [0.0, 1.0, 0.0],
                            [-np.sin(rad), 0.0, np.cos(rad)]])
        self.direction = (roty@start_direction.T).T

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        depth: torch.Tensor,
        t_near: float = 0.1,
        t_far: float = 100.0,
        normal: Optional[torch.Tensor] = None
    ):

        denom = rays_d @ self.floor_normal
        t = (self.floor_point - rays_o) @ self.floor_normal / denom
        cond = ((t > t_near) & (t < t_far))
        cond2 = ((cond) & (depth < 0.25))

        #depth[cond2] = t[cond2]
        depth = torch.where(cond2, t, depth)
        rays_o_2nd = rays_o + rays_d * (depth[..., None])
        rays_d_2nd = F.normalize(-self.direction).expand(len(rays_o), -1)

        if normal is not None:
            normal[cond2] = self.floor_normal
        return rays_o_2nd, rays_d_2nd
    
    def compute_light(
        self, 
        rendered: Dict[str, Any],
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        w2ls: torch.Tensor,
        visibility: GaussianVisibility,
        N_unique: int = 1,
        density: Optional[torch.Tensor] = None,
        pts: Optional[torch.Tensor] = None,
        bgs: Optional[torch.Tensor] = None,
        valid_idxs: Optional[torch.Tensor] = None,
        global_iter: Optional[int] = None,
    ):

        # 2nd ray casting
        # assert self.light is not None
        # assert self.visibility is not None
        depth = rendered['depth_map'].detach()
        normal = rendered['normal_map'].detach()
        rays_o_2nd, rays_d_2nd = self.forward(rays_o, rays_d, depth=depth, normal=normal)

        if self.use_gaussians:
            shadow_map = visibility(w2ls, rays_o_2nd, rays_d_2nd, N_unique=N_unique)
        else:
            shadow_map = torch.ones(rays_o_2nd.shape[0])

        light_dir = self.direction
        light_color = self.color
        ambient = self.ambient
            
        if 'normal_map' in rendered and self.use_diffuse:
            diffuse_map = torch.clip((rendered['normal_map'] @ (-F.normalize(light_dir, dim=-1).T))[:,0]*self.light_intensity, 0.0, self.light_intensity) # clip to remove negative
            diffuse_map[rendered['acc_map'] < 0.25] = 1.0
            irradiance_map = light_color * shadow_map[..., None] * diffuse_map[..., None] + ambient 
            if self.clip_irradiance:
                irradiance_map = torch.clip(irradiance_map, 0, self.light_intensity) # light intensity > 1.0 (so white shirts dont get messed up)
                irradiance_map[rendered['acc_map'] < 0.25] = torch.clip(irradiance_map[rendered['acc_map'] < 0.25], 0.0, 1.0) # bg should not get light > 1
        else:
            irradiance_map = torch.clip(light_color*shadow_map[..., None] + ambient, 0.0, 1.0)
            irradiance_map[rendered['acc_map'] < 0.25] = torch.clip(irradiance_map[rendered['acc_map'] < 0.25], 0.0, 1.0)


        #####

        if bgs is not None:
            rgb_map = rendered['rgb_map']
            acc_map = rendered['acc_map']
            if self.training:
                # during training time: shadow shouldn't be applied to background 
                #rgb_lit_map = (rgb_map * irradiance_map + (1. - acc_map)[..., None] * bgs)
                rgb_lit_map = (rgb_map + (1. - acc_map)[..., None] * bgs) * irradiance_map # override if you do lol
            else:
                rgb_lit_map = (rgb_map + (1. - acc_map)[..., None] * bgs) * irradiance_map
            rgb_map = rgb_map + (1. - acc_map)[..., None] * bgs
            rendered.update(
                shadow_map=shadow_map,
                irradiance_map=irradiance_map,
                rgb_map=rgb_map,
                rgb_lit_map=rgb_lit_map
            )
            if self.use_diffuse:
                rendered.update(
                    diffuse_map=diffuse_map,
                )

        extras = {}
        return rendered, extras