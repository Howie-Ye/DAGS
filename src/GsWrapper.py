from src import (
    os, sys, math, tqdm, random, time, json, field,
    torch, nn, torchvision, F, o3d,
    np, cv2,
    Console, List, Tensor,
    rearrange, repeat, reduce
)

import plotly.graph_objs as go
from plyfile import PlyData, PlyElement

from src.gaussian_splatting.scene.gaussian_model import GaussianModel
from src.gaussian_splatting.utils.system_utils import searchForMaxIteration,mkdir_p
from src.gaussian_splatting.scene.cameras import Camera
from src.gaussian_splatting.utils.general_utils import build_rotation, get_expon_lr_func
from src.gaussian_splatting.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2

from configs.gs_configs import TrainConfig,ModelParams,PipelineParams,OptimizationParams

class GsWrapper(GaussianModel):
    def __init__(self, traincfg):
        self.model_params = traincfg.model_param
        self.pipeline_params = traincfg.pipe_param
        self.opt_params = traincfg.opt_param
        background_color = [1, 1, 1] if self.model_params.white_background else [0, 0, 0]
        self.bg_color = torch.tensor(background_color, dtype=torch.float32, device="cuda")
        super().__init__(sh_degree=self.model_params.sh_degree)
        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = self.model_params.cameras_extent
        self.active_gs = None

    @property
    def device(self):
        with torch.no_grad():
            return self.get_xyz.device

    @property
    def gs_num(self):
        return self.get_xyz.shape[0]

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self._rotation)

    def add_new_gs(self, xyz, surf_points, rgb, normal, inital:bool ,frame_idx=None):
        # Input: xyz , colors, normals
        fused_point_cloud = xyz
        points_num = xyz.shape[0]
        if surf_points is not None:
            tmp_pcd = torch.cat([xyz, surf_points],dim=0)
        else:
            tmp_pcd = fused_point_cloud
        fused_color = RGB2SH(rgb)
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min( distCUDA2(tmp_pcd),  0.0000001,)[:points_num]
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)


        z_w = torch.tensor([[0.,0.,1.]]).to(normal.device)
        rots = self.qua_between_vectors(z_w, normal)
        #rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        opacities = self.inverse_opacity_activation(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        if inital:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            if frame_idx is not None:
                self._idx = torch.full_like(scales,frame_idx).int()
                self._idx[:,1] = torch.as_tensor(1)
        else:
            self.densification_postfix(
                new_xyz = fused_point_cloud,
                new_features_dc = features[:,:,0:1].transpose(1, 2),
                new_features_rest = features[:,:,1:].transpose(1, 2),
                new_opacities = opacities,
                new_scaling = scales,
                new_rotation = rots )
            if frame_idx is not None:
                new_idx = torch.full_like(scales, frame_idx).int()
                new_idx[:,1] = torch.as_tensor(1)
                self._idx = torch.cat([self._idx,new_idx],dim=0)

    def update_spatial_lr_scale(self,cam_centers):
        avg_cam_center = cam_centers.mean(dim=0)
        dist_max = (cam_centers - avg_cam_center).norm(dim=-1).max()
        self.spatial_lr_scale = dist_max.item()
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.opt_params.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=self.opt_params.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=self.opt_params.position_lr_delay_mult,
                                                    max_steps=self.opt_params.position_lr_max_steps)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, frame_idx=None, mask=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        if mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask,mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        if frame_idx != None:
            new_idx = self._idx[selected_pts_mask].repeat(N, 1)
            self._idx = torch.cat([self._idx, new_idx], dim=0)
            self._idx = self._idx[~prune_filter]
        if self.active_gs != None:
            self.active_gs = torch.cat([self.active_gs, self.active_gs[selected_pts_mask].repeat(N)], dim=0)
            self.active_gs = self.active_gs[~prune_filter]

    def densify_and_clone(self, grads, grad_threshold, scene_extent, frame_idx=None, mask=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling,dim=1).values <= self.percent_dense * scene_extent)

        if mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask,mask)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc,
                                   new_features_rest, new_opacities,
                                   new_scaling, new_rotation)
        if frame_idx != None:
            new_idx = self._idx[selected_pts_mask]
            self._idx = torch.cat([self._idx, new_idx], dim=0)
        if self.active_gs != None:
            self.active_gs = torch.cat([self.active_gs, self.active_gs[selected_pts_mask]], dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, frame_idx=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        mask =None if (self.active_gs==None) else (self.active_gs>self.model_params.active_gs_threshold)
        self.densify_and_clone(grads, max_grad, extent, frame_idx=frame_idx,mask=mask)

        mask =None if (self.active_gs==None) else (self.active_gs>self.model_params.active_gs_threshold)
        self.densify_and_split(grads, max_grad, extent, frame_idx=frame_idx,mask=mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if self.active_gs != None:
            prune_mask = torch.logical_and(prune_mask, (self.active_gs>self.model_params.active_gs_threshold))

        self.prune_points(prune_mask)
        if frame_idx != None:
            self._idx = self._idx[~prune_mask]
        torch.cuda.empty_cache()


    def save_ply_3dgs(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = torch.cat([self._scaling.detach().cpu(), torch.ones((self._scaling.shape[0], 1)) * (-9)],dim=-1).numpy()

        rotation = self._rotation.detach().cpu().numpy()

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1] + 1):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def plot_point_cloud(
            self,
            points=None,
            colors=None,
            n_points_to_plot: int = 50000,
            width=1000,
            height=500,
    ):
        """Plot the generated 3D point cloud with plotly.

        Args:
            n_points_to_plot (int, optional): _description_. Defaults to 50000.
            points (_type_, optional): _description_. Defaults to None.
            colors (_type_, optional): _description_. Defaults to None.
            width (int, optional): Defaults to 1000.
            height (int, optional): Defaults to 1000.

        Raises:
            ValueError: _description_

        Returns:
            go.Figure: The plotly figure.
        """

        with torch.no_grad():
            if points is None:
                points, colors = self.generate_point_cloud()

            points_idx = torch.randperm(points.shape[0])[:n_points_to_plot]
            points_to_plot = points[points_idx].cpu()
            colors_to_plot = colors[points_idx].cpu()

            z = points_to_plot[:, 2]
            x = points_to_plot[:, 0]
            y = points_to_plot[:, 1]
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors_to_plot,  # set color to an array/list of desired values
                    # colorscale = 'Magma'
                ),
            )
            layout = go.Layout(
                scene=dict(bgcolor="white", aspectmode="data"),
                template="none",
                width=width,
                height=height,
            )
            fig = go.Figure(data=[trace], layout=layout)
            # fig.update_layout(template='none', scene_aspectmode='data')

            # fig.show()
            return fig


    @staticmethod
    def qua_between_vectors(v1: Tensor, v2: Tensor):
        '''
        Calculate the quaternion to transform v1 to v2, v has been normalized, return (w,x,y,z)
        shape of v is [n,3]
        '''
        q_xyz = torch.cross(v1, v2, dim=-1)  # [n,3]
        q_w = torch.sum(v1 * v2, dim=-1, keepdim=True) + 1.  # [n,1]
        q = F.normalize(torch.cat((q_w, q_xyz), dim=-1), dim=-1)
        return q

    @staticmethod
    def xyz2uv(xyz,w2c,intrins):
        ''' project each 3d point to camera '''
        pts3d_homo = torch.cat([xyz,torch.ones_like(xyz[:, 0]).reshape(-1,1)],dim=-1)[...,None]   #[n,4,1]
        pts2d = ( w2c @ pts3d_homo )[:,:3]
        uv = intrins @ pts2d  # [3,3] @ [n, 3, 1] = [Pn, Cn, 3, 1]
        z = uv[:, -1:] + 1e-5
        uv = (uv[:, :2] / z ).squeeze(-1)
        return uv