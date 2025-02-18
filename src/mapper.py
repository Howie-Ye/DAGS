from src import (os, sys, math, random, time,
    torch, nn, torchvision, F, o3d,
    np, cv2,
    Console, List, Tensor,Optional,
    rearrange, repeat, reduce)

from multiprocessing.connection import Connection

from src.utils.common import get_samples,align_scale_and_shift
from src.utils.datasets import get_dataset, load_mono_depth,BaseDataset

from src.utils.eval_render import eval_kf_imgs
from src.utils.Printer import Printer, FontColor
from src.depth_video import DepthVideo
from src.deblur import DeblurModule
from src.Exposure import Exposure
from src.GsWrapper import GsWrapper
from src.gaussian_splatting.render import render
from src.KF_Camera import KF_Camera
from src.gaussian_splatting.scene.cameras import MiniCam
from src.gaussian_splatting.utils.point_utils import depth_to_normal
from fused_ssim import fused_ssim
from tqdm import tqdm
import wandb
from wandb import sdk as wandb_sdk
import functools
print = tqdm.write
from src.utils.p3d_functions import matrix_to_quaternion, quaternion_multiply
import open3d as o3d
from configs.gs_configs import TrainConfig
class Mapper(object):
    """
    Mapper thread.
    """
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.gs_cfg:TrainConfig = slam.gs_cfg
        self.printer:Printer = slam.printer
        if self.cfg['only_tracking']:
            return
        self.pipe = pipe
        self.output = slam.output
        self.verbose = slam.verbose
        self.video:DepthVideo = slam.video
        self.stream = slam.stream
        self.low_gpu_mem = True
        self.device = self.cfg['device']
        self.gswrapper:GsWrapper = GsWrapper(self.gs_cfg)
        self.exposure:Optional[Exposure] = None
        self.iteration = 0
        self.deblur:Optional[DeblurModule] = None
        self.logger:wandb_sdk.wandb_run.Run = slam.logger
        self.use_deblur = self.cfg['deblur']
        self.use_exposure = self.cfg['exposure_compensate']
        self.mapping_window_size = self.cfg['mapping']['mapping_window_size']
        self.opt_idx = None
        self.use_sensor_depth = self.cfg['use_sensor_depth']
        self.map_deformation = self.cfg['map_deformation']
        self.depth_valid_threshold = self.cfg['depth_valid_threshold']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.HW = self.H * self.W
        self.FoVx = 2 * math.atan(self.W / (2 * self.fx))
        self.FoVy = 2 * math.atan(self.H / (2 * self.fy))
        self.rgb_loss,self.normal_loss,self.normal_prior_loss,self.depth_loss,self.dist_loss = 0.,0.,0.,0.,0.

    def set_pipe(self, pipe):
        self.pipe = pipe


    def get_pose_and_depth(self, v_idx):
        """
        Given the index of keyframe in depth_video, fetch the estimated pose and depth

        Args:
            v_idx : index of the keyframe in depth_video

        Returns:
            c2w (tensor, [n,4,4]): estimated pose of the selected keyframe
            w2c (tensor, [n,4,4]): estimated pose of the selected keyframe
            est_droid_depth (tensor, [H,W]): esitmated depth map from the tracker
        """
        est_droid_depth, valid_depth_mask, c2w, w2c = self.video.get_depth_and_pose(v_idx,self.device)

        est_droid_depth[~valid_depth_mask] = 0

        return c2w, w2c, est_droid_depth
    
    def anchor_gs(self, cur_v_idx:int):

        idx_list = [cur_v_idx]
        if self.gswrapper.gs_num == 0:
            for i in self.dirty_video_idx:
                if i != cur_v_idx and i > 0:
                    idx_list.append(i.item())

        for idx in idx_list:
            if (self.kf_cam.fuse_depth_info[idx,2] < (self.depth_valid_threshold/self.HW) ):
                continue
            gt_color = self.video.images[idx]
            rgb_o3d = o3d.geometry.Image((gt_color * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            raw_depth=self.kf_cam.proxy_depth[idx].clone()
            anchor_mask = raw_depth > 0
            anchor_mask[[0,-1],:] = False
            anchor_mask[:,[0, -1]] = False

            raw_normal = self.kf_cam.normal[idx].permute(1, 2, 0)
            w2c = self.kf_cam.w2c[idx].cpu().numpy()
            c2w = self.kf_cam.c2w[idx]

            if self.gswrapper.gs_num == 0:
                add_pts_num = int(anchor_mask.sum().item() / 8)
                surf_points = None
            else:
                with torch.no_grad():
                    pkg = render(self.kf_cam[idx],self.gswrapper,self.gswrapper.pipeline_params,
                                 self.gswrapper.bg_color,transmittance_trunc=0.3)
                alpha_mask = pkg['rend_alpha'] < 0.5
                depth_mask = pkg['surf_depth'] > (raw_depth + raw_depth.std())
                surf_points = self.gswrapper.get_xyz[pkg['n_touched'] > 0]
                anchor_mask *= (alpha_mask + depth_mask)[0]
                add_pts_num = int( anchor_mask.sum().item() / 16 )
                if add_pts_num == 0:  continue


            raw_depth[~anchor_mask] += 100
            depth_o3d = o3d.geometry.Image(raw_depth.contiguous().cpu().numpy().astype(np.float32))
            normal_np = raw_normal[anchor_mask].cpu().numpy()
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.kf_cam.PinholeCameraIntrinsic,
                extrinsic=w2c,
                project_valid_depth_only=True,
            )
            pcd.normals = o3d.utility.Vector3dVector(normal_np)
            pcd = pcd.farthest_point_down_sample(add_pts_num)

            xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)
            rgb = torch.from_numpy(np.asarray(pcd.colors)).float().to(self.device)
            normal_c = torch.from_numpy(np.asarray(pcd.normals)).float().to(self.device)
            # rotate normals from cam coordinate to world coordinate
            R_c2w = c2w[:3, :3]
            normal = torch.matmul(R_c2w[None], normal_c[..., None]).squeeze(-1)
            self.gswrapper.add_new_gs(xyz, surf_points,
                                      rgb, normal,
                                      inital=self.init,
                                      frame_idx=idx)
            if self.init:
                self.gswrapper.training_setup(self.gswrapper.opt_params)
                self.init = False

        torch.cuda.empty_cache()

    def update_map_once(self, kf_id, iteration,deblur:bool, final_refine:bool = False):
        global_ba:bool = self.global_ba
        traking:bool = not (global_ba or final_refine)
        opt_deblur:bool = (self.deblur!=None) and deblur
        is_blur_kf:bool = (self.deblur!=None) and (kf_id in self.deblur.deblur_idx)


        cam:MiniCam = self.opt_cam[kf_id]
        gt_image = self.video.images[kf_id]
        ref_depth = self.kf_cam.proxy_depth[kf_id]

        if final_refine:
            self.gswrapper.update_learning_rate(iteration)

        opt = self.gswrapper.opt_params

        # Render
        if opt_deblur or is_blur_kf:
            render_pkg_subframes = []
            subframes = self.deblur.provide_subframes(kf_id, cam)
            override_opa,override_scaling=None,None
            if opt_deblur and (not final_refine):
                scale_map = torch.ones_like(self.gswrapper.get_opacity) * self.deblur.scale_down_rate
                override_scaling = self.gswrapper.get_scaling * self.deblur.scale_down_rate
                override_opa = self.gswrapper.get_opacity * scale_map
            render_pkg = render(subframes[0], self.gswrapper,
                                self.gswrapper.pipeline_params,
                                self.gswrapper.bg_color,
                                override_opa=override_opa,
                                override_scaling=override_scaling)

            image0, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            render_pkg_subframes.append(image0)
            subframes.pop(0)
            for frame in subframes:
                render_pkg_sub = render(frame, self.gswrapper,
                                        self.gswrapper.pipeline_params,
                                        self.gswrapper.bg_color,
                                        override_opa=override_opa,
                                        override_scaling=override_scaling)
                render_pkg_subframes.append(render_pkg_sub['render'])

            render_subframes = torch.stack(render_pkg_subframes)
            image = render_subframes.mean(dim=0)
        else:
            render_pkg = render(viewpoint_camera = cam,
                                pc = self.gswrapper,
                                pipe = self.gswrapper.pipeline_params,
                                bg_color = self.gswrapper.bg_color)
            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        radii_visible = radii[visibility_filter]
        n_touched = render_pkg['n_touched']
        alpha_visible = n_touched > 0  # [n,]
        surf_depth = render_pkg['surf_depth'][0]

        if final_refine and self.use_exposure:
            exp_scale, exp_shift = self.exposure.get_scale_shift(kf_id)
            image = image * exp_scale + exp_shift


        Ll1 = torch.abs((image - gt_image))
        if (final_refine):
            with torch.no_grad():
                mask = Ll1 > (Ll1.mean() * opt.lambda_l1_rgb_threshold)
            Ll1_rgb = Ll1[mask].mean()
        else:
            Ll1_rgb = Ll1.mean()

        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1_rgb + opt.lambda_dssim * (1.0 - fused_ssim(image[None], gt_image[None]))
        loss_total = loss_rgb

        # regularization

        lambda_normal = opt.lambda_normal
        lambda_normal_prior = opt.lambda_normal_prior

        lambda_depth = opt.lambda_depth
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal)[:,1:-1,1:-1].sum(dim=0))
        loss_normal = lambda_normal * (normal_error).mean()
        loss_total += loss_normal

        normal_prior = self.kf_cam.normal[kf_id]


        loss_normal_prior = (lambda_normal_prior *
             ((1 - (surf_normal * normal_prior).sum(dim=0).abs()))[1:-1,1:-1].mean())
        loss_total += loss_normal_prior

        # depth loss
        Ll1_depth = torch.abs((surf_depth - ref_depth))
        if self.use_sensor_depth:
            Ll1_depth = Ll1_depth[ref_depth>0]
        loss_depth = lambda_depth * Ll1_depth.mean()
        loss_total += loss_depth

        loss_total.backward()

        # note activate gaussians
        if (traking and (not opt_deblur) and (iteration <= opt.densification_interval)):
            if self.gswrapper.active_gs is None:
                self.gswrapper.active_gs = torch.zeros(self.gswrapper.gs_num).cuda()
            self.gswrapper.active_gs[alpha_visible] += 1

        if global_ba and (iteration < self.opt_idx.numel()):
            self.visible_count[alpha_visible] += 1

        # Densification
        if (iteration < opt.densify_until_iter):

            self.gswrapper.max_radii2D[visibility_filter] = (
                torch.max(self.gswrapper.max_radii2D[visibility_filter],radii_visible) )
            if (not opt_deblur) or final_refine:
                self.gswrapper.add_densification_stats(viewspace_point_tensor, visibility_filter)
            # update gswrapper._idx
            replace_idx = torch.where( (n_touched>=self.gswrapper._idx[:,1])+(self.gswrapper._idx[:,0]==kf_id) )[0]
            self.gswrapper._idx[replace_idx,0] = torch.as_tensor(kf_id).int()
            self.gswrapper._idx[replace_idx,1] = n_touched[replace_idx]

            if not global_ba:
                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if (iteration >= opt.opacity_reset_interval ) else None
                    self.gswrapper.densify_and_prune(opt.densify_grad_threshold,
                                                     opt.opacity_cull,
                                                     self.gswrapper.spatial_lr_scale,
                                                     size_threshold,
                                                     frame_idx=kf_id)
                    self.gswrapper.active_gs = None
                    if iteration % opt.opacity_reset_interval == 0 or \
                            (self.gswrapper.model_params.white_background and iteration == opt.densify_from_iter):
                        if final_refine:
                            self.gswrapper.reset_opacity()

            # Optimizer step
        if final_refine and self.use_exposure:
            self.exposure.optimizer.step()
            self.exposure.optimizer.zero_grad(set_to_none=True)

        if opt_deblur:
            self.deblur.optimizer.step()
            self.deblur.optimizer.zero_grad(set_to_none=True)

        self.gswrapper.optimizer.step()
        self.gswrapper.optimizer.zero_grad(set_to_none=True)

        self.iteration += 1

        loss_dict = (
            loss_total.item(),
            loss_rgb.item(),
            loss_normal.item(),
            loss_normal_prior.item(),
            loss_depth.item(),
        )

        return loss_dict


    def optimize_map(self, cur_idx, rounds):

        if (self.opt_idx is not None):
            self.opt_idx = torch.cat( (self.opt_idx,torch.tensor([cur_idx]).cuda()))

        else:
            self.first_optimize = True
            self.opt_idx = self.dirty_video_idx[self.kf_cam.fuse_depth_info[self.dirty_video_idx,2]>self.depth_valid_threshold/self.HW]

        if self.global_ba:
            self.opt_idx = self.dirty_video_idx
            self.visible_count = torch.zeros(self.gswrapper.gs_num).cuda()
            rounds = max(rounds,self.opt_idx.shape[0]*2)

        self.opt_idx = self.opt_idx[self.opt_idx>0]
        opt_kf_num = len(self.opt_idx)
        if (not self.global_ba) and (not self.first_optimize) :
            if opt_kf_num > self.mapping_window_size:
                self.opt_idx = self.opt_idx[-self.mapping_window_size:]
                opt_kf_num = self.mapping_window_size
        kf_stack = None
        if self.deblur is not None:
            self.set_param_groups_grad(optimizer=self.deblur.optimizer,require_grad=False)
        self.opt_cam = {}
        for idx in self.opt_idx:
            self.opt_cam[idx.item()] = self.kf_cam[idx]

        iteration = 0
        while (iteration < rounds):
            if not kf_stack:
                kf_stack = list(range(opt_kf_num))
                random.shuffle(kf_stack)

            kf_indice = kf_stack.pop()
            if self.global_ba and iteration==self.opt_idx.shape[0]:
                self.gswrapper.prune_points(mask=(self.visible_count==0))
                if self.gswrapper._idx is not None:
                    self.gswrapper._idx = self.gswrapper._idx[self.visible_count>0]
                self.gswrapper.densify_and_prune(self.gswrapper.opt_params.densify_grad_threshold,
                                                 self.gswrapper.opt_params.opacity_cull,
                                                 self.gswrapper.spatial_lr_scale,
                                                 20,
                                                 frame_idx=self.opt_idx[kf_indice].item())
                self.gswrapper.reset_opacity()


            loss_dict = self.update_map_once(kf_id = self.opt_idx[kf_indice].item(),
                                             iteration = iteration,
                                             deblur = False,
                                             final_refine = False)
            iteration += 1
            if (not kf_stack) and (not self.global_ba) and (not self.first_optimize):
                random_idx = torch.randint(1, self.opt_idx.min(), ()).item()
                if random_idx not in self.opt_cam.keys():
                    self.opt_cam[random_idx] = self.kf_cam[random_idx]
                iteration += 1
                self.update_map_once(kf_id=random_idx,
                                     iteration=iteration,
                                     deblur=False,
                                     final_refine=False)

            self.log_loss(loss_dict,prefix='')

        if self.use_deblur and (not self.global_ba):
            self.set_param_groups_grad(optimizer=self.deblur.optimizer, require_grad=True)
            self.set_param_groups_grad(optimizer=self.gswrapper.optimizer,require_grad=False)
            rank = 0
            for i in self.opt_idx:
                idx = i.item()
                if idx in self.deblur.ready_opt_idx:
                    for it in range(20):
                        self.update_map_once(idx, iteration=it, deblur=True, final_refine=False)
                    self.deblur.ready_opt_idx.remove(idx)
                    # adjust subframes num if the blurring is not significent
                    self.deblur.change_subframe_num(idx=idx, note_idx=True)
            if cur_idx % 5 == 0:
                self.printer.print(f"deblur_idx: {self.deblur.deblur_idx}")
            self.set_param_groups_grad(optimizer=self.gswrapper.optimizer, require_grad=True)
        self.global_ba = False
        self.first_optimize = False



    def mapping_keyframe(self, cur_v_idx):
        self.anchor_gs(cur_v_idx)
        rounds = 200 if self.global_ba else 150
        self.optimize_map(cur_idx=cur_v_idx,rounds=rounds)
        return True


    def deformation_kf(self, v_idxs, c2w_n, w2c_n):
        # calculate delta xyz and rotation of gs in this really dirty kf frustum
        # cos(|alpha|) = 0.5 * ( trace(r1^t * r2) - 1 )
        # diff_rot = 1 - cos(|alpha|)

        c2w_f = self.kf_cam.c2w[v_idxs]
        w2c_f = self.kf_cam.w2c[v_idxs]
        trac = (w2c_n[:,:3, :3] @ c2w_f[:,:3, :3]).diagonal(offset=0, dim1=1, dim2=2).sum(dim=-1)
        diff_rot = 1 - 0.5 * (trac - 1)
        diff_pos = (c2w_n[:,:3, 3] - c2w_f[:,:3, 3]).norm(dim=-1)


        tmp_idx = torch.arange(len(v_idxs)).int()

        opt_priority = torch.min(
            (diff_rot.sort(descending=True).indices).sort().indices,
            (diff_pos.sort(descending=True).indices).sort().indices
        )
        self.opt_idx = v_idxs[opt_priority<self.mapping_window_size]
        if tmp_idx.numel()==0:    return
        xyz_new = self.gswrapper.get_xyz.clone()
        rot_new = self.gswrapper.get_rotation.clone()
        scale_new = self.gswrapper.get_scaling.clone()
        for i in tmp_idx:
            idx = v_idxs[i].item()
            kf_related_gs = torch.where(self.gswrapper._idx == idx)[0]  # n -> n'
            if kf_related_gs.shape[0] == 0:  continue
            camera_center_f = c2w_f[i,:3,3]
            with torch.no_grad():
                tmp_pkg = render(self.kf_cam[idx], self.gswrapper, self.gswrapper.pipeline_params,
                                 self.gswrapper.bg_color)
                surf_depth = tmp_pkg['surf_depth'][0]
            related_gs_xyz = self.gswrapper.get_xyz[kf_related_gs]
            proxy_depth_previous = self.kf_cam.proxy_depth[idx]
            gs_uv = self.gswrapper.xyz2uv(xyz=related_gs_xyz,
                                          w2c=self.kf_cam.w2c[idx],
                                          intrins=self.kf_cam.intrinsic_matrix).round().int()  # [n',2],uv

            frustum_mask = (gs_uv[:,0]>=0) & (gs_uv[:,0]<self.W) & (gs_uv[:,1]>=0) & (gs_uv[:,1]<self.H)  # n' -> n''
            gs_uv_valid = gs_uv[frustum_mask].permute([1,0])  # [n'',2]
            hi, wi = gs_uv_valid[1], gs_uv_valid[0]
            gs_depth_uv = (related_gs_xyz[frustum_mask]-camera_center_f).norm(dim=-1)
            # avoid mismatch of depth in the edge regions.
            pd_uv = proxy_depth_previous[hi, wi]
            sd_uv = surf_depth[hi, wi]
            delta_gs_pd = pd_uv - gs_depth_uv
            delta_gs_sd = sd_uv - gs_depth_uv
            pre_depth_uv = torch.where((delta_gs_pd.abs() < delta_gs_sd.abs()),
                                       pd_uv - torch.clamp(delta_gs_pd, min=0),
                                       sd_uv - torch.clamp(delta_gs_sd, min=0))
            gs_rot_f = self.gswrapper.get_rotation[kf_related_gs]  # [n',4]
            gs_xyz_f = self.gswrapper.get_xyz[kf_related_gs]  # [n',3]
            gs_scale_f = self.gswrapper.get_scaling[kf_related_gs]

            # xyz_n = (c2w_n @ w2c_f)@(Ray_dir*(depth_n-surf_d)+xyz)
            # Ray_dir = normalize(xyz-camera_center)
            # camera_center = c2w[:3,3]
            delta_depth = torch.ones_like(gs_xyz_f)
            new_depth_uv = self.get_proxy_depth(v_idx=idx)[0, hi, wi]
            ray_dir = F.normalize(gs_xyz_f - camera_center_f,dim=-1)
            delta_depth_frustum = (new_depth_uv - pre_depth_uv)[:, None]
            delta_depth[frustum_mask] = delta_depth_frustum
            delta_depth[~frustum_mask] = torch.as_tensor(delta_depth_frustum.mean())
            transform_mtx = c2w_n[i] @ w2c_f[i]
            xyz_n = (transform_mtx[None, :3, :3] @ (delta_depth * ray_dir + gs_xyz_f)[..., None]
                     + transform_mtx[:3, 3][None, :, None])  # [n,3,1]
            transform_q = matrix_to_quaternion(transform_mtx[:3, :3])
            rot_n = quaternion_multiply(transform_q[None], gs_rot_f)
            depth_scale = torch.ones_like(gs_scale_f)
            depth_scale_frustum=( new_depth_uv / (pre_depth_uv + 1e-6) )[:, None]
            depth_scale[frustum_mask] = depth_scale_frustum
            depth_scale[~frustum_mask] = torch.as_tensor(depth_scale_frustum.mean())
            scale_n = gs_scale_f * depth_scale
            xyz_new[kf_related_gs] = xyz_n.squeeze(-1)
            rot_new[kf_related_gs] = rot_n
            scale_new[kf_related_gs] = scale_n

        optimizable_tensors = self.gswrapper.replace_tensor_to_optimizer(xyz_new, "xyz")
        self.gswrapper._xyz = optimizable_tensors["xyz"]
        optimizable_tensors = self.gswrapper.replace_tensor_to_optimizer(rot_new, "rotation")
        self.gswrapper._rotation = optimizable_tensors["rotation"]
        optimizable_tensors = self.gswrapper.replace_tensor_to_optimizer(
            self.gswrapper.scaling_inverse_activation(scale_new), "scaling")
        self.gswrapper._scaling = optimizable_tensors["scaling"]
        torch.cuda.empty_cache()


    def get_proxy_depth(self, v_idx):
        if type(v_idx)==int:
            v_idx = [v_idx]
        elif (type(v_idx)==torch.Tensor) and (v_idx.dim()==0):
            v_idx = [v_idx.item()]
        prior_depth = self.video.prior[v_idx]
        droid_depth = self.kf_cam.droid_depth[v_idx]
        scale,shift = self.kf_cam.fuse_depth_info[v_idx,0],self.kf_cam.fuse_depth_info[v_idx,1]
        proxy_depth = prior_depth * scale[:,None,None] + shift[:,None,None]
        proxy_depth = torch.clamp(proxy_depth, min=0.)
        if self.use_sensor_depth:
            return proxy_depth
        else:
            proxy_depth[droid_depth>0] = droid_depth[droid_depth>0]
            return proxy_depth

    def add_new_cam(self, v_idx, init)->bool:
        c2w, w2c, droid_depth = self.get_pose_and_depth(v_idx)
        if init:
            self.kf_cam = KF_Camera(width=self.W, height=self.H,
                                    c2w=c2w, w2c=w2c,
                                    fovx=self.FoVx, fovy=self.FoVy,
                                    fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                                    droid_depth=droid_depth)
            valid_mask = droid_depth > 0
            tmp_mask = valid_mask.sum(dim=[1, 2]) > self.depth_valid_threshold
            self.dirty_video_idx = self.dirty_video_idx[tmp_mask]

            scale, shift, avg_error = align_scale_and_shift(self.video.prior[self.dirty_video_idx],
                                                            droid_depth[tmp_mask],
                                                            weights=valid_mask[tmp_mask].float())
            self.kf_cam.fuse_depth_info = torch.zeros((droid_depth.shape[0],4)).cuda()
            self.kf_cam.fuse_depth_info[tmp_mask,0] = scale
            self.kf_cam.fuse_depth_info[tmp_mask,1] = shift
            self.kf_cam.fuse_depth_info[:,2] = valid_mask.sum(dim=[1, 2]) / self.HW
            for i in v_idx:
                self.kf_cam.normal[i.item()] = (
                    depth_to_normal(self.kf_cam[i], self.video.prior[[i.item()]]).permute(2, 0, 1))

        else:
            self.kf_cam.append(c2w=c2w, w2c=w2c, droid_depth=droid_depth)
            valid_mask = droid_depth > 0

            scale, shift, avg_error = align_scale_and_shift(self.video.prior[v_idx],
                                                            droid_depth,
                                                            weights=valid_mask.float())
            percent = valid_mask.sum() / self.HW
            fuse_depth_info = torch.tensor([[scale,shift,percent,0.]]).cuda()
            self.kf_cam.fuse_depth_info = torch.cat([self.kf_cam.fuse_depth_info,fuse_depth_info],dim=0)
            self.kf_cam.normal[v_idx] = (
                depth_to_normal(self.kf_cam[v_idx], self.video.prior[[v_idx]]).permute(2, 0, 1))
            if self.deblur is not None:
                self.deblur.add_cam(c2w=c2w)


    def update_cam(self,v_idx):
        c2w, w2c, droid_depth = self.get_pose_and_depth(v_idx)

        valid_mask = droid_depth>0
        percent = valid_mask.sum([1,2])/self.HW
        state = self.kf_cam.fuse_depth_info[v_idx,-2:]

        scale, shift, avg_error = align_scale_and_shift(self.video.prior[v_idx],
                                                        droid_depth,
                                                        weights=valid_mask.float())
        if self.global_ba:
            replace_mask = ((self.kf_cam.fuse_depth_info[v_idx,-1]==1.) *
                            ((percent<0.9) + self.kf_cam.fuse_depth_info[v_idx,-2]>=0.9))
            replace_idx = v_idx[replace_mask]
            unreplce_idx = v_idx[~replace_mask]
            if replace_idx.numel() > 0 :
                self.kf_cam.fuse_depth_info[replace_idx, 0] = scale[replace_mask]
                self.kf_cam.fuse_depth_info[replace_idx, 1] = shift[replace_mask]
                self.kf_cam.fuse_depth_info[replace_idx, 2] = percent[replace_mask]
                self.kf_cam.droid_depth[replace_idx] = droid_depth[replace_mask]
            if unreplce_idx.numel() > 0:
                self.kf_cam.fuse_depth_info[unreplce_idx, 3] = torch.as_tensor(1.)
        else:
            self.kf_cam.fuse_depth_info[v_idx,0] = scale
            self.kf_cam.fuse_depth_info[v_idx,1] = shift
            self.kf_cam.fuse_depth_info[v_idx,2] = percent
            self.kf_cam.fuse_depth_info[v_idx,3] = torch.as_tensor(0.)
            self.kf_cam.droid_depth[v_idx] = droid_depth

        if self.map_deformation:
            self.deformation_kf(v_idxs=v_idx,
                                c2w_n=c2w, w2c_n=w2c)

        if self.deblur is not None:
            self.deblur.update(v_idxs=v_idx, c2w=c2w)

        self.kf_cam.update(v_idxs=v_idx, c2w=c2w, w2c=w2c)

        torch.cuda.empty_cache()


    def run(self,stream:BaseDataset):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.  
        """
        cfg = self.cfg
        self.init = True
        self.kf_rank = 0

        while (1):
            frame_info = self.pipe.recv()
            self.cur_v_idx = frame_info['video_idx']
            is_finished = frame_info['end']

            if is_finished:
                break

            self.global_ba = frame_info['global_ba']

            # update dirty frames
            self.dirty_video_idx = torch.where(self.video.npc_dirty)[0]

            if self.init:
                self.add_new_cam(v_idx=self.dirty_video_idx, init=self.init)
                if self.use_deblur:
                    self.deblur = DeblurModule(self.gs_cfg.deblur_args, self.kf_cam)
                self.gswrapper.update_spatial_lr_scale(self.kf_cam.camera_center)
            else:
                self.add_new_cam(v_idx=self.cur_v_idx, init=self.init)

                self.update_cam(v_idx=self.dirty_video_idx[:-1]) # without current camera

                if self.cur_v_idx % 10 == 0:
                    self.gswrapper.update_spatial_lr_scale(self.kf_cam.camera_center)
                    for param_group in self.gswrapper.optimizer.param_groups:
                        if param_group["name"] == "xyz":
                            param_group['lr'] = self.gswrapper.opt_params.position_lr_init * self.gswrapper.spatial_lr_scale
                            break
            proxy_depth = self.get_proxy_depth(self.dirty_video_idx)
            for i,vidx in enumerate(self.dirty_video_idx):
                self.kf_cam.proxy_depth[vidx.item()] = proxy_depth[i]
            self.video.npc_dirty[self.dirty_video_idx]=False

            self.pipe.send("continue")
            # mapping: 1.add gs  2.optimize gs
            self.mapping_keyframe(self.cur_v_idx)

            torch.cuda.empty_cache()

    def final_refine(self,save_final_pcl=True):
        """
        Final global refinement after mapping all the keyframes
        """
        video_idx = self.video.counter.value-1

        self.dirty_video_idx = torch.arange(len(self.kf_cam)).cuda()
        self.global_ba = True
        self.update_cam(v_idx=self.dirty_video_idx)
        self.dirty_video_idx = self.dirty_video_idx[1:]
        self.global_ba = False
        if self.use_deblur:
            for i in self.deblur.ready_opt_idx:
                self.deblur.n_subframes[i] = 1
            if (self.cfg['deblur_idx'] != '') and self.cfg['only_deblur_test']:
                gt_idxs = torch.full((len(self.kf_cam),), False, dtype=torch.bool)
                gt_idxs[self.cfg['deblur_idx']] = True
                pre_idxs = torch.full((len(self.kf_cam),), False, dtype=torch.bool)
                pre_idxs[self.deblur.deblur_idx] = True

                tp = torch.sum((gt_idxs & pre_idxs)).float()
                fp = torch.sum((~gt_idxs & pre_idxs)).float()
                fn = torch.sum((gt_idxs & ~pre_idxs)).float()
                tn = torch.sum((~gt_idxs & ~pre_idxs)).float()

                # 计算准确率 (Accuracy): (TP + TN) / (TP + TN + FP + FN)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                # 计算召回率 (Recall): TP / (TP + FN)
                recall = tp / (tp + fn)
                output_str = f"deblur idx: {self.deblur.deblur_idx}\n"
                output_str += f"deblur accuracy: {accuracy*100:.2f} %, recall: {recall*100:.2f} % \n"
                log_path = f"{self.output}/deblur_test.txt"
                with open(log_path, 'w+') as fp:
                    fp.write(output_str)
                return
        self.printer.print('Start final refinement.',FontColor.MAPPER)
        self.iteration = 0

        self.gswrapper.opt_params.position_lr_max_steps = self.cfg['final_refine']['max_steps']
        self.gswrapper.opt_params.densify_from_iter = self.cfg['final_refine']['densify_from_iter']
        self.gswrapper.opt_params.densification_interval = self.cfg['final_refine']['densification_interval']
        self.gswrapper.opt_params.opacity_reset_interval = self.cfg['final_refine']['opacity_reset_interval']
        self.gswrapper.opt_params.densify_until_iter = self.cfg['final_refine']['densify_until_iter']
        self.rgb_loss,self.normal_loss,self.normal_prior_loss,self.depth_loss = 0.,0.,0.,0.
        self.gswrapper.update_spatial_lr_scale(self.kf_cam.camera_center)

        if self.use_exposure:
            self.exposure = Exposure(kf_cam=self.kf_cam)
        self.opt_idx = self.dirty_video_idx

        opt_kf_num = len(self.opt_idx)
        self.opt_cam = {}
        for idx in self.opt_idx:
            self.opt_cam[idx.item()] = self.kf_cam[idx]
        kf_stack = None
        self.gswrapper.reset_opacity()
        if self.cfg['dataset']=='replica' or self.gs_cfg.pipe_param.depth_ratio_shift:
            depth_ratio_decay = self.gswrapper.pipeline_params.depth_ratio/(self.cfg['final_refine']['max_steps'])
        else:
            depth_ratio_decay = 0.
        for iteration in tqdm(range(0, self.cfg['final_refine']['max_steps'])):
            self.gswrapper.pipeline_params.depth_ratio -= depth_ratio_decay
            if not kf_stack:
                kf_stack = list(range(opt_kf_num))
                random.shuffle(kf_stack)

            kf_indice = kf_stack.pop()
            loss_dict = self.update_map_once(self.opt_idx[kf_indice].item(),
                                             iteration=iteration,
                                             deblur=self.use_deblur,
                                             final_refine=True)
            self.log_loss(loss_dict, prefix='refine_')


        if save_final_pcl:
            self.save_pcd('final')
            self.gswrapper.save_ply_3dgs(path=f"{self.output}/point_cloud.ply")

            self.printer.print('Saved Gaussians as 3D-GS format.',FontColor.INFO)


    @ torch.no_grad()
    def save_rendered_image(self,idx,dir_name:str,prefix:str,deblur=False):
        cam: MiniCam = self.kf_cam[idx]
        gt_image = self.video.images[idx]
        render_pkg = render(cam, self.gswrapper,
                            self.gswrapper.pipeline_params, self.gswrapper.bg_color)
        surf_depth = render_pkg['surf_depth']
        image = render_pkg["render"]
        surf_normal = render_pkg["surf_normal"]
        depth_max = surf_depth.max()
        depth_scaled = ((surf_depth - 0) / (depth_max - 0) * 255)[0].int().to('cpu')
        depth_cv = cv2.applyColorMap(np.uint8(depth_scaled), 2)
        save_depth = torch.from_numpy(depth_cv).to('cuda').permute(2, 0, 1) / 255
        save_normal = (surf_normal +1)/2
        if deblur:
            render_pkg_subframes = []
            subframes = self.deblur.provide_subframes(idx, cam)
            render_pkg = render(subframes[0], self.gswrapper, self.gswrapper.pipeline_params, self.gswrapper.bg_color)
            image_0 = render_pkg["render"]
            render_pkg_subframes.append(image_0)
            subframes.pop(0)
            for frame in subframes:
                render_pkg_sub = render(frame, self.gswrapper, self.gswrapper.pipeline_params, self.gswrapper.bg_color)
                render_pkg_subframes.append(render_pkg_sub['render'])

            render_subframes = torch.stack(render_pkg_subframes)
            image = render_subframes.mean(dim=0)
            save_deblur = torch.cat([image_0,image],dim=-1)
            save_pic = torch.cat([gt_image,image,image0,save_normal, save_depth], dim=1)
        else:
            save_pic = torch.cat([gt_image,image,save_normal,save_depth], dim=1)

        output_dir = f'{self.output}/{dir_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torchvision.utils.save_image(save_pic,f"{output_dir}/{prefix}{idx}.png")

    def log_loss(self,loss_dict,prefix,interval=10):
        with torch.no_grad():
            # Progress bar
            self.rgb_loss = 0.4 * loss_dict[1] + 0.6 * self.rgb_loss
            self.normal_loss = 0.4 * loss_dict[2] + 0.6 * self.normal_loss
            self.normal_prior_loss = 0.4 * loss_dict[3] + 0.6 * self.normal_prior_loss
            self.depth_loss = 0.4 * loss_dict[4] + 0.6 * self.depth_loss
            if self.iteration % interval == 0:
                for param_group in self.gswrapper.optimizer.param_groups:
                    if param_group["name"] == "xyz":
                        lr = param_group["lr"]
                        break

                loss_dict = {
                    f'{prefix}rgb_loss': self.rgb_loss,
                    f'{prefix}normal_loss': self.normal_loss,
                    f'{prefix}normal_prior_loss': self.normal_prior_loss,
                    f'{prefix}depth_loss': self.depth_loss,
                    f'{prefix}xyz_lr': lr,
                    'gs_num': int(self.gswrapper.gs_num)
                }

                if self.logger:
                    self.logger.log(loss_dict)

    def set_param_groups_grad(self,optimizer:torch.optim.Optimizer,require_grad:bool):
        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = require_grad

    def save_pcd(self, file_name):
        from plyfile import PlyData, PlyElement
        points = self.gswrapper.get_xyz.detach().cpu().numpy()  # 这里只是随机生成一些数据作为例子

        # 创建一个结构化的numpy数组来存储点的信息
        vertex = np.empty(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        # 将xyz值填充到结构化数组中
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]

        # 创建PlyElement对象
        el = PlyElement.describe(vertex, 'vertex')

        # 写入PLY文件
        PlyData([el]).write(f"{self.output}/{file_name}.ply")

Mapper.eval_kf_imgs = eval_kf_imgs

