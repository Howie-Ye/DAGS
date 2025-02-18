import os
import shutil
import torch
import cv2
import torchvision.utils
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.utils.datasets import load_mono_depth
from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
from pytorch_msssim import ms_ssim
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
from src.gaussian_splatting.render import render
import open3d as o3d
import trimesh
from evaluate_3d_reconstruction import run_evaluation
from .eval_recon import calc_3d_metric
import traceback
import numpy as np
from tqdm import tqdm
def clean_mesh(mesh):
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(
        mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 100
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {old_idx: vertex_count +
                         new_idx for new_idx, old_idx in enumerate(component)}
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[np.any(
            np.isin(mesh_tri.faces, component), axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.remove_degenerate_faces()
    cleaned_mesh_tri.remove_duplicate_faces()
    print(
        f'Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}')

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces)
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64))

    return cleaned_mesh
@ torch.no_grad()
def eval_kf_imgs(self,stream,traj_est,prefix='',generate_mesh=False):
    # re-render frames at the end for meshing
    self.printer.print('Starting re-rendering keyframes...',FontColor.EVAL)
    frame_cnt = 0
    psnr_sum, ssim_sum, lpips_sum, l1_depth_sum,l1_depth_sum_mean = 0,0,0,0,0

    os.makedirs(f'{self.output}/{prefix}rerendered_keyframe_image', exist_ok=True)
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', normalize=True).to(self.device)
    if generate_mesh:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    try:
        for id in tqdm(self.dirty_video_idx):
            timestamp = self.video.timestamp[id].int()
            gt_depth = stream[timestamp][2].cuda()
            cam = self.kf_cam[id]
            gt_image = self.video.images[id]
            if self.use_deblur:
                render_pkg_subframes = []
                subframes = self.deblur.provide_subframes(id, cam)

                render_pkg = render(subframes[0], self.gswrapper, self.gswrapper.pipeline_params,
                                    self.gswrapper.bg_color)
                image0, viewspace_point_tensor, visibility_filter, radii = \
                    render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg[
                        "radii"]
                render_pkg_subframes.append(image0)
                subframes.pop(0)
                for frame in subframes:
                    render_pkg_sub = render(frame, self.gswrapper, self.gswrapper.pipeline_params,
                                            self.gswrapper.bg_color)
                    render_pkg_subframes.append(render_pkg_sub['render'])

                render_subframes = torch.stack(render_pkg_subframes)
                image = render_subframes.mean(dim=0)
            else:
                render_pkg = render(cam, self.gswrapper, self.gswrapper.pipeline_params,
                                    self.gswrapper.bg_color)
                image, viewspace_point_tensor, visibility_filter, radii = \
                    render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg[
                        "radii"]
                image0 = gt_image

            if self.use_exposure and self.exposure is not None:
                exp_scale, exp_shift = self.exposure.get_scale_shift(id)
                image = image * exp_scale + exp_shift

            depth = render_pkg['surf_depth']
            mask = (gt_depth>0) * (depth[0]>0)
            depth_scale = (gt_depth / depth[0])[mask].mean()
            depth *=  depth_scale
            mse_loss = torch.nn.functional.mse_loss(gt_image, image)
            psnr_frame = -10. * torch.log10(mse_loss)
            ssim_value = ms_ssim(gt_image[None],image[None],data_range=1.0, size_average=True)
            lpips_value = cal_lpips(gt_image[None],torch.clamp(image[None], 0.0, 1.0))

            l1_depth = (gt_depth - depth[0])[mask].abs().mean()
            psnr_sum += psnr_frame
            ssim_sum += ssim_value
            lpips_sum += lpips_value
            l1_depth_sum += l1_depth

            if id % 10 == 0:
                # save imgs of kfs
                save_rgb = torch.cat([image,gt_image], dim=-1)
                depth_gt = torch.cat([depth,gt_depth[None]],dim=-1)
                prior_depth = self.video.prior[id]
                scale, shift = self.kf_cam.fuse_depth_info[id, 0], self.kf_cam.fuse_depth_info[id, 1]
                prior_depth_scale = (prior_depth * scale + shift)*depth_scale
                depth_sup_0 = torch.cat([prior_depth_scale, prior_depth_scale], dim=-1)[None]
                depth_sup_1 = torch.cat([self.kf_cam.droid_depth[id]*depth_scale, self.get_proxy_depth(v_idx=id)[0]*depth_scale], dim=-1)[None]
                depth_gt = torch.cat([depth_sup_0,depth_sup_1,depth_gt],dim=1)
                depth_max = depth_gt.max()
                depth_scaled = ((depth_gt - 0) / (depth_max - 0) * 255)[0].int().to('cpu')
                depth_cv = cv2.applyColorMap(np.uint8(depth_scaled), 2)
                save_depth = torch.from_numpy(depth_cv).to('cuda').permute(2, 0, 1) / 255

                prior_normal = (self.kf_cam.normal[id.item()]+1)/2
                save_depth[:,:prior_normal.shape[1],:prior_normal.shape[2]]=prior_normal
                normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
                save_normal = (normal + 1) / 2
                save_img0_normal = torch.cat([image0,save_normal],dim=-1)
                save_img = torch.cat([save_rgb,save_img0_normal,save_depth],dim=-2)
                torchvision.utils.save_image(save_img,f'{self.output}/{prefix}rerendered_keyframe_image/frame_{id:05d}.png')

            if generate_mesh:
                depth[gt_depth.unsqueeze(0) == 0] += 30.
                depth_o3d = np.ascontiguousarray(depth.permute(1, 2, 0).cpu().numpy().astype(np.float32))
                depth_o3d = o3d.geometry.Image(depth_o3d)
                color_o3d = np.ascontiguousarray(
                    (np.clip(image.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8))
                color_o3d = o3d.geometry.Image(color_o3d)

                w2c_o3d = np.linalg.inv(traj_est[id])  # convert from c2w to w2c

                fx = cam.fx
                fy = cam.fy
                cx = cam.cx
                cy = cam.cy
                W = depth.shape[-1]
                H = depth.shape[1]
                intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d,
                    depth_o3d,
                    depth_scale=1.0,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                # use gt pose for debugging
                # w2c_o3d = torch.linalg.inv(pose).cpu().numpy() @ dataset.w2c_first_pose
                volume.integrate(rgbd, intrinsic, w2c_o3d)

            frame_cnt += 1

        avg_psnr = psnr_sum / frame_cnt
        avg_ssim = ssim_sum / frame_cnt
        avg_lpips = lpips_sum / frame_cnt
        avg_l1_depth = l1_depth_sum / frame_cnt
        self.printer.print(f'avg_msssim: {avg_ssim}',FontColor.EVAL)
        self.printer.print(f'avg_psnr: {avg_psnr}',FontColor.EVAL)
        self.printer.print(f'avg_lpips: {avg_lpips}',FontColor.EVAL)
        self.printer.print(f'avg_l1_depth: {avg_l1_depth}', FontColor.EVAL)

        out_path=f'{self.output}/logs/metrics_render_kf.txt'
        output_str = f"avg_ssim: {avg_ssim}\n"
        output_str += f"avg_psnr: {avg_psnr}\n"
        output_str += f"avg_lpips: {avg_lpips}\n"
        output_str += f"avg_l1_depth: {avg_l1_depth}\n"
        output_str += f"gaussian number: {self.gswrapper.gs_num}\n"
        if self.use_deblur:
            output_str += f"deblur idx: {self.deblur.deblur_idx}\n"
            if self.cfg['deblur_idx'] != '':
                gt_idxs = torch.full((len(self.kf_cam),), False, dtype=torch.bool)
                gt_idxs[self.cfg['deblur_idx']] = True
                pre_idxs = torch.full((len(self.kf_cam),), False, dtype=torch.bool)
                pre_idxs[self.deblur.deblur_idx] = True

                tp = torch.sum((gt_idxs & pre_idxs)).float()  # 预测正确的正样本
                fp = torch.sum((~gt_idxs & pre_idxs)).float()  # 错误预测为正的样本
                fn = torch.sum((gt_idxs & ~pre_idxs)).float()  # 漏掉的正样本
                tn = torch.sum((~gt_idxs & ~pre_idxs)).float()  # 正确预测的负样本

                # 计算准确率 (Accuracy): (TP + TN) / (TP + TN + FP + FN)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                # 计算召回率 (Recall): TP / (TP + FN)
                recall = tp / (tp + fn)
                output_str += f"deblur accuracy: {accuracy*100:.2f} %, recall: {recall*100:.2f} % \n"

        output_str += f"###############\n"
        opt = self.gswrapper.opt_params
        output_str += f'depth_ratio = {self.gswrapper.pipeline_params.depth_ratio}\n'
        output_str += f'lambda_l1_rgb_threshold = {opt.lambda_l1_rgb_threshold}\n'
        output_str += f'lambda_normal_prior = {opt.lambda_normal_prior}\n'
        output_str += f'lambda_depth = {opt.lambda_depth}\n'
        output_str += f'opacity_cull = {opt.opacity_cull}\n'

        with open(out_path, 'w+') as fp:
            fp.write(output_str)

        if generate_mesh:
            # Mesh the final volumetric model
            scene_name = self.output.split('/')[-2]
            mesh_name = f'{scene_name}_kf.ply'
            os.makedirs(f'{self.output}/mesh', exist_ok=True)
            mesh_out_file = f'{self.output}/mesh/{mesh_name}'
            o3d_mesh = volume.extract_triangle_mesh()
            o3d.io.write_triangle_mesh(f'{self.output}/mesh/{scene_name}_before_clean.ply', o3d_mesh)
            o3d_mesh = clean_mesh(o3d_mesh)
            o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
            print('Meshing finished.')

            # evaluate the mesh
            try:
                pred_ply = mesh_out_file.split('/')[-1]
                last_slash_index = mesh_out_file.rindex('/')
                path_to_pred_ply = mesh_out_file[:last_slash_index]
                gt_mesh = self.cfg['meshing']['gt_mesh_path']
                result_recon = {}
                result_3d = run_evaluation(pred_ply, path_to_pred_ply, "mesh",
                                           distance_thresh=0.05, full_path_to_gt_ply=gt_mesh, icp_align=True)
                result_recon = result_recon | result_3d
                result_3d_nice = calc_3d_metric(rec_meshfile=path_to_pred_ply + "/" + pred_ply,
                                                gt_meshfile=gt_mesh)
                result_recon = result_recon | result_3d_nice
                print(f"3D Mesh evaluation: {result_recon}")
                output_str_recon = ""
                for k, v in result_recon.items():
                    output_str_recon += f"{k}: {v}\n"
                out_path = f'{self.output}/logs/metrics_mesh.txt'
                with open(out_path, 'w+') as fp:
                    fp.write(output_str_recon)

            except Exception as e:
                traceback.print_exception(e)


    except Exception as e:
        traceback.print_exception(e)
        self.printer.print('Rerendering frames failed.',FontColor.ERROR)
    self.printer.print(f'Finished rendering {frame_cnt} frames.',FontColor.EVAL)


