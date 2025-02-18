import torch
from torch import nn
from src.gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixShift
import open3d as o3d
class KF_Camera:
    def __init__(self,
                 width, height, c2w, w2c,
                 fovx, fovy, fx, fy, cx, cy,
                 droid_depth=None,
                 gt_alpha_mask=None, znear=0.01, zfar=100.):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width, height, self.fx, self.fy, self.cx, self.cy )
        self.intrinsic_matrix = torch.tensor(
            [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.0]]).float().cuda()
        self.znear = znear
        self.zfar = zfar
        self.droid_depth = droid_depth
        self.w2c = w2c
        self.c2w = c2w
        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask.to('cuda')
        else:
            self.gt_alpha_mask = None
        self.world_view_transform = self.w2c.transpose(1, 2)
        if abs(cx *2 - width) > 2 or abs(cy *2 - height) > 2:
            self.projection_matrix = getProjectionMatrixShift(self.znear, self.zfar,
                                                              fx, fy, cx, cy,
                                                              width, height, fovx, fovy).transpose(0, 1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.bmm(self.projection_matrix[None].expand(w2c.shape[0],-1,-1)))
        self.camera_center = c2w[:,:3,3]
        self.fuse_depth_info = None               #[n,4]  (scale,shift,percent,global-ba:(0.1.))
        self.proxy_depth = {}
        self.normal ={}

    def update(self,v_idxs,c2w,w2c):
        self.w2c[v_idxs] = w2c
        self.c2w[v_idxs] = c2w
        self.world_view_transform[v_idxs] = w2c.transpose(1, 2)
        self.full_proj_transform[v_idxs] = (
            self.world_view_transform[v_idxs].bmm(self.projection_matrix[None].expand(w2c.shape[0],-1,-1)))
        self.camera_center[v_idxs] = c2w[:,:3,3]


    def __getitem__(self, v_idx):
        if self.gt_alpha_mask is not None:
            gt_alpha_mask = self.gt_alpha_mask[v_idx]
        else:
            gt_alpha_mask = None
        cam = MiniCam(width=self.image_width, height=self.image_height,
                      fovx=self.FoVx, fovy=self.FoVy,
                      world_view_transform=self.world_view_transform[v_idx],
                      full_proj_transform=self.full_proj_transform[v_idx],
                      camera_center=self.camera_center[v_idx],
                      intrinsic_matrix=self.intrinsic_matrix,
                      c2w=self.c2w[v_idx],
                      projection_matrix = self.projection_matrix,
                      gt_alpha_mask = gt_alpha_mask)
        return cam

    def __len__(self):
        return self.c2w.shape[0]

    def append(self,w2c,c2w,droid_depth):
        self.w2c = torch.cat([self.w2c, w2c[None]],dim=0)
        self.c2w = torch.cat([self.c2w, c2w[None]], dim=0)
        self.droid_depth = torch.cat([self.droid_depth, droid_depth[None]], dim=0)
        world_view_transform = w2c.transpose(0, 1)[None]
        self.world_view_transform = torch.cat([self.world_view_transform, world_view_transform],dim=0)

        full_proj_transform = world_view_transform.bmm(self.projection_matrix[None])
        self.full_proj_transform = torch.cat([self.full_proj_transform, full_proj_transform],dim=0)
        self.camera_center = torch.cat([self.camera_center,c2w[:3, 3][None]],dim=0)


class MiniCam:
    def __init__(self, width, height, fovy, fovx,
                 world_view_transform,full_proj_transform,camera_center,intrinsic_matrix,c2w,projection_matrix,gt_alpha_mask,
                 znear=0.01,zfar=100.):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.intrinsic_matrix = intrinsic_matrix
        self.c2w = c2w
        self.projection_matrix = projection_matrix
        self.gt_alpha_mask =gt_alpha_mask
        self.fx = intrinsic_matrix[0,0].item()
        self.fy = intrinsic_matrix[1,1].item()
        self.cx = intrinsic_matrix[0,2].item()
        self.cy = intrinsic_matrix[1,2].item()

    def align_pose_for_mesh(self,c2w):
        # only for mesh
        self.c2w = c2w.to(torch.float32)
        self.camera_center = self.c2w[:3, 3]
        self.world_view_transform = self.c2w.inverse().transpose(0,1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix