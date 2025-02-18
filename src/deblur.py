from src import (os, sys, math, random, time,
    torch, nn, torchvision, F, o3d,
    np, cv2,
    Console, List, Tensor,
    rearrange, repeat, reduce)
from src.utils.p3d_functions import quaternion_apply,quaternion_to_matrix,quaternion_invert,quaternion_multiply,matrix_to_quaternion
from src.gaussian_splatting.scene.gaussian_model import GaussianModel
from src.gaussian_splatting.scene.dataset_readers import CameraInfo
from src.KF_Camera import KF_Camera
from src.gaussian_splatting.scene.cameras import MiniCam

from src.gaussian_splatting.utils.general_utils import inverse_sigmoid
from configs.gs_configs import DeblurArgs

class DeblurModule:
    def __init__(self, args, kf_cam:KF_Camera):
        self.max_subframes_num = args.num_subframes
        self.n_subframes = []
        self.lr_rot = args.lr_rot
        self.lr_trans = args.lr_trans
        self.thres_trans = args.thres_trans
        self.thres_rot = args.thres_rot
        self.deblur_idx = []
        self.kf_cam = kf_cam
        self.projection_matrix = kf_cam.projection_matrix
        c2w = kf_cam.c2w
        self.scale_down_rate = args.scale_down_rate
        # Initial Parameters
        self._set_initial_parameters(c2w[:,:3,:3], c2w[:,:3,3])
        self.n_subframes = [self.max_subframes_num] * len(kf_cam)
        self.ready_opt_idx = list(range(len(kf_cam)))
    def _set_initial_parameters(self, rotations, translations):
        rot_params = matrix_to_quaternion(rotations)  # [n,4]
        rot_params = rot_params[:,None,:].repeat(1, self.max_subframes_num, 1).contiguous()  #[n,c,4]
        self._rot = nn.Parameter(rot_params.requires_grad_(True))
        trans_params = translations[:,None,:].repeat(1, self.max_subframes_num, 1).contiguous()  #[n,c,3]
        self._trans = nn.Parameter(trans_params.requires_grad_(True))
        l = [
            {'params': self._rot, 'lr': self.lr_rot, 'name': 'rot'},
            {'params': self._trans, 'lr': self.lr_trans, 'name': 'trans'}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def provide_world_view_matrix(self,idx):
        rot_opt, trans_opt = self._sample_c2w(idx)  # c2w
        c2w = torch.eye(4)[None].expand(self.n_subframes[idx], -1, -1).to(rot_opt.device)
        c2w[:, :3, :3] = rot_opt
        c2w[:, :3, 3] = trans_opt.squeeze(-1)
        r_w2c = rot_opt.transpose(1, 2)
        w2c = torch.eye(4)[None].expand(self.n_subframes[idx], -1, -1).to(rot_opt.device)
        t_w2c = - r_w2c @ trans_opt[..., None]
        w2c[:, :3, :3] = r_w2c
        w2c[:, :3, 3] = t_w2c.squeeze(-1)
        return c2w,w2c

    def get_c2ws(self,idxs,avg:bool=False):
        c2w_qua = F.normalize(self._rot[idxs].mean(dim=1),dim=-1)  if avg else self._rot[idxs, 0]
        c2w_t = self._trans[idxs].mean(dim=1) if avg else self._rot[idxs, 0]
        c2ws = torch.eye(4)[None].expand(c2w_qua.shape[0],-1,-1).cuda()
        c2ws[:, :3, :3] = quaternion_to_matrix(c2w_qua)
        c2ws[:, :3, 3] = c2w_t.squeeze(-1)
        return c2ws

    def provide_subframes(self,idx,cam:MiniCam) -> List[MiniCam]:
        c2w,w2c = self.provide_world_view_matrix(idx)
        world_view_transform = w2c.transpose(1,2)
        full_proj_transform = world_view_transform @ self.projection_matrix
        minicam_list = []
        for i in range(self.n_subframes[idx]):
            minicam_list.append(
                MiniCam(width=cam.image_width, height=cam.image_height,
                        fovx=cam.FoVx, fovy=cam.FoVy,
                        world_view_transform=world_view_transform[i],
                        full_proj_transform=full_proj_transform[i],
                        camera_center=c2w[i,:3,3],
                        intrinsic_matrix=cam.intrinsic_matrix,
                        c2w=c2w[i],
                        gt_alpha_mask=cam.gt_alpha_mask)
            )
        return minicam_list

    def add_cam(self,c2w):
        # add new cam pose
        rotation = c2w[:3,:3]                                            #[3,3]
        translation = c2w[:3,3][None]                                    #[1,3]
        rot_quat = matrix_to_quaternion(rotation)[None]               #[1,4]
        new_rot = rot_quat[:, None, :].repeat(1, self.max_subframes_num, 1)  # [1,c,d]

        new_trans = translation[:, None, :].repeat(1, self.max_subframes_num, 1)  # [1,c,d]

        cat_dict = {'rot':new_rot,
                    'trans':new_trans}
        optimizable_tensors = self._cat_tensors_to_optimizer(cat_dict)
        self._rot = optimizable_tensors['rot']
        self._trans = optimizable_tensors['trans']
        self.n_subframes.append(self.max_subframes_num)
        self.ready_opt_idx.append(len(self.n_subframes) - 1)

    def _cat_tensors_to_optimizer(self,tensors_dict):
        optimizable_tensors = {}
        dict_key = list(tensors_dict.keys())
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group['name'] not in dict_key:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"],torch.zeros_like(extension_tensor)),dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def update(self,v_idxs,c2w):
        # update cam after tracking
        # 1. get w2c_f
        c2w_qua = self._rot[v_idxs,0]
        c2w_t = self._trans[v_idxs,0]
        w2c_qua = quaternion_invert(c2w_qua)
        w2c_t = -quaternion_apply(w2c_qua, c2w_t)
        w2c_f = torch.eye(4)[None].expand(v_idxs.shape[0],-1,-1).cuda()
        w2c_f[:,:3,:3] = quaternion_to_matrix(w2c_qua)
        w2c_f[:,:3,3] = w2c_t
        transform_matrix = w2c_f.bmm(c2w)
        tran_qua = matrix_to_quaternion(transform_matrix[:,:3,:3])    # [n,4]
        tran_qua = tran_qua[:,None,:].repeat(1,self.max_subframes_num,1)  # [n,m,4]
        tran_t = transform_matrix[:,:3,3][:,None,:].repeat(1,self.max_subframes_num,1)      # [n,m,3]

        rot = quaternion_multiply(F.normalize(self._rot[v_idxs],dim=-1),tran_qua)
        t = quaternion_apply(self._rot[v_idxs],tran_t) + self._trans[v_idxs]

        new_rot = self._rot.clone()
        new_trans = self._trans.clone()
        new_rot[v_idxs] = rot
        new_trans[v_idxs] = t
        optimizable_tensor = self._replace_tensor_to_optimizer(new_rot, 'rot')
        self._rot = optimizable_tensor['rot']
        optimizable_tensor = self._replace_tensor_to_optimizer(new_trans, 'trans')
        self._trans = optimizable_tensor['trans']

    def _sample_c2w(self, idx):
        """

        RETURNS
        -------
        c2w_rotations: Tensor of shape [num_subframes, 3, 3]
        c2w_translations: Tensor of shape [num_subframes, 3]
        """

        rot_quaternion = self._rot[idx,:self.n_subframes[idx]]  # [f,4]
        rot_quaternion = F.normalize(rot_quaternion,dim=-1)  # [f,4]
        c2w_rotations = quaternion_to_matrix(rot_quaternion)  # [f,3,3]
        c2w_translations = self._trans[idx,:self.n_subframes[idx]]  # [f,3]

        return c2w_rotations, c2w_translations

    def change_subframe_num(self, idx, note_idx:bool):
        qua_0 = F.normalize(self._rot[idx,[0]])
        R = quaternion_multiply(self._rot[idx,1:], quaternion_invert(qua_0))
        trac = quaternion_to_matrix(R).diagonal(offset=0, dim1=1, dim2=2).sum(dim=-1)
        diff_rot = 1 - 0.5 * (trac - 1)
        diff_rot_bool = torch.any(diff_rot > self.thres_rot)
        trans_0 = self._trans[idx,[0]]
        diff_trans = (self._trans[idx,1:] - trans_0).norm(dim=-1)
        diff_trans_bool = torch.any(diff_trans > self.thres_trans)

        if not (diff_rot_bool or diff_trans_bool):
            self.n_subframes[idx] = 1
            return True
        else:
            if note_idx:
                self.deblur_idx.append(idx)
            return False




    @property
    def device(self):
        return self._rot.device

    def is_optimizing(self):
        return self._rot.requires_grad


    def save(self, state_dict_path: str):
        """
        Save camera motion parameters.
        """

        assert (state_dict_path.endswith(".pth"))

        sdict = {"rot": self._rot.state_dict(),
                 "trans": self._trans.state_dict()}

        torch.save(sdict, state_dict_path)
        print("[SAVED] Camera Motion")

    def load(self, path: str):
        """
        Load camera motion parameters.
        """

        if path.endswith(".pth"):
            state_dict_path = path
        else:
            state_dict_path = os.path.join(path, "cm.pth")

        sdict = torch.load(state_dict_path)
        self._rot.load_state_dict(sdict["rot"])
        self._trans.load_state_dict(sdict["trans"])
        print("[LOADED] Camera Motion")