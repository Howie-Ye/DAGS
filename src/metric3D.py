import torch
import torch.nn.functional as F

class Metric3D:
    def __init__(self,cfg):
        repo = cfg['prior_estimator']['model']
        m = cfg['prior_estimator']['type']
        hubconf_dir = cfg['prior_estimator']['hubconf_dir']
        try:
            self.model = torch.hub._load_local(hubconf_dir, m, pretrain=True)
        except:
            self.model = torch.hub.load(repo, m, pretrain=True)
        self.model.cuda().eval()
        self.gt_depth_scale = 256.0
        self.input_size = (616, 1064)  # for vit model   here we use vit
        # for convnext self.input_size = model(544, 1216)
        self.H_out = cfg['cam']['H_out']
        self.W_out = cfg['cam']['W_out']

        self.scale = min(self.input_size[0] / self.H_out, self.input_size[1] / self.W_out)
        self.s_h, self.s_w = int(self.H_out * self.scale), int(self.W_out * self.scale)
        pad_h = self.input_size[0] - self.s_h
        pad_w = self.input_size[1] - self.s_w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        # remember to scale intrinsic, hold depth

        intrinsic = [self.scale * cfg['cam']['fx'],
                     self.scale * cfg['cam']['fy'],
                     self.scale * cfg['cam']['cx'],
                     self.scale * cfg['cam']['cy']]

        # padding to input_size
        self.padding = torch.tensor([0.485, 0.456, 0.406]).cuda()

        self.pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        self.canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera

    def pred(self,img):
        # scaling and padding to input size in Metric3D
        rgb = F.interpolate(img[0], size=(self.s_h,self.s_w), mode='bilinear', align_corners=False)[0]  # [c,h,w]
        rgb_pad = torch.ones((*self.input_size,3)).cuda() * self.padding   # [h,w,c]
        rgb_pad[self.pad_info[0]:rgb_pad.shape[0]-self.pad_info[1],
                self.pad_info[2]:rgb_pad.shape[1]-self.pad_info[3]] = rgb.permute(1,2,0)
        rgb_input = rgb_pad.permute(2,0,1)[None]
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({'input': rgb_input})

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[self.pad_info[0]: pred_depth.shape[0] - self.pad_info[1],
                                self.pad_info[2]: pred_depth.shape[1] - self.pad_info[3]]

        # upsample to original size
        pred_depth = F.interpolate(pred_depth[None, None, :, :], (self.H_out,self.W_out),
                                                     mode='bilinear').squeeze()
        ###################### canonical camera space ######################

        #### de-canonical transform
        pred_depth = pred_depth * self.canonical_to_real_scale  # now the depth is metric
        depth_max = pred_depth.mean() + 3 * pred_depth.std()
        pred_depth = torch.clamp(pred_depth, 0, depth_max)
        '''
        # normal
        pred_normal = output_dict['prediction_normal'].squeeze()
        # un pad and resize to some size if needed
        pred_normal = pred_normal[:,
                                  self.pad_info[0]: pred_normal.shape[1] - self.pad_info[1],
                                  self.pad_info[2]: pred_normal.shape[2] - self.pad_info[3]]
        pred_normal = torch.nn.functional.interpolate(pred_normal[None], (self.H_out, self.W_out),
                                                     mode='bilinear').squeeze()
        pred_normal[:3] = F.normalize(pred_normal[:3],dim=0)
        # scale the confidence
        pred_normal[-1] = 1 - pred_normal[-1]/30    # max kappa = 30
        '''
        return pred_depth


