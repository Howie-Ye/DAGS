from src import (os, sys, math, random, time,
                 torch, nn, torchvision, F, o3d,
                 np, cv2,
                 Console, List, Tensor, Optional,
                 rearrange, repeat, reduce)

from collections import OrderedDict
import torch.multiprocessing as mp
from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from src.backend import Backend
import wandb


class SLAM:
    def __init__(self, cfg, gs_cfg, stream: BaseDataset, output_dir):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.gs_cfg = gs_cfg
        self.device = cfg['device']
        self.verbose: bool = cfg['verbose']
        self.only_tracking: bool = cfg['only_tracking']
        self.logger = None
        self.wandb = self.cfg['wandb']
        if self.wandb:
            scene_name = self.cfg["scene"]
            dataset_name = self.cfg["dataset"]
            self.logger = \
                wandb.init(resume="allow", config=self.cfg, project=self.cfg["setting"], group=f'{dataset_name}',
                           name=scene_name,
                           settings=wandb.Settings(code_dir="."), dir=self.cfg["wandb_folder"],
                           tags=[scene_name])
            self.logger.log_code(".")
        self.output = output_dir
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(f'{self.output}/logs/', exist_ok=True)

        self.H, self.W, \
            self.fx, self.fy, \
            self.cx, self.cy = update_cam(cfg)

        self.printer = Printer(len(stream))  # use an additional process for printing all the info

        # load pretrained droid-net
        self.droid_net: DroidNet = DroidNet()
        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        # note multiprocess info
        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        self.video = DepthVideo(cfg, self.printer)
        self.ba = Backend(self.droid_net, self.video, self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net=self.droid_net, video=self.video,
                                                printer=self.printer, device=self.device)

        self.stream: BaseDataset = stream

        self.tracker: Optional[Tracker] = None
        self.mapper: Optional[Mapper] = None

    def load_pretrained(self, cfg):
        droid_pretrained = cfg['tracking']['pretrained']
        state_dict = OrderedDict([
            (k.replace('module.', ''), v) for (k, v) in torch.load(droid_pretrained).items()
        ])
        state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
        state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
        state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
        state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(f'Load droid pretrained checkpiont from {droid_pretrained}!', FontColor.INFO)

    def tracking(self, pipe):
        self.tracker = Tracker(self, pipe)
        self.printer.print('Tracking Triggered!', FontColor.TRACKER)
        self.all_trigered += 1
        while (self.all_trigered < self.num_running_thread):
            pass
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print('Tracking Done!', FontColor.TRACKER)
        if self.only_tracking:
            self.terminate()

    def mapping(self, pipe):
        if self.only_tracking:
            self.all_trigered += 1
            return
        self.mapper = Mapper(self, pipe)
        self.printer.print('Mapping Triggered!', FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while (self.all_trigered < self.num_running_thread):
            pass
        self.mapper.run(self.stream)
        self.printer.print('Mapping Done!', FontColor.MAPPER)
        self.terminate()

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)
        self.ba = Backend(self.droid_net, self.video, self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

    def terminate(self):
        """ fill poses for non-keyframe images and evaluate """

        if self.cfg['tracking']['backend']['final_ba']:
            self.backend()
        if not self.only_tracking:
            self.mapper.final_refine(save_final_pcl=True)
            if self.cfg['only_deblur_test']:
                return
        self.video.save_video(f"{self.output}/video.npz")
        torch.cuda.empty_cache()

        do_evaluation = True
        if do_evaluation:

            try:
                ate_statistics, traj_scale, r_a, t_a, traj_est = kf_traj_eval(
                    f"{self.output}/video.npz",
                    f"{self.output}/traj",
                    "kf_traj", self.stream, self.logger, self.printer)
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

            try:
                full_traj, full_traj_aligned, full_traj_ref = full_traj_eval(self.traj_filler,
                                                                             f"{self.output}/traj",
                                                                             "full_traj",
                                                                             self.stream, self.logger, self.printer)
                np.save(f"{self.output}/traj/full_traj_aligned.npy", full_traj_aligned.poses_se3)
                np.save(f"{self.output}/traj/full_traj_gt.npy", full_traj_ref.poses_se3)
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)
            torch.cuda.empty_cache()

            if not self.only_tracking:
                self.mapper.eval_kf_imgs(stream=self.stream,traj_est=traj_est,generate_mesh=self.cfg['generate_mesh'])
            '''
            if not self.only_tracking:
                # extract mesh
                mesh = self.mapper.mesh.extract_mesh(output=self.output)
                
                if self.cfg["dataset"] in ["replica"]:
                    try:
                        recon_result = eval_recon_with_cfg(cfg=self.cfg,
                                                           output=self.output,
                                                           traj_est = traj_est,
                                                           eval_3d=True, eval_2d=False,
                                                           kf_mesh=True, every_mesh=False,
                                                           printer=self.printer)
                        if self.wandb:
                            self.logger.log(recon_result)
                        output_str = ""
                        for k, v in recon_result.items():
                            output_str += f"{k}: {v}\n"
                        out_path = f'{self.output}/logs/metrics_mesh.txt'
                        with open(out_path, 'w+') as fp:
                            fp.write(output_str)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        self.printer.print(e)
            '''

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

    def run(self):
        m_pipe, t_pipe = mp.Pipe()
        processes = [
            mp.Process(target=self.tracking, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe,)),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        if self.wandb:
            self.printer.print('wandb finished.', FontColor.INFO)
            self.logger.finish()

        self.printer.terminate()