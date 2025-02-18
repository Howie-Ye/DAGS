import time

import numpy as np
import torch
import argparse
import os
import sys

sys.path.append('./src/gaussian_splatting')
from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from configs.gs_configs import TrainConfig,PipelineParams,OptimizationParams,ModelParams,MeshExtArgs,DeblurArgs
from time import gmtime, strftime
from colorama import Fore, Style

import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    parser.add_argument("--wandb",action="store_true", help="Wandb")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of frames,-1 means all frames.")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--lambda_pd", type=float, default=-1)
    parser.add_argument("--lambda_pn", type=float, default=-1)
    parser.add_argument("--depth_ratio", type=float, default=-1)
    parser.add_argument("--thres_trans", type=float, default=-1)
    parser.add_argument("--thres_rot", type=float, default=-1)
    parser.add_argument("--lr_trans", type=float, default=-1)
    parser.add_argument("--lr_rot", type=float, default=-1)
    parser.add_argument("--scale_down", type=float, default=-1)
    parser.add_argument("--sh_degree", type=float, default=-1)
    parser.add_argument("--only_deblur_test", action="store_true")
    parser.add_argument("--use_sensor_depth", action="store_true")

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(
        args.config, './configs/mono_point_slam.yaml'
    )
    setup_seed(cfg['setup_seed'])

    cfg['wandb'] = args.wandb

    if args.only_tracking:
        cfg['only_tracking'] = True
        cfg['wandb'] = False
    if args.max_frames > 0:
        cfg['tracking']['buffer'] = args.max_frames
        cfg['max_frames'] = args.max_frames

    opt_param = OptimizationParams()
    pipe_param = PipelineParams()
    deblur_args = DeblurArgs()
    model_param = ModelParams()
    if args.sh_degree != -1:
        model_param.sh_degree = args.sh_degree
        args.note += f"_sh={args.sh_degree}"
    if args.lambda_pd != -1:
        opt_param.lambda_depth = args.lambda_pd
        args.note += f"_pd={args.lambda_pd}"
    if args.lambda_pn != -1:
        opt_param.lambda_normal_prior = args.lambda_pn
        args.note += f"_pn={args.lambda_pn}"
    if args.depth_ratio != -1:
        pipe_param.depth_ratio = args.depth_ratio
        args.note += f"_depth_ratio={args.depth_ratio}"
    if args.thres_trans != -1:
        deblur_args.thres_trans = args.thres_trans
        args.note += f"_trans={args.thres_trans}"
    if args.thres_rot != -1:
        deblur_args.thres_rot = args.thres_rot
        args.note += f"_rot={args.thres_rot}"
    if args.lr_trans != -1:
        deblur_args.lr_trans = args.lr_trans
        args.note += f"_lr_trans={args.lr_trans}"
    if args.lr_rot != -1:
        deblur_args.lr_rot = args.lr_rot
        args.note += f"_lr_rot={args.lr_rot}"
    if args.scale_down != -1:
        deblur_args.scale_down_rate = args.scale_down
        args.note += f"_scale_down={args.scale_down}"
    if args.only_deblur_test:
        cfg['only_deblur_test'] = True
    if args.use_sensor_depth:
        cfg['use_sensor_depth'] = True
        args.note += f"_gt_depth"

    Gs_cfgs = TrainConfig(opt_param=opt_param,pipe_param=pipe_param,deblur_args=deblur_args)
    start_time = time.localtime()
    start_time_m = strftime("%Y-%m-%d-%H_%M", time.localtime())
    output_dir = cfg['data']['output']
    output_dir = output_dir + f"/{cfg['scene']}/{start_time_m}_{args.note}"

    start_time_str = strftime("%Y-%m-%d %H:%M:%S", start_time)
    start_info = "-" * 30 + Fore.LIGHTRED_EX + \
                 f"\nStart DAGS at {start_time_str},\n" + Style.RESET_ALL + \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   only_tracking: {cfg['only_tracking']},\n" \
                 f"   output: {output_dir}\n" + \
                 "-" * 30
    print(start_info)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    dataset = get_dataset(cfg)

    slam = SLAM(cfg, Gs_cfgs, dataset, output_dir)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("-" * 30 + Fore.LIGHTRED_EX + f"\nDAGS finishes!\n" + Style.RESET_ALL + f"{end_time}\n" + "-" * 30)
