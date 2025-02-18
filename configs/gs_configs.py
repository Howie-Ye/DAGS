from dataclasses import dataclass,field

@dataclass
class PipelineParams():
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    depth_ratio = 1.    # bounded scene: 1.0  ; unbounded scene: 0.0
    depth_ratio_shift = False

@dataclass
class ModelParams():
    sh_degree = 3
    images = "images"
    resolution = -1
    white_background = False
    eval = True
    cameras_extent = 1.0
    active_gs_threshold = 10

@dataclass
class OptimizationParams():
    iterations = 30_000
    position_lr_init = 0.00016            #0.00016
    position_lr_final = 0.0000016         #0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2
    lambda_l1_rgb_threshold = 0.3
    lambda_normal = 0.05     # 0.05 default
    lambda_normal_prior = 0.05
    lambda_depth = 1.3
    opacity_cull = 0.05       # 0.05 default

    densification_interval = 100   # 100 default
    opacity_reset_interval = 3000  # 3000 default
    densify_from_iter = 100    # 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002     # 0.0002  default


@dataclass
class DeblurArgs():
    num_subframes = 2   #include the original frame
    lr_rot = 2e-5
    lr_trans = 4e-5
    thres_trans = 1.5e-3
    thres_rot = 3.8e-5
    scale_down_rate = 0.9

@dataclass
class TrainConfig():
    model_param: ModelParams = field(default_factory=lambda :ModelParams)
    pipe_param: PipelineParams = field(default_factory=lambda :PipelineParams)
    opt_param: OptimizationParams = field(default_factory=lambda :OptimizationParams)
    deblur_args: DeblurArgs = field(default_factory=lambda :DeblurArgs)
@dataclass
class MeshExtArgs():
    mesh_res = 1024
    depth_trunc = -1.0
    voxel_size = -1.0
    sdf_trunc = -1.0
    num_cluster = 10








