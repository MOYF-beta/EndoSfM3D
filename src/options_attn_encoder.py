DefaultOpt_dict = {
        # TRAINING options
        "split": "endovis",
        "num_layers": 18,
        "dataset": "endovis",
        "png": False,
        "height": 256,
        "width": 320,
        "disparity_smoothness": 1e-5,
        "position_smoothness": 1e-5,
        "transform_constraint": 1e-4,
        "transform_smoothness": 1e-4,
        "warm_up_step": 5,
        "self_ssi_constraint": 0.1,
        "scales": [0, 1, 2, 3],
        "self_ssi": True,
        "min_depth": 0.1,
        "max_depth": 150.0,
        "use_stereo": False,
        "frame_ids": [0, -1, 1],
        "other_frame_init_weight" : 1e-5,

        # OPTIMIZATION options
        "batch_size": 10,
        "learning_rate": 1e-5,
        "weight_decay": 1e-7,
        "weight_decay_pose" : 1e-4,
        "pos_learning_rate": 1e-4,
        "num_epochs": 20,  # Set to 1 for testing
        "scheduler_step_size": 10,

        # ABLATION options
        "learn_intrinsics" : True,
        "simplified_intrinsic" : False,
        "avg_reprojection": False,
        "no_ssim": False,
        "weights_init": "pretrained",
        "pose_model_input": "pairs",
        "pose_model_type": "separate_resnet",

        # SYSTEM options
        "no_cuda": False,
        "num_workers": 12,

        # LOADING options
        "load_weights_folder": None,

        # LOGGING options
        "log_frequency": 100,
        "save_frequency": 1,

        # EVALUATION options
        "eval_stereo": False,
        "eval_mono": False,
        "disable_median_scaling": False,
        "pred_depth_scale_factor": 1,
        "ext_disp_to_eval": None,
        "eval_split": "endovis",
        "save_pred_disps": False,
        "no_eval": False,
        "eval_eigen_to_benchmark": False,
        "eval_out_dir": None,
        "post_process": False,
        "visualize_depth": False,
        "save_recon": False,
        "scared_pose_seq": 1,
        "zero_shot": False,
        "dam_hf_weights": None,
    }

class DotDict(dict):
    """Dictionary with dot notation access to attributes"""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]

AttnEncoderOpt = DotDict(DefaultOpt_dict)