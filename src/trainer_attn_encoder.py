import os
import sys
import torch

# Add dares to path for direct imports
current_dir = os.path.dirname(__file__)
dares_path = os.path.join(current_dir, '..', 'dares')
dares_networks_path = os.path.join(dares_path, 'networks')
endo3dac_path = os.path.join(current_dir, '..', 'endo3dac', 'utils')

sys.path.insert(0, dares_networks_path)

from dares_peft import DARES
from resnet_encoder import AttentionalResnetEncoder, MultiHeadAttentionalResnetEncoder
from pose_decoder import PoseDecoder_with_intrinsics as PoseDecoder_i
from optical_flow_decoder import PositionDecoder
from appearance_flow_decoder import TransformDecoder
from trainer_abc import Trainer

class TrainerAttnEncoder(Trainer):
    def load_model(self):
        # Initialize depth model
        encoders = {
            "depth_model": DARES(use_dora=True,target_modules=['query', 'value'],full_finetune=True),
            # NOTE +1 is the testing for apply appearance flow
            "pose_encoder": MultiHeadAttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=self.num_pose_frames),
            "position_encoder": MultiHeadAttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2),
            "transform_encoder": MultiHeadAttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2)
        }

        decoders = {
            "pose": PoseDecoder_i(
                encoders["pose_encoder"].num_ch_enc, 
                image_width=self.opt.width, 
                image_height=self.opt.height, 
                predict_intrinsics=self.opt.learn_intrinsics, 
                simplified_intrinsic=self.opt.simplified_intrinsic, 
                num_input_features=1, 
                num_frames_to_predict_for=2,
                auto_scale=True,
                # use_knn=True,
                # knn_k=5,
                # knn_temperature=1.0
            ),
            "position": PositionDecoder(encoders["position_encoder"].num_ch_enc, self.opt.scales),
            "transform": TransformDecoder(encoders["transform_encoder"].num_ch_enc, self.opt.scales)
        }

        all_models = [
            ("depth_model",         encoders["depth_model"],        self.param_monodepth),
            ("pose_encoder",        encoders["pose_encoder"],       self.param_monodepth),
            ("pose",                decoders["pose"],               self.param_monodepth),
            ("position_encoder",    encoders["position_encoder"],   self.param_pose_net),
            ("position",            decoders["position"],           self.param_pose_net),
            ("transform_encoder",   encoders["transform_encoder"],  self.param_pose_net),
            ("transform",           decoders["transform"],          self.param_pose_net)
        ]
        # Initialize models and parameters, and move to device
        for model_name, model_instance, param_list in all_models:
            self.models[model_name] = model_instance
            param_list += list(filter(lambda p: p.requires_grad, model_instance.parameters()))
            # for name, param in model_instance.named_parameters():
            #     print(f"\033[94m{name} requires grad: {param.requires_grad}\033[0m")
            self.models[model_name].to(self.device)
        # Load pretrained weights if available
        if self.pretrained_root_dir is not None:
            model_paths = [
            "depth_model",
            "position_encoder",
            "position",
            "transform_encoder",
            "transform",
            "pose_encoder",
            "pose"
            ]
            for model_name in model_paths:
                model_path = os.path.join(self.pretrained_root_dir, "best", f"{model_name}.pth")
                if os.path.exists(model_path):
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                    model_state_dict = self.models[model_name].state_dict()
                    filtered_state_dict = {}
                    for key, value in state_dict.items():
                        if key in model_state_dict:
                            model_shape = model_state_dict[key].shape
                            value_shape = value.shape
                            
                            if model_shape == value_shape:
                                filtered_state_dict[key] = value
                            else:
                                # Try to patch size mismatches by padding with near-zero values
                                try:
                                    if len(model_shape) == len(value_shape):
                                        # Create tensor with model shape, filled with small random values
                                        patched_tensor = torch.randn(model_shape, device=value.device) * 1e-6
                                        
                                        # Copy original values to the overlapping region
                                        slices = tuple(slice(0, min(m, v)) for m, v in zip(model_shape, value_shape))
                                        patched_tensor[slices] = value[slices]
                                        
                                        filtered_state_dict[key] = patched_tensor
                                        print(f"\033[92mPatched parameter {key} from {value_shape} to {model_shape}\033[0m")
                                    else:
                                        print(f"\033[93mSkipping parameter {key} due to dimension mismatch: {value_shape} vs {model_shape}\033[0m")
                                except Exception as e:
                                    print(f"\033[93mFailed to patch parameter {key}: {e}\033[0m")
                        else:
                            print(f"\033[93mSkipping parameter {key} - not found in model\033[0m")
                    self.models[model_name].load_state_dict(filtered_state_dict, strict=False)
                    print(f"\033[94mLoaded pretrained weights for {model_name} from {model_path}\033[0m")
                else:
                    print(f"\033[93mNo pretrained weights found for {model_name} at {model_path}\033[0m")
    
    def get_depth_input(self, inputs):
        return inputs['color_aug', 0, 0]