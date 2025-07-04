import os,sys
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
this script is used to load the pose encoder and decoder for other models
'''


def load_pose_encoder_decoder_DARES(opt ):
    from dares.networks import ResnetEncoder, PoseDecoder
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location=device.type))

    pose_decoder = PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path, map_location=device.type))

    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()
    return pose_encoder, pose_decoder

def load_DARES(opt, weight_path=None, pth_name='depth_model.pth', refine=False, peft=True):
    if peft:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dares', 'networks')))
        from dares_peft import DARES
    else:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dares', 'networks')))
        from dares import DARES
    if weight_path is None:
        weight_path = opt.load_weights_folder
    depth_model_path = os.path.join(weight_path, pth_name)
    if not os.path.exists(depth_model_path):
        depth_model_path = os.path.join(weight_path, 'depth.pth')
    depth_model_dict = torch.load(depth_model_path)
    depth_model = DARES(enable_refine_net=refine)
    if peft:
        for k in list(depth_model_dict.keys()):
            if 'module.' in k:
                depth_model_dict[k.replace('module.', '')] = depth_model_dict.pop(k)
        for k in list(depth_model_dict.keys()):
            if '.default' in k and '.default.weight' not in k:
                depth_model_dict[k.replace('.default', '.default.weight')] = depth_model_dict.pop(k)

        depth_model.load_state_dict(depth_model_dict)
    else:
        model_dict = depth_model.state_dict()
        depth_model.load_state_dict({k: v for k, v in depth_model_dict.items() if k in model_dict})
    depth_model.cuda()
    depth_model.eval()

    return depth_model

def load_DARES_CPE(opt, weight_path=None, pth_name='depth_model.pth'):
    # DARES_CPE was removed - this is a placeholder
    raise NotImplementedError("DARES_CPE was not included in this release")
    if weight_path is None:
        weight_path = opt.load_weights_folder
    depth_model_path = os.path.join(weight_path, pth_name)
    if not os.path.exists(depth_model_path):
        depth_model_path = os.path.join(weight_path, 'depth.pth')
    depth_model = DARES_cpe(opt.frame_ids, opt.other_frame_init_weight)
    state_dict = torch.load(depth_model_path, map_location=device, weights_only=True)
    depth_model.load_state_dict(state_dict)
    depth_model.cuda()
    depth_model.eval()

    return depth_model