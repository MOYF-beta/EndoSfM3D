import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dares'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'endo3dac'))
from src.trainer_attn_encoder import TrainerAttnEncoder
from src.options_attn_encoder import AttnEncoderOpt
from src.exp_setup import (ds_val, check_test_only, get_unique_name, log_path, 
                          ds_train, ds_train_c3vd, ds_test, ds_test_c3vd)
from src.find_best_parametric import find_best_parametric
from src.load_other_models import load_DARES

opt = AttnEncoderOpt

pretrained_root_dir = './pretrained_weights'
if __name__ == "__main__":
    exp_name = 'demo'
    # SCARED
    model_name = f'{exp_name}_scared'
    trainer = TrainerAttnEncoder(model_name, log_path, opt, 
                      train_eval_ds={'train': ds_train, 'val': ds_val},
                      pretrained_root_dir=pretrained_root_dir)
    trainer.train()  # Enable training for 1 epoch test
    find_best_parametric(load_DARES, model_name,
                          only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq=1)
    find_best_parametric(load_DARES, model_name,
                          only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq=2)
    
    # C3VD training disabled for SCARED-only test
    # model_name = f'{exp_name}_c3vd'
    # trainer = TrainerAttnEncoder(model_name, log_path, opt, 
    #                   train_eval_ds={'train': ds_train_c3vd, 'val': ds_test_c3vd},
    #                   pretrained_root_dir=pretrained_root_dir)
    # trainer.train()
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='C3VD', dataset=ds_test_c3vd, peft=True)
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='C3VD', dataset=ds_test_c3vd, peft=True, pose_seq=2)
