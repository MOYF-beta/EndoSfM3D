import sys
import torch, random, os
import numpy as np
from dataset import SCAREDRAWDataset
from options import DefaultOpt

'''
This file is used to set up the environment for the experiments
The setup includes:
    - setting random seeds
    - defining the paths to the dataset and the splits
    - dataset objects
'''
log_path = './logs'

# Default paths - update these to match your dataset locations
DEFAULT_SCARED_PATH = '/workspace/data/SCARED_Images_Resized'
DEFAULT_C3VD_PATH = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'cuda' in device:
    torch.backends.cudnn.benchmark = True

def check_test_only():
    return os.getenv('TEST_ONLY', 'False').lower() == 'true'

def random_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def get_unique_name(name):
    base_name = os.path.join(log_path, name)
    unique_name = base_name
    counter = 1
    while os.path.exists(unique_name):
        unique_name = f"{base_name}_{counter}"
        counter += 1
    return unique_name

def setup_datasets(scared_path=None, c3vd_path=None):
    """
    Setup datasets with custom paths
    Args:
        scared_path: Path to SCARED dataset
        c3vd_path: Path to C3VD dataset
    """
    if scared_path is None:
        scared_path = DEFAULT_SCARED_PATH
    if c3vd_path is None:
        c3vd_path = DEFAULT_C3VD_PATH
    
    random_seeds(23333666)
    
    global ds_train, ds_val, ds_test, ds_train_c3vd, ds_test_c3vd
    
    # SCARED DATASET SETUP
    if os.path.exists(scared_path):
        split_train = os.path.join(scared_path, 'splits/train_files.txt')
        split_val = os.path.join(scared_path, 'splits/val_files.txt')
        split_test = os.path.join(scared_path, 'splits/test_files.txt')
        
        if all(os.path.exists(f) for f in [split_train, split_val, split_test]):
            train_filenames = readlines(split_train)
            val_filenames = readlines(split_val)
            test_filenames = readlines(split_test)

            ds_train = SCAREDRAWDataset(
                data_path=scared_path,
                filenames=train_filenames,
                frame_idxs=DefaultOpt.frame_ids,
                height=DefaultOpt.height,
                width=DefaultOpt.width,
                min_depth=0.1,
                max_depth=150,
                num_scales=4,
                is_train=True,
                load_depth=True,
                img_ext='.png'
            )

            ds_val = SCAREDRAWDataset(
                data_path=scared_path,
                filenames=val_filenames,
                frame_idxs=DefaultOpt.frame_ids,
                height=DefaultOpt.height,
                width=DefaultOpt.width,
                min_depth=0.1,
                max_depth=150,
                num_scales=4,
                is_train=False,
                img_ext='.png'
            )

            ds_test = SCAREDRAWDataset(
                data_path=scared_path,
                filenames=test_filenames,
                frame_idxs=[0],
                height=DefaultOpt.height,
                width=DefaultOpt.width,
                min_depth=0.1,
                max_depth=150,
                num_scales=1,
                is_train=False,
                img_ext='.png'
            )
        else:
            print(f"Warning: SCARED split files not found in {scared_path}")
            ds_train = ds_val = ds_test = None
    else:
        print(f"Warning: SCARED dataset path not found: {scared_path}")
        ds_train = ds_val = ds_test = None

    # C3VD DATASET SETUP
    if os.path.exists(c3vd_path):
        split_train = os.path.join(c3vd_path, 'splits/train_files.txt')
        split_test = os.path.join(c3vd_path, 'splits/test_files.txt')
        
        if all(os.path.exists(f) for f in [split_train, split_test]):
            train_filenames = readlines(split_train)
            test_filenames = readlines(split_test)
            
            ds_train_c3vd = SCAREDRAWDataset(
                data_path=c3vd_path,
                filenames=train_filenames,
                frame_idxs=DefaultOpt.frame_ids,
                height=DefaultOpt.height,
                width=DefaultOpt.width,
                min_depth=0.1,
                max_depth=150,
                num_scales=4,
                is_train=True,
                depth_rescale_factor=150 / 65535,
                load_depth=True,
                img_ext='.png'
            )
            
            ds_test_c3vd = SCAREDRAWDataset(
                data_path=c3vd_path,
                filenames=test_filenames,
                frame_idxs=[0],
                height=DefaultOpt.height,
                width=DefaultOpt.width,
                num_scales=1,
                depth_rescale_factor=150 / 65535,
                min_depth=0.1,
                max_depth=150,
                is_train=False,
                load_depth=True,
                img_ext='.png'
            )
        else:
            print(f"Warning: C3VD split files not found in {c3vd_path}")
            ds_train_c3vd = ds_test_c3vd = None
    else:
        print(f"Warning: C3VD dataset path not found: {c3vd_path}")
        ds_train_c3vd = ds_test_c3vd = None

# Initialize with default paths
setup_datasets()
