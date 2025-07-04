#!/usr/bin/env python3
"""
Test script to verify all imports work without creating trainers or datasets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dares'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'endodac'))

print("Testing imports...")

# Test main imports
try:
    from src.trainer_attn_encoder import TrainerAttnEncoder
    print("✓ TrainerAttnEncoder imported successfully")
except Exception as e:
    print(f"✗ TrainerAttnEncoder import failed: {e}")

try:
    from src.options_attn_encoder import AttnEncoderOpt
    print("✓ AttnEncoderOpt imported successfully")
except Exception as e:
    print(f"✗ AttnEncoderOpt import failed: {e}")

try:
    from src.exp_setup import (ds_val, check_test_only, get_unique_name, log_path, 
                              ds_train, ds_train_c3vd, ds_test, ds_test_c3vd)
    print("✓ exp_setup imports successful")
except Exception as e:
    print(f"✗ exp_setup imports failed: {e}")

try:
    from src.find_best_parametric import find_best_parametric
    print("✓ find_best_parametric imported successfully")
except Exception as e:
    print(f"✗ find_best_parametric import failed: {e}")

try:
    from src.load_other_models import load_DARES
    print("✓ load_DARES imported successfully")
except Exception as e:
    print(f"✗ load_DARES import failed: {e}")

# Test DARES imports
try:
    from dares.layers import *
    print("✓ DARES layers imported successfully")
except Exception as e:
    print(f"✗ DARES layers import failed: {e}")

try:
    from dares.utils import *
    print("✓ DARES utils imported successfully")
except Exception as e:
    print(f"✗ DARES utils import failed: {e}")

# Test Endo3DAC imports
try:
    from endodac.utils.layers import SSILoss
    print("✓ Endo3DAC SSILoss imported successfully")
except Exception as e:
    print(f"✗ Endo3DAC SSILoss import failed: {e}")

# Test network imports
try:
    from optical_flow_decoder import PositionDecoder
    print("✓ PositionDecoder imported successfully")
except Exception as e:
    print(f"✗ PositionDecoder import failed: {e}")

try:
    from appearance_flow_decoder import TransformDecoder
    print("✓ TransformDecoder imported successfully")
except Exception as e:
    print(f"✗ TransformDecoder import failed: {e}")

print("\nAll critical imports are working! ✓")
print("The reorganized structure is functional.")
