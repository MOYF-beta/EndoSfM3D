#!/usr/bin/env python3
"""
Test script to verify the attention encoder DORA setup is working correctly.
This tests the core imports and model loading without requiring datasets.
"""

import sys
import os
# Add both src and the root directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all key modules can be imported"""
    print("Testing core imports...")
    
    try:
        from src.trainer_attn_encoder import TrainerAttnEncoder
        print("✓ TrainerAttnEncoder imported successfully")
    except Exception as e:
        print(f"✗ TrainerAttnEncoder import failed: {e}")
        return False
    
    try:
        from src.options_attn_encoder import AttnEncoderOpt
        print("✓ AttnEncoderOpt imported successfully")
    except Exception as e:
        print(f"✗ AttnEncoderOpt import failed: {e}")
        return False
    
    try:
        from src.find_best_parametric import find_best_parametric
        print("✓ find_best_parametric imported successfully")
    except Exception as e:
        print(f"✗ find_best_parametric import failed: {e}")
        return False
    
    try:
        from src.load_other_models import load_DARES
        print("✓ load_DARES imported successfully")
    except Exception as e:
        print(f"✗ load_DARES import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that the DARES model can be instantiated"""
    print("\nTesting model instantiation...")
    
    try:
        from dares.networks.dares_peft import DARES
        model = DARES(use_dora=True, target_modules=['query', 'value'], full_finetune=True)
        print("✓ DARES model created successfully")
        print(f"  Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"✗ DARES model creation failed: {e}")
        return False

def test_configuration():
    """Test that configuration options work"""
    print("\nTesting configuration...")
    
    try:
        from src.options_attn_encoder import AttnEncoderOpt
        opt = AttnEncoderOpt
        print("✓ Configuration loaded successfully")
        print(f"  Image size: {opt.height}x{opt.width}")
        print(f"  Frame IDs: {opt.frame_ids}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Attention Encoder DORA Setup Test")
    print("=" * 40)
    
    tests = [
        ("Core Imports", test_imports),
        ("Model Loading", test_model_loading), 
        ("Configuration", test_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The setup is working correctly.")
        print("\nNext steps:")
        print("1. Set up your datasets in ./data/")
        print("2. Configure dataset paths in src/exp_setup.py")
        print("3. Run: python train_attn_encoder_dora.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
