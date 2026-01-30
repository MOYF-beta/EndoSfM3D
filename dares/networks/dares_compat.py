"""
Compatibility module for DARES architecture selection.

This module provides a compatibility layer to support both the old DARES
architecture and the new DARES PEFT architecture based on environment variables.

When OLD_DARES_ARCH=1 is set, it uses the legacy 'dares' module.
Otherwise, it uses the newer 'dares_peft' module (default).
"""

import os
import sys
import importlib


def get_dares_module():
    """
    Import and return the appropriate DARES module based on environment variable.
    
    Returns:
        module: Either dares_peft or dares module containing DARES class
        
    Environment Variables:
        OLD_DARES_ARCH: When set to '1', uses legacy dares module instead of dares_peft
    """
    use_old_arch = os.environ.get('OLD_DARES_ARCH', '0') == '1'
    
    # Determine which module to import
    module_name = 'dares' if use_old_arch else 'dares_peft'
    
    # Clear any cached imports to respect environment variable changes
    full_module_names = [module_name]
    
    # Try to construct full module path if we're in a package
    if '.' in __name__:
        parent_module = '.'.join(__name__.split('.')[:-1])
        full_module_names.append(f"{parent_module}.{module_name}")
    
    # Try importing with each method
    for full_name in full_module_names:
        # Remove from cache if it exists to allow reload
        if full_name in sys.modules:
            del sys.modules[full_name]
    
    try:
        # Direct import (when dares/networks is in sys.path)
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        raise ImportError(f"Failed to import {module_name}: {e}")


def get_DARES_class():
    """
    Get the DARES class from the appropriate module.
    
    Returns:
        class: DARES class from either dares_peft or dares module
    """
    module = get_dares_module()
    return module.DARES
