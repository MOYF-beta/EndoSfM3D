"""
Compatibility module for DARES architecture selection.

This module provides a compatibility layer to support both the old DARES
architecture and the new DARES PEFT architecture based on environment variables.

When OLD_DARES_ARCH=1 is set, it uses the legacy 'dares' module.
Otherwise, it uses the newer 'dares_peft' module (default).
"""

import os
import sys


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
    
    # Try different import methods to handle various contexts
    try:
        # Try relative import first (when used as part of dares.networks package)
        if '.' in __name__:
            parent_module = '.'.join(__name__.split('.')[:-1])
            full_module_name = f"{parent_module}.{module_name}"
            module = __import__(full_module_name, fromlist=[module_name])
        else:
            # Direct import (when dares/networks is in sys.path)
            module = __import__(module_name)
        return module
    except ImportError as e:
        # Fallback: try direct import
        try:
            module = __import__(module_name)
            return module
        except ImportError:
            raise ImportError(f"Failed to import {module_name}: {e}")


def get_DARES_class():
    """
    Get the DARES class from the appropriate module.
    
    Returns:
        class: DARES class from either dares_peft or dares module
    """
    module = get_dares_module()
    return module.DARES
