# DARES Framework - Depth Estimation and Related Systems
"""
DARES (Depth Accurate Reconstruction and Estimation System)
A framework for monocular depth estimation in medical imaging.
"""

__version__ = "1.0.0"
__author__ = "DARES Team"

# Lazy imports to avoid circular dependencies
def _get_layers():
    from . import layers
    return layers

def _get_utils():
    from . import utils
    return utils

def _get_networks():
    from . import networks
    return networks