# DARES Architecture Compatibility

This document explains the compatibility patch that allows switching between the old DARES architecture and the new DARES PEFT architecture.

## Overview

The codebase supports two different DARES architectures:
- **dares_peft** (default): The newer architecture with PEFT/DoRA support
- **dares**: The legacy architecture

## Usage

### Default Behavior

By default, the code uses the newer `dares_peft` module:

```bash
python train_attn_encoder_dora.py
```

### Using the Old Architecture

To use the legacy `dares` architecture, set the environment variable `OLD_DARES_ARCH=1`:

```bash
export OLD_DARES_ARCH=1
python train_attn_encoder_dora.py
```

Or set it inline:

```bash
OLD_DARES_ARCH=1 python train_attn_encoder_dora.py
```

## Technical Details

### Compatibility Module

The compatibility is implemented through the `dares/networks/dares_compat.py` module, which provides:

- `get_dares_module()`: Returns the appropriate DARES module based on environment variable
- `get_DARES_class()`: Returns the DARES class from the appropriate module

### Environment Variable

- `OLD_DARES_ARCH=1`: Use the old `dares` architecture
- `OLD_DARES_ARCH` unset or any other value: Use the new `dares_peft` architecture (default)

### Modified Files

The following files have been updated to use the compatibility module:

1. **src/load_other_models.py**: The `load_DARES()` function now uses the compatibility module when `peft=True`
2. **src/trainer_attn_encoder.py**: Imports DARES using the compatibility module
3. **test_setup.py**: Imports DARES using the compatibility module

## Testing

A test script is provided to verify the compatibility logic:

```bash
python test_dares_compat.py
```

This test validates:
- Default behavior uses `dares_peft`
- Setting `OLD_DARES_ARCH=1` uses `dares`
- Other values use the default `dares_peft`

## Example Usage Scenarios

### Scenario 1: Regular Training (New Architecture)
```bash
# Uses dares_peft by default
python train_attn_encoder_dora.py
```

### Scenario 2: Legacy Compatibility (Old Architecture)
```bash
# Uses old dares architecture
OLD_DARES_ARCH=1 python train_attn_encoder_dora.py
```

### Scenario 3: Testing with Old Architecture
```bash
# Run tests with old architecture
OLD_DARES_ARCH=1 python test_setup.py
```

### Scenario 4: Loading Old Model Weights
```python
import os
from src.load_other_models import load_DARES

# Set environment variable before loading
os.environ['OLD_DARES_ARCH'] = '1'

# Load with old architecture
model = load_DARES(opt, peft=True)
```

## Migration Guide

If you have existing code that imports from `dares_peft` directly:

### Before
```python
from dares.networks.dares_peft import DARES
```

### After
```python
import sys
sys.path.append('dares/networks')
from dares_compat import get_DARES_class
DARES = get_DARES_class()
```

Or when using in scripts with proper path handling:
```python
from dares_compat import get_DARES_class
DARES = get_DARES_class()
```

## Notes

- The environment variable must be set **before** importing any modules that use DARES
- Both architectures require the `transformers` library
- The compatibility layer has minimal performance overhead
- This patch maintains backward compatibility while allowing for future architecture changes
