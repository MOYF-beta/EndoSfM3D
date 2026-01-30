#!/usr/bin/env python3
"""
Integration test to verify the compatibility module works with actual import paths.
"""

import sys
import os

# Get the repository root
repo_root = os.path.dirname(os.path.abspath(__file__))

# Add paths as the real code does
sys.path.append(os.path.join(repo_root, 'src'))
sys.path.insert(0, os.path.abspath(os.path.join(repo_root, 'dares', 'networks')))  # Use insert(0) like real code

print('Testing imports from actual source files...')
print('=' * 50)

# Test 1: Import from load_other_models context
print('\n1. Testing src/load_other_models.py context (default):')
os.environ.pop('OLD_DARES_ARCH', None)  # Ensure it's not set

from dares_compat import get_DARES_class

try:
    DARES = get_DARES_class()
    print('   ✗ Should have failed (no transformers installed)')
except ImportError as e:
    if 'dares_peft' in str(e) or 'transformers' in str(e):
        print('   ✓ Correctly tries to import dares_peft by default')
    else:
        print(f'   ✗ Unexpected error: {e}')

# Test 2: With OLD_DARES_ARCH=1
# Need to reimport in a fresh Python interpreter to properly test
# Instead, we'll test the logic directly
print('\n2. Testing with OLD_DARES_ARCH=1:')
print('   (Starting fresh subprocess to test environment variable...)')

import subprocess
test_code = """
import sys
import os
os.environ['OLD_DARES_ARCH'] = '1'
# Use insert(0) like the real code does
sys.path.insert(0, os.path.abspath(os.path.join('.', 'dares', 'networks')))
from dares_compat import get_DARES_class
try:
    DARES = get_DARES_class()
    print('ERROR: Should have failed')
except ImportError as e:
    if 'Failed to import dares:' in str(e):
        print('SUCCESS: Correctly tries to import dares')
    else:
        print(f'ERROR: Wrong module - {e}')
"""

result = subprocess.run(
    [sys.executable, '-c', test_code],
    cwd=repo_root,
    capture_output=True,
    text=True
)

if 'SUCCESS' in result.stdout:
    print('   ✓ Correctly tries to import dares with OLD_DARES_ARCH=1')
elif 'ERROR' in result.stdout:
    print(f'   ✗ {result.stdout.strip()}')
    if result.stderr:
        print(f'   Error details: {result.stderr}')
else:
    print(f'   ✗ Unexpected output: {result.stdout}')
    if result.stderr:
        print(f'   Stderr: {result.stderr}')

print('\n' + '=' * 50)
print('✓ Integration test completed!')


