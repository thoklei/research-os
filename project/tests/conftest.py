"""
Root conftest.py for test suite.

Configures Python path and provides shared fixtures for all tests.
"""

import sys
from pathlib import Path

# Add src directory to Python path for all tests
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Note: Module-specific fixtures should be defined in their respective
# subdirectory conftest.py files (e.g., tests/models/conftest.py)
