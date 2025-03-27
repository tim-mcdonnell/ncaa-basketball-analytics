"""Configure pytest."""

import sys
import os
import importlib.util
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Load conftest_fixtures.py programmatically to avoid E402 issues
fixtures_path = os.path.join(os.path.dirname(__file__), "conftest_fixtures.py")
spec = importlib.util.spec_from_file_location("conftest_fixtures", fixtures_path)
fixtures_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures_module)

# Import all fixtures into the global namespace
for name in dir(fixtures_module):
    if name.startswith("__"):
        continue
    globals()[name] = getattr(fixtures_module, name)
