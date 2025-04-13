# tests/conftest.py

import sys
from pathlib import Path

# Determine the project root directory.
project_root = Path(__file__).resolve().parent.parent
# Insert the project root at the start of sys.path.
sys.path.insert(0, str(project_root))
