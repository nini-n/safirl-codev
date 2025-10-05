import os
import sys
import numpy as np
import pytest

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture(autouse=True)
def _seed_every_test():
    np.random.seed(0)
    yield
