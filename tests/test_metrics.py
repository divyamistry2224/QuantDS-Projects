import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.metrics import simple_return, log_return, sharpe_ratio

def test_simple_return():
    assert simple_return(105, 100) == 0.05

def test_log_return():
    import math
    assert math.isclose(log_return(105, 100), 0.04879, rel_tol=1e-4)

def test_sharpe_ratio():
    returns = [0.01, 0.02, -0.005]
    assert isinstance(sharpe_ratio(returns), float)
