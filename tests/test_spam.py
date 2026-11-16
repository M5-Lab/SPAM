import numpy as np
from SPAM import run_spam

def test_spam_aligns_head_tail_equivalent_directors():
    positions = np.array([[0,0,0], [1,0,0], [2,0,0]], float)
    directors = np.array([[1,0,0], [-1,0,0], [1,0,0]], float)
    signed, flags, eigenvalue = run_spam(positions, directors, 1.5)
    assert np.allclose(np.abs(signed[:, 0]), 1)
    assert set(np.unique(flags)) <= {-1, 1}
    assert np.isfinite(eigenvalue)
