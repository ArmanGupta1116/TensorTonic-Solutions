import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> np.ndarray:
    """
    Returns T float64 spectral-norm powers.
    """
    norm_decay = [1]
    for i in range(T-1):
        norm_decay.append(norm_decay[-1] * np.linalg.norm(W_hh, ord=2))
    return np.array(norm_decay, dtype=np.float64)