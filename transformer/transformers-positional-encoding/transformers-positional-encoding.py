import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    PE = np.zeros((seq_length,d_model))
    pos = np.arange(seq_length).reshape(-1,1)
    print(pos)
    angle = (pos/(10000**(np.arange(0,d_model,2)/d_model)))
    PE[:,0::2] = np.sin(angle)
    PE[:,1::2] = np.cos(angle)
    return PE