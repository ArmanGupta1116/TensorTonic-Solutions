import numpy as np

def vanilla_rnn(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, W_hy: np.ndarray, b_h: np.ndarray,
                b_y: np.ndarray) -> dict:
    """
    Returns outputs and final_hidden_state as float64 arrays.
    """
    h_t = h_0.copy()
    B,N,D = X.shape
    output = []
    for i in range(N):
        x = X[:,i,:]
        h_t = np.tanh(x@W_xh.T + h_t@W_hh.T + b_h)
        y_t = h_t@W_hy.T + b_y
        output.append(y_t.copy())

    output = np.stack(output, axis = 1)
    return {"outputs":output, "final_hidden_state": h_t}
    