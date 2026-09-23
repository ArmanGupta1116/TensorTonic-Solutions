import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    hidden_states = []
    h_t = h_0.copy()
    B,N,D = X.shape
    for i in range(N):
        x = X[:,i,:]
        h_t = np.tanh(x@W_xh.T + h_t@W_hh.T + b_h)
        hidden_states.append(h_t.copy())
        
    hidden_states = np.stack(hidden_states, axis=1)
    return {"hidden_states":np.array(hidden_states), "final_hidden_state":np.array(h_t)}
    