import numpy as np

def softmax(input):
    max = np.max(input, axis = -1, keepdims=True)
    input = input - max
    num = np.exp(input)
    return num/ np.sum(num, axis=-1,keepdims=True)
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    Q = np.matmul(Q,W_q)  # B, N, D
    K = np.matmul(K,W_k)  # B, N, D
    V = np.matmul(V,W_v)  # B, N, D
    B,N,D = Q.shape
    if D%num_heads !=0:
        raise Exception("Dimension is not correct!")
    head_dim = D//num_heads
    Q = Q.reshape(B,N,num_heads,head_dim).transpose(0,2,1,3) # B, H,N, d
    K = K.reshape(B,N,num_heads,head_dim).transpose(0,2,1,3) # B, H,N, d
    V = V.reshape(B,N,num_heads,head_dim).transpose(0,2,1,3)# B, H,N, d

    score = (Q@K.mT)/(head_dim ** 0.5) # B,H,N,N
    attn = softmax(score)
    output = attn@V # B,H,N,d
    output = output.transpose(0,2,1,3).reshape(B,N,-1)
    output = np.matmul(output,W_o)
    return output
    
    