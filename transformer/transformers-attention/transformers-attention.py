import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    shape = Q.shape
    score = (Q@K.mT)/(Q.shape[-1] ** 0.5)
    score = torch.nn.Softmax(dim=-1)(score)
    return score@V
    