import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _bce(y: np.ndarray , y_pred: np.ndarray) -> float:
    loss = (y * np.log(y_pred)).sum() 
    loss += ((1-y) * np.log(1-y_pred)).sum()
    return -loss / len(y)

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    shape_x = X.shape
    weight = np.zeros(shape_x[1])
    bias = 0
    for e in range(steps):
        log_odds = np.matmul(X,weight) + bias
        output = _sigmoid(log_odds.squeeze())
        loss = _bce(y,output)
        weight = weight - lr * (1/shape_x[0] * np.matmul((output - y), X))
        bias = bias - lr * (output - y).mean()
    # print(weight, bias)
    return weight, bias
        
        