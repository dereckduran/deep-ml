import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    z = W @ a_prev + b
    a = sigmoid(z)
    C = np.sum((a - y) ** 2)
    dC_dz = 2 * (a - y) * a * (1 - a)

    db = dC_dz
    dW = np.outer(dC_dz, a_prev)
    da_prev = dC_dz @ W

    return {'dW': dW.tolist(), 'db': db.tolist(), 'da_prev': da_prev.tolist()}



