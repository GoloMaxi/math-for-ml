import numpy as np

def norm(vector: np.ndarray, L: float) -> float:
    if L == 1:
        return np.sum(np.abs(vector))
    elif L == 2:
        return np.sqrt(np.sum(np.square(vector)))
    elif L == np.inf:
        return np.max(np.abs(vector))
    else:
        raise ValueError("L must be 1, 2 or np.inf")


def cosine_distance(a: np.ndarray, b:np.ndarray) -> float:
    if np.sum(np.abs(a)) == 0 or np.sum(np.abs(b)) == 0:
        raise ValueError("Both vectors must be nonzero")
    cosine = np.dot(a, b) / (norm(a, 2) * norm(b, 2))
    return 1 - cosine


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a_sq = np.sum(np.square(A), axis=1)
    b_sq = np.sum(np.square(B), axis=1)
    dots = A @ B.T
    squared_dist = a_sq[:, None] + b_sq - 2 * dots
    return np.sqrt(squared_dist)