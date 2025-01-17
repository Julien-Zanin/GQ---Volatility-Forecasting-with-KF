import numpy as np
from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])

def gaussian_repr(s):
    mean = np.atleast_1d(s.mean)
    var = np.atleast_2d(s.var)
    
    if mean.size == 1:
        return f'𝒩(μ={mean.item():.3f}, 𝜎²={var.item():.3f})'
    
    mean_str = np.array2string(mean, precision=3, separator=', ', suppress_small=True)
    var_str = np.array2string(var, precision=3, separator=', ', suppress_small=True)
    return f'𝒩(μ={mean_str}, Σ={var_str})'

gaussian.__repr__ = gaussian_repr

def make_invertible(matrix, epsilon=1e-8):
    """Ensure matrix is invertible by adding small values to diagonal if needed."""
    matrix = np.atleast_2d(matrix).astype(float)
    if matrix.shape == (1, 1):
        if abs(matrix[0, 0]) < epsilon:
            return np.array([[epsilon]])
        return matrix
    try:
        np.linalg.inv(matrix)
        return matrix
    except np.linalg.LinAlgError:
        return matrix + epsilon * np.eye(matrix.shape[0])
    
def add_gaussian(g1, g2):
    """Add two Gaussians."""
    mean = g1.mean + g2.mean
    var = g1.var + g2.var
    return gaussian(mean, var)

def mul_gaussian(g1, g2, cov=None, scaling_factor=None):
    """Multiply two Gaussians with optional scaling."""
    mean1 = np.atleast_1d(g1.mean)
    mean2 = np.atleast_1d(g2.mean)
    var1 = np.atleast_2d(g1.var)
    var2 = np.atleast_2d(g2.var)
    
    if scaling_factor is None:
        scaling_factor = 1.0
    else:
        scaling_factor = np.atleast_2d(scaling_factor)
    
    var1 = make_invertible(var1)
    var2 = make_invertible(var2)
    
    var1_inv = np.linalg.inv(var1)
    var2_inv = np.linalg.inv(var2)
    
    if isinstance(scaling_factor, (int, float)):
        var = np.linalg.inv(make_invertible(var1_inv + scaling_factor**2 * var2_inv))
        mean = var @ (var1_inv @ mean1 + scaling_factor * var2_inv @ mean2)
    else:
        var = np.linalg.inv(make_invertible(var1_inv + scaling_factor.T @ var2_inv @ scaling_factor))
        mean = var @ (var1_inv @ mean1 + np.reshape(scaling_factor.T @ var2_inv @ mean2,(mean1.shape)))
    if np.isscalar(mean):
        return gaussian(float(mean), float(var))
    
    return gaussian(mean, var)