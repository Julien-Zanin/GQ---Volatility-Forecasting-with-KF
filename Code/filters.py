from gaussians import gaussian, add_gaussian, mul_gaussian
import numpy as np
from state_space_models import *
from typing import Union
from scipy import linalg

SSmodel = Union[SSARIMA, SSVECH]


####################################################################################################
################### LINEAR KALMAN FILTER ###########################################################
####################################################################################################

def KalmanFilter(model: SSmodel, observations: np.ndarray) -> dict:
    """
    Apply Linear Kalman filter on state-space model with observations
    
    Parameters:
    -----------
        model: StateSpaceModel object
        observations: array of observed data
    
    Returns:
    --------
        dict with filtered states, filtered variances, state predictions, state estimates, priors and posteriors
    """
    
    n = len(observations)
    filtered_states = []
    filtered_variances = []
    state_predictions = []
    state_estimates = []
    priors = []
    posteriors = []
    residuals = []
    residuals_var = []
    
    # Initial state
    posterior = gaussian(model.a1, model.P1)
    process_noise = gaussian(np.array([[0.0]]), model.Q)

    for t in range(n):
        # Predict step
        prior = predict(posterior, process_noise, model.T, model.R)
        state_predictions.append(prior)
        priors.append(gaussian(float(sum(prior.mean[:model.d])), float(sum(prior.var[:model.d, :model.d]))))
        
        # Update step
        observation_noise = gaussian(0, model.Z @ prior.var @ model.Z.T + model.H)
        residuals_var.append(float(observation_noise.var))
        posterior = update(prior, observation_noise, observations[t], model.Z)

        state_estimates.append(posterior)
        posteriors.append(gaussian(float(sum(posterior.mean[:model.d])), float(sum(posterior.var[:model.d, :model.d]))))
        filtered_states.append(float(sum(posterior.mean[:model.d])))
        filtered_variances.append(float(sum(posterior.var[:model.d, :model.d])))
        residuals.append(observations[t] - float(sum(posterior.mean[:model.d])))

    return {
        "filtered states": filtered_states,
        "filtered variances": filtered_variances,
        "state predictions": state_predictions,
        "state estimates": state_estimates,
        "priors": priors,
        "posteriors": posteriors,
        "residuals": residuals,
        "residuals var": residuals_var
    }

def predict(posterior: gaussian, noise: gaussian, T: np.ndarray, R: np.ndarray) -> gaussian:
    """Kalman filter prediction step"""
    state = gaussian(T @ posterior.mean, T @ posterior.var @ T.T)
    noise = gaussian(R * noise.mean, R @ noise.var @ R.T)
    return add_gaussian(state, noise)

def update(prior: gaussian, noise: gaussian, obs: float, Z: np.ndarray) -> gaussian:
    """Kalman filter update step as a product of Gaussians"""
    measurement = gaussian(mean=np.array([[obs]]), var=noise.var)
    return mul_gaussian(prior, measurement, scaling_factor=Z)

####################################################################################################
################### UNSCENTED KALMAN FILTER ########################################################
####################################################################################################

def UnscentedKalmanFilter(model: SSmodel, observations: np.ndarray) -> dict:
    """
    Apply Unscented Kalman filter on state-space model with observations
    
    Parameters:
    -----------
        model: StateSpaceModel object
        observations: array of observed data
    
    Returns:
    --------
        dict with filtered states, filtered variances, state predictions and state estimates
    """
    
    n = observations.shape[0]
    idx = model.n * (1+model.n) // 2
    filtered_states = []
    filtered_variances = []
    state_predictions = []
    state_estimates = []
    priors = []
    posteriors = []

    # Initial state
    posterior = gaussian(model.a1, model.P1)
    process_noise = gaussian(np.zeros((model.dim,1)), model.Q)

    for t in range(n):
        # Predict step
        prior, sigma_points, weights = unscented_predict(posterior, process_noise, model.state_transform, model.residuals_transform)
        state_predictions.append(prior)
        priors.append(gaussian(prior.mean[1:model.m+1], prior.var[1:model.m+1, 1:model.m+1]))
        # Update step
        posterior = unscented_update(prior, model.H, sigma_points, model.measurement_transform, weights, observations[t,:])
        state_estimates.append(posterior)
        posteriors.append(gaussian(posterior.mean[1:model.m+1], posterior.var[1:model.m+1, 1:model.m+1]))
        filtered_states.append(posterior.mean[1:model.m+1])
        filtered_variances.append(posterior.var[1:model.m+1, 1:model.m+1])

    return {
        "filtered states": filtered_states,
        "filtered variances": filtered_variances,
        "state predictions": state_predictions,
        "state estimates": state_estimates,
        "priors": priors,
        "posteriors": posteriors
    }

def unscented_predict(posterior: gaussian, 
                      noise: gaussian, 
                      state_t: callable, 
                      residuals_t: callable) -> gaussian:
    """Unscented Kalman filter prediction step"""
    sigma_pts, Wm, Wc = get_sigma_points(posterior.mean, posterior.var)
    sigma_points_r, Wm_r, Wc_r = get_sigma_points(noise.mean, noise.var)
    
    transformed_sigmas = np.array([state_t(sigma_pts[i,:]) for i in range(sigma_pts.shape[0])])
    transformed_sigmas_r = np.array([residuals_t(sigma_points_r[i,:]) for i in range(sigma_points_r.shape[0])])
    mean, var = unscented_transform(transformed_sigmas, Wm, Wc)
    g1 = gaussian(mean, var)
    mean_r, var_r  = unscented_transform(transformed_sigmas_r, Wm_r, Wc_r)
    g2 = gaussian(mean_r, var_r)
    return add_gaussian(g1, g2), transformed_sigmas, (Wm, Wc)

def unscented_update(prior: gaussian, 
                     obs_var: np.ndarray, 
                     sigma_points: np.ndarray,
                     measurement_t: callable,
                     W: tuple,
                     obs: float) -> gaussian:
    """Unscented Kalman filter update step"""
    sigma_obs = np.array([measurement_t(sigma_points[i,:]) for i in range(sigma_points.shape[0])])
    zp, Pz = unscented_transform(sigma_obs, W[0], W[1], obs_var)
    Pxz = np.zeros((prior.mean.shape[0], sigma_obs.shape[1]))
    for i in range(sigma_points.shape[0]):
            Pxz += W[1][i] * np.outer(sigma_points[i] - prior.mean, sigma_obs[i] - zp)
    K = np.dot(Pxz, linalg.inv(Pz))
    x = prior.mean + np.dot(K, obs - zp)
    P = prior.var - np.dot(K, Pz).dot(K.T)
    return gaussian(x, P)

def get_sigma_points(X: np.ndarray, P: np.ndarray, n: int=None, alpha: float=0.5,beta: int=2,kappa: int=None)->tuple:
    n = X.shape[0] if n is None else n
    kappa = 3 - n if kappa is None else kappa
    lambda_ = alpha**2 * (n + kappa) - n
    Wc = np.full(2*n + 1,  1. / (2*(n + lambda_)))
    Wm = np.full(2*n + 1,  1. / (2*(n + lambda_)))
    Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
    Wm[0] = lambda_ / (n + lambda_)
    sigmas = np.zeros((2*n+1, n))
    U = linalg.cholesky((n+lambda_)*P + np.eye(P.shape[0]) * 1e-10)
    sigmas[0,:] = X.ravel()
    for k in range(n):
        sigmas[k+1] = (X.ravel() + U[:, k].ravel())
        sigmas[n+k+1] = (X.ravel() - U[:, k].ravel())
    return sigmas, Wm, Wc

def unscented_transform(transformed_sigmas, Wm, Wc, H=np.array([[0]])):
    x = np.dot(Wm, transformed_sigmas)
    kmax, n = transformed_sigmas.shape
    P = np.zeros((n, n))
    for k in range(kmax):
        y = transformed_sigmas[k] - x
        P += Wc[k] * np.outer(y, y)
    P+=H
    return x, P