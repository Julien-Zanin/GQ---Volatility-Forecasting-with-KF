import numpy as np
import pandas as pd 
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterGrid

from scipy.optimize import minimize

# fonctions personnalisées 
from filters import * 
from gaussians import * 
from state_space_models import *

# 
import warnings
from itertools import product


class BaseModel:
    def __init__(self, name):
        self.name = name
        self.fitted = None
        self.residuals = None
        self.mae = None
        self.rmse = None
        self.forecast = None
    
    def fit(self, y):
        """Override in child classes."""
        raise NotImplementedError
        
    def calculate_errors(self, true_vals, fitted_vals):
        errs = true_vals - fitted_vals
        self.residuals = errs
        self.mae = np.mean(np.abs(errs))
        self.rmse = np.sqrt(np.mean(errs**2))


class GARCHModel(BaseModel):
    """
    Implémentation d'un modèle GARCH(1,1) à distribution Student-t
    pour coller aux spécifications du papier.
    """
    def __init__(self):
        super().__init__(name='GARCH(1,1)')

    def fit(self, y):
        """
        y doit contenir la série de retours centrés (r*_t), 
        comme décrit dans le papier :
            r*_t = (r_t - r̄) / ρ
        où r_t est le log-return, r̄ la moyenne de r_t, et ρ son écart-type.
        """
        # Mean='zero' => pas d'ARMA, juste un terme nul pour la moyenne.
        # vol='GARCH' => on veut un GARCH(1,1).
        # dist='studentst' => distribution Student-t pour les résidus.
        garch = arch_model(
            y, 
            mean='zero', 
            vol='GARCH', 
            p=1, 
            q=1, 
            dist='studentst',
            rescale=False  # on ne veut pas de re-scaling
        )
        self.fitted = garch.fit(disp='off')
        self.shape = self.fitted.params["nu"] # renvoie le paramètre de forme de la Student-t
        self.params =   self.fitted.params
        
    def predict_vol(self, horizon=1):
        """
        Prévision hors-échantillon de la volatilité (sigma). 
        horizon=1 signifie la prévision à 1 pas de temps.
        """
        forecast_result = self.fitted.forecast(horizon=horizon)
        # forecast_result.variance est la variance conditionnelle prédite
        # puis on prend la racine carrée pour obtenir la volatilité.
        predicted_variance = forecast_result.variance.iloc[-1]
        predicted_volatility = np.sqrt(predicted_variance)
        return predicted_volatility

    def fitted_values(self):
        """
        Retourne la volatilité conditionnelle (in-sample) estimée.
        """
        return self.fitted.conditional_volatility
    
    def calculate_model_errors(self, y_true):
        """
        Exemple : si vous voulez évaluer l’erreur sur la volatilité
        in-sample.  On récupère la volatilité conditionnelle ajustée
        et on la compare à la volatilité « observée » (si vous en disposez).
        
        Si vous souhaitez évaluer l’erreur sur les retours, 
        adaptez la logique en conséquence.
        """
        # Exemple : on compare la volatilité in-sample du modèle
        # à une volatilité 'observée' (y_true).
        # Si y_true = sqrt(var_observée), c’est cohérent.
        fitted_vol = self.fitted_values()
        self.calculate_errors(y_true, fitted_vol)

def run_garch_pipeline(y_returns, y_obs_vol, horizon=1):
    """
    - Instancie GARCHModel,
    - Fit sur y_returns,
    - Compare aux y_obs_vol in-sample,
    - Prévoit la vol sur 'horizon' pas,
    - Retourne un dict avec fitted, RMSE, MAE, forecast, etc.
    """
    model = GARCHModel()
    model.fit(y_returns)

    # Récupération de la volatilité in-sample
    fitted_vol = model.fitted_values()

    # Calcul des erreurs in-sample entre la vol fitted et la vol observée
    # (si c'est ce qu'on veut comparer)
    model.calculate_errors(y_obs_vol, fitted_vol)

    # Prévision horizon
    forecast_vol = model.predict_vol(horizon=horizon)

    # On renvoie un dict
    return {
        'Fitted_values': fitted_vol,
        'Residuals': model.residuals,
        'MAE': model.mae,
        'RMSE': model.rmse,
        'Forecast_vals': forecast_vol,
        "Shape": model.shape,
        "Params": model.params
    }

# --------------------------NNAR MODEL---------------------------------------

class NNARModel(BaseModel):
    """
    Une implémentation simple d'un NNAR (p, k) à la nnetar.
    
    - p = nombre de retards (lags).
    - k = nombre de neurones (units) dans la couche cachée (une seule couche).
    - activation = 'logistic' (sigmoid) ou 'relu', etc.
    
    Par défaut, on ne gère pas la saisonnalité ici, mais on peut l'ajouter
    en créant plus de colonnes de lags.
    """

    def __init__(self, p=5, k=None, activation='logistic', max_iter=1000, random_state=42):
        super().__init__(name='NNAR')
        self.p = p
        # Si l’utilisateur ne précise pas k, on met k = (p+1)//2 comme dans nnetar
        self.k = k if k is not None else (p + 1) // 2
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.mlp = None   # contiendra l'estimateur MLP
        self.y = None     # la série d'entraînement
        self.X = None     # matrice des features (lags)
        self.Y = None     # cible

    def _build_lagged_matrix(self, y, p):
        """
        Construit la matrice X et le vecteur Y pour les p retards.
        
        X[t] = [ y[t-1], y[t-2], ..., y[t-p] ]
        Y[t] = y[t]
        
        On saute les premiers p points qui ne peuvent pas former un lag complet.
        """
        X, Y = [], []
        for t in range(p, len(y)):
            X.append(y[t-p:t])   # p lags
            Y.append(y[t])       # valeur courante
        return np.array(X), np.array(Y)

    def fit(self, y):
        """
        Ajuste le modèle NNAR sur la série y.
        """
        self.y = np.array(y)
        
        # Construction de X, Y (retards)
        self.X, self.Y = self._build_lagged_matrix(self.y, self.p)
        # Instanciation du MLP: 1 couche cachée de self.k neurones
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(self.k,),
            activation=self.activation,
            solver='adam',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        # Entraînement
        self.mlp.fit(self.X, self.Y)

        # On peut stocker le "fitted model" dans self.fitted 
        self.fitted = self.mlp

    def fitted_values(self):
        """
        Renvoie les prédictions in-sample (dans la zone d'entraînement).
        Note: on ne peut prédire que pour les observations >= p.
        """
        if self.mlp is None:
            raise ValueError("Le modèle n'a pas encore été ajusté (fit).")

        # On prédit sur la même matrice X qu'on a utilisée à l'entraînement
        y_pred = self.mlp.predict(self.X)

        # Pour aligner avec la série initiale, on insère des NaN devant
        padding = [np.nan] * self.p
        return np.concatenate([padding, y_pred])

    def predict_nn(self, steps=1):
        """
        Prévision récursive sur 'steps' pas de temps.
        
        - On part de la dernière fenêtre de taille p (les p derniers points de self.y).
        - On prédit le point suivant, on l'ajoute à la fenêtre.
        - On répète jusqu'à 'steps' fois.
        
        Retourne un array de longueur 'steps' avec les prévisions.
        """
        if self.mlp is None:
            raise ValueError("Le modèle n'a pas encore été ajusté (fit).")

        # Derniers p points de la série connue
        last_history = self.y[-self.p:].tolist()  # on convertit en list pour faire append()

        preds = []
        for _ in range(steps):
            # On forme un vecteur (1, p) à partir des p dernières valeurs
            X_input = np.array(last_history[-self.p:]).reshape(1, -1)
            # Prédiction d'un pas
            y_next = self.mlp.predict(X_input)[0]
            preds.append(y_next)
            # On ajoute y_next dans l'historique
            last_history.append(y_next)

        return np.array(preds)
    
def run_nnar_pipeline(
    y_obs_vol,
    p_range=(1, 40),    # exploration de p=1..40
    k_range=(1, 40),    # exploration de k=1..40
    activation='logistic',
    max_iter=1000,
    horizon=1
):
    """
    1) Explore (p,k) dans p_range x k_range via un mini-grid search.
    2) Sélectionne la combinaison minimisant le RMSE in-sample.
    3) Reconstruit un NNARModel(p=k*, k=k*) final et fit sur y_obs_vol.
    4) Calcule le fitted in-sample, les erreurs, la prévision (horizon).
    5) Retourne un dictionnaire contenant les hyperparamètres optimaux 
       et les métriques (MAE, RMSE), etc.
    """
    best_rmse = np.inf
    best_p = None
    best_k = None
    for p_test in range(p_range[0], p_range[1] + 1):
        for k_test in range(k_range[0], k_range[1] + 1):

            model_temp = NNARModel(p=p_test, k=k_test, 
                                   activation=activation, max_iter=max_iter)
            model_temp.fit(y_obs_vol)

            fitted_vals_temp = model_temp.fitted_values()
            valid_idx = ~np.isnan(fitted_vals_temp)
            errs = y_obs_vol[valid_idx] - fitted_vals_temp[valid_idx]
            rmse_temp = np.sqrt(np.mean(errs**2))

            if rmse_temp < best_rmse:
                best_rmse = rmse_temp
                best_p = p_test
                best_k = k_test

    print(f"[NNAR GridSearch] Best p={best_p}, k={best_k}, RMSE={best_rmse:.4f}")

    # Reconstruit le modèle final avec (p*, k*) ---
    final_model = NNARModel(
        p=best_p, k=best_k, activation=activation, max_iter=max_iter
    )
    final_model.fit(y_obs_vol)

    #  Fitted in-sample
    fitted_vals = final_model.fitted_values()
    fv_series = pd.Series(fitted_vals, index=y_obs_vol.index)

    # Calcul erreurs + forecast
    valid_idx = ~np.isnan(fitted_vals)
    final_model.calculate_errors(y_obs_vol[valid_idx], fitted_vals[valid_idx])
    forecast_vals = final_model.predict_nn(steps=horizon)

    # Returns du dictionnaire
    return {
        'Fitted_values': fv_series,
        'Residuals': final_model.residuals,
        'MAE': final_model.mae,
        'RMSE': final_model.rmse,
        'Forecast_vals': forecast_vals,
        'NNAR_order': (best_p, best_k),
    }
# -------------------------------------------------------------------------
# SS ARIMA MODEL

# On commence par faire le test qui permet d'optimiser l'ordre d'intégration 
  
# def optimal_integration_order(time_series, max_order=3, alpha=0.05):
#     """
#     Imitation de 'optimal_integration_order' en R :
#     répète le test KPSS pour déterminer d (ordre d'intégration).
#     """
#     def kpss_pvalue(series):
#         stat, p_val, _, _ = kpss(series, regression='c')
#         return p_val

#     d = 0
#     p_val = kpss_pvalue(time_series)
#     if p_val < alpha:
#         for order in range(1, max_order+1):
#             diff_series = np.diff(time_series, n=order)
#             try:
#                 p_val_diff = kpss_pvalue(diff_series)
#             except:
#                 p_val_diff = 0.0
#             if p_val_diff >= alpha:
#                 d = order
#                 break
#     return d

# def find_best_arima_order(y, 
#                           p_range=(0, 5), 
#                           d_range=(0, 2), 
#                           q_range=(0, 5), 
#                           ic='aic'):
#     """
#     Recherche brute de (p, d, q) minimisant un critère (AIC ou BIC) 
#     via SARIMAX de statsmodels, sans pmdarima.
    
#     y : la série univariée (numpy array ou pandas Series)
#     p_range : tuple (min_p, max_p)
#     d_range : tuple (min_d, max_d)
#     q_range : tuple (min_q, max_q)
#     ic : 'aic' ou 'bic'
    
#     Retourne (p_best, d_best, q_best, best_ic_value)
#     """
#     best_ic = np.inf
#     best_order = (0,0,0)
#     for p in range(p_range[0], p_range[1]+1):
#         for d in range(d_range[0], d_range[1]+1):
#             for q in range(q_range[0], q_range[1]+1):
#                 try:
#                     model = sm.tsa.SARIMAX(y, order=(p,d,q), trend='n',
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#                     res = model.fit(disp=False)
#                     if ic == 'aic':
#                         current_ic = res.aic
#                     else:
#                         current_ic = res.bic
                        
#                     if current_ic < best_ic:
#                         best_ic = current_ic
#                         best_order = (p,d,q)
#                 except:
#                     # Peut arriver si le modèle ne converge pas
#                     continue
#     return best_order[0], best_order[1], best_order[2], best_ic

# def ss_kf_fit(returns, observed_volatility):
#     """
#     Reproduit la logique du script R:
#     - Trouve d par test KPSS (sur 'returns', comme le code R)
#     - Trouve (p,d,q) par auto_arima(returns, d=...)
#     - Construit SARIMAX(endog=observed_volatility, order=(p,d,q))
#     - Fit => applique KF en interne
#     - Compare fitted_values vs. observed_volatility => MAE, RMSE
#     - Fait un forecast(1-step)
#     """

#     # Trouver d
#     d_opt = 0  # par exemple, ou votre fonction optimal_integration_order()
#     #Recherche (p, d, q) par mini-grid search sur 'returns'
#     p_opt, d_opt, q_opt, best_ic = find_best_arima_order(
#         returns, 
#         p_range=(0,5), 
#         d_range=(0,2), 
#         q_range=(0,5), 
#         ic='aic'
#     )
#     print(f"Best ARIMA order via grid search: p={p_opt}, d={d_opt}, q={q_opt} (AIC={best_ic:.2f})")

#     # Construire un SARIMAX sur observed_volatility 
#     #    => c'est statsmodels' state-space ARIMA
#     sarimax_mod =    sm.tsa.statespace.SARIMAX(
#         observed_volatility, 
#         order=(p_opt, d_opt, q_opt),
#         trend='n', 
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     )

#     # Fit => MLE + Kalman Filter
#     res = sarimax_mod.fit(disp=False)

#     # Comparaison in-sample
#     fitted_vals = res.fittedvalues  # .fittedvalues = E[y_t|t-1], version statsmodels
#     # On calcule l'erreur vs. 'observed_volatility'
#     residuals = observed_volatility - fitted_vals
#     mae = np.mean(np.abs(residuals))
#     rmse = np.sqrt(np.mean(residuals**2))

#     # 6) Forecast(1-step)
#     forecast_1step = res.forecast(steps=1).iloc[0]

#     return {
#         'Model': res,
#         'Fitted_values': fitted_vals,
#         'Residuals': residuals,
#         'Forecast_vals': forecast_1step,
#         'MAE': mae,
#         'RMSE': rmse,
#         'ARIMA_order': (p_opt, d_opt, q_opt),
#         'AIC': best_ic,
#         # Coefficients estimés (AR, MA, etc.)
#         'EstParams': res.params
#     }

## --------------- SS ARIMA ALEX ----------------------

def optimal_integration_order(time_series, max_order=2, alpha=0.05):
    """Find optimal integration order using KPSS test"""
    def kpss_test(series):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, p_value, _, _ = kpss(series, regression='c', nlags="auto")
            return p_value

    order = 0
    p_value = kpss_test(time_series)
    
    if p_value < alpha:
        for order in range(1, max_order + 1):
            diff_series = np.diff(time_series, n=order)
            p_value = kpss_test(diff_series)
            if p_value >= alpha:
                return order
    return order

def find_optimal_arima(time_series, integration_order, max_p=5, max_q=5):
    """Find optimal ARIMA model parameters using AIC"""
    best_aic = float('inf')
    best_model = None
    best_order = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(time_series, order=(p, integration_order, q))
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_order = (p, integration_order, q)
            except:
                continue

    if best_model is None:
        raise ValueError("Could not fit any ARIMA models")

    return {
        "Optimal orders": best_order,
        "AIC": best_aic,
        "AR coefficients": best_model.arparams if best_model.arparams.size > 0 else [],
        "MA coefficients": best_model.maparams if best_model.maparams.size > 0 else []
    }
    
def ss_arima_kf_pipeline(volatility, obs_index=None):
    """
    Pipeline for SSARIMA + Kalman Filter, extended to compute the 
    standard deviation of the T+1 forecast.
    
    Returns:
    --------
        Dictionary containing the model, fitted values, residuals, forecasted variance,
        forecasted standard deviation, ARIMA order, AIC, and parameters.
    """
    # Compute variance from returns if needed
    variance_series = volatility ** 2

    # Drop NaNs
    variance_series = variance_series.dropna()
    if obs_index is not None:
        variance_series = variance_series.reindex(obs_index).dropna()

    var_array = variance_series.values

    # Determine integration order d
    d_opt = optimal_integration_order(var_array, max_order=2, alpha=0.05)

    # Find best ARIMA parameters
    arima_info = find_optimal_arima(var_array, d_opt, max_p=5, max_q=5)
    p_opt, d_, q_opt = arima_info["Optimal orders"]
    best_aic = arima_info["AIC"]
    phi = arima_info["AR coefficients"]
    theta = arima_info["MA coefficients"]

    # Build the SSARIMA model
    ss_model = SSARIMA(d=d_opt, phi=phi, theta=theta)

    # Apply Kalman Filter
    kf_result = KalmanFilter(ss_model, var_array)

    # Get fitted values
    filtered_array = np.array(kf_result["filtered states"], dtype=float)
    fitted_series = pd.Series(filtered_array, index=obs_index) if obs_index is not None else pd.Series(filtered_array)

    #Compute residuals
    residuals_array = var_array - filtered_array if len(var_array) == len(filtered_array) else None
    residuals_series = pd.Series(residuals_array, index=obs_index) if residuals_array is not None and obs_index is not None else pd.Series(residuals_array)

    # Compute one-step-ahead forecast and standard deviation
    final_posterior = kf_result["state estimates"][-1]
    process_noise = gaussian(np.array([[0.0]]), ss_model.Q)
    out_of_sample_prior = predict(final_posterior, process_noise, ss_model.T, ss_model.R)
    forecast_1step_mean = float(sum(out_of_sample_prior.mean[:ss_model.d]))
    forecast_1step_std = float(np.sqrt(sum(out_of_sample_prior.var[:ss_model.d, :ss_model.d])))

    # 9) Return results in dictionary format
    result_dict = {
        "Model": ss_model,
        "Fitted_values": fitted_series,
        "Residuals": residuals_series,
        "Forecast_vals": {
            "mean": forecast_1step_mean,
            "std": forecast_1step_std
        },
        "MAE": None,  
        "RMSE": None,
        "ARIMA_order": (p_opt, d_opt, q_opt),
        "AIC": best_aic,
        "EstParams": {
            "phi": phi,
            "theta": theta
        }
    }

    return result_dict

#------------------------MODELE VECH --------------------------------
def vech(matrix):
    """Transform a square matrix into a vector by stacking the upper triangular part"""
    return matrix[np.triu_indices(matrix.shape[0])]


class VECHModel:
    def __init__(self, max_p=2, max_q=2, n_tries=3):
        self.max_p = max_p
        self.max_q = max_q
        self.n_tries = n_tries
    
        self.p = None
        self.q = None
        self.gamma = None
        self.A_matrices = []
        self.B_matrices = []
        self.likelihood = None
        self.aic = None
        self.bic = None
        self.is_fitted = False
    
    def _series_to_vech(self, variances, covariances):
        if not isinstance(variances, np.ndarray):
            variances = np.array(variances)
        if not isinstance(covariances, np.ndarray):
            covariances = np.array(covariances)
            
        T, n = variances.shape
        expected_cov_dim = n * (n - 1) // 2
        
        if covariances.shape[1] != expected_cov_dim:
            raise ValueError(f"For {n} assets there should be {expected_cov_dim} covariances")
        
        vech_dim = n * (n + 1) // 2
        vech_data = np.zeros((T, vech_dim))
        vech_data[:, :n] = variances
        vech_data[:, n:] = covariances
        
        return vech_data
    
    def _calculate_likelihood(self, params, vech_data, residuals, p, q, n):
        T = len(vech_data)
        m = n * (n + 1) // 2
        
        gamma = params[:m]
        A_diags = params[m:m*(1+q)].reshape(q, m)
        B_diags = params[m*(1+q):m*(1+q+p)].reshape(p, m)
        
        if any(gamma[:n] <= 0) or any(A_diags.flatten() < 0) or any(B_diags.flatten() < 0):
            return np.inf
        
        H_t = np.zeros((T, m))
        H_t[0] = vech_data[0]
        
        try:
            for t in range(max(p,q), T):
                H_t[t] = gamma.copy()
                
                for j in range(q):
                    if t-j-1 >= 0:
                        H_t[t] += A_diags[j] * residuals[t-j-1]
                
                for j in range(p):
                    if t-j-1 >= 0:
                        H_t[t] += B_diags[j] * H_t[t-j-1]
                
                if any(H_t[t, :n] <= 0):
                    return np.inf
                
            diff = vech_data[max(p,q):] - H_t[max(p,q):]
            ll = np.sum(diff ** 2)
            
        except:
            return np.inf
            
        return ll
    
    def _calculate_aic_bic(self, likelihood, n_params, T):
        aic = 2 * n_params + 2 * likelihood
        bic = np.log(T) * n_params + 2 * likelihood
        return aic, bic
    
    def fit(self, variances, covariances):
        vech_data = self._series_to_vech(variances, covariances)
        T, m = vech_data.shape
        n = variances.shape[1]
        
        residuals = np.diff(vech_data, axis=0)
        
        best_result = {'likelihood': np.inf}
        
        for p, q in product(range(1, self.max_p + 1), range(1, self.max_q + 1)):
            n_params = m + m*q + m*p 
            
            for _ in range(self.n_tries):
                initial_gamma = np.concatenate([
                    np.abs(np.random.randn(n)) + 0.1,
                    np.random.randn(m-n) * 0.1
                ])
                initial_A = np.abs(np.random.randn(q * m)) * 0.1
                initial_B = np.abs(np.random.randn(p * m)) * 0.1
                initial_params = np.concatenate([initial_gamma, initial_A, initial_B])
                
                bounds = [(1e-6, None)] * n + [(None, None)] * (m-n) + [(0, None)] * (m*(p+q))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        self._calculate_likelihood,
                        initial_params,
                        args=(vech_data, residuals, p, q, n),
                        method='L-BFGS-B',
                        bounds=bounds
                    )
                
                if result.fun < best_result['likelihood']:
                    aic, bic = self._calculate_aic_bic(result.fun, n_params, T)
                    
                    self.gamma = result.x[:m]
                    
                    # Extraction des matrices dans le bon ordre
                    A_diags = result.x[m:m*(1+q)].reshape(q, m)
                    B_diags = result.x[m*(1+q):m*(1+q+p)].reshape(p, m)
                    
                    self.A_matrices = [np.diag(diag) for diag in A_diags]
                    self.B_matrices = [np.diag(diag) for diag in B_diags]
                    
                    self.p = p
                    self.q = q 
                    self.likelihood = result.fun
                    self.aic = aic
                    self.bic = bic
                    
                    best_result = {
                        'likelihood': result.fun,
                        'convergence': result.success,
                        'message': result.message
                    }
        
        if self.likelihood == np.inf:
            raise RuntimeError("Could not fit the model")
            
        self.is_fitted = True
        return self
    
    def predict(self, last_variances, last_covariances, last_residuals, n_periods=1):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
            
        n = len(last_variances)
        m = n * (n + 1) // 2
        
        last_vech = self._series_to_vech(
            last_variances.reshape(1, -1),
            last_covariances.reshape(1, -1)
        )
        
        predictions = []
        current_vech = last_vech[-1]
        current_residuals = last_residuals[-self.q:]
        
        for _ in range(n_periods):
            next_vech = self.gamma.copy()
            
            for j, A in enumerate(self.A_matrices):
                next_vech += A @ current_residuals[-(j+1)]
            
            for j, B in enumerate(self.B_matrices):
                if j < len(predictions):
                    next_vech += B @ predictions[-(j+1)]
                else:
                    next_vech += B @ current_vech
            
            predictions.append(next_vech)
            
        predictions = np.array(predictions)
        
        return predictions[:, :n], predictions[:, n:]
    
    @staticmethod
    def simulate_data(T=500, n_assets=2, seed=42):
        np.random.seed(seed)
        
        base_variance = 0.01
        persistence = 0.94
        
        returns = np.zeros((T, n_assets))
        variances = np.zeros((T, n_assets))
        covariances = np.zeros((T, n_assets * (n_assets - 1) // 2))
        
        variances[0] = base_variance
        covariances[0] = base_variance * 0.5
        
        for t in range(1, T):
            variances[t] = base_variance * (1 - persistence) + persistence * variances[t-1] + \
                          0.1 * np.random.randn(n_assets) * np.sqrt(variances[t-1])
            variances[t] = np.maximum(variances[t], 1e-6)
            
            covariances[t] = base_variance * 0.5 * (1 - persistence) + persistence * covariances[t-1] + \
                            0.05 * np.random.randn(n_assets * (n_assets - 1) // 2) * \
                            np.sqrt(np.mean(variances[t]))
            
            cov_matrix = np.zeros((n_assets, n_assets))
            np.fill_diagonal(cov_matrix, np.sqrt(variances[t]))
            cov_matrix[0,1] = cov_matrix[1,0] = covariances[t,0]
            
            returns[t] = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=cov_matrix @ cov_matrix.T
            )
        
        return returns, variances, covariances
    

