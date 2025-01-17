import pandas as pd 
import numpy as np
from scipy.stats import t, norm
from scipy.optimize import minimize
def descriptive_statistics(returns_series,arima_params, nnar_params):
    """
    Compute maximum, minimum, mean, median, std, skewness, kurtosis.
    """
    desc = {}
    
    desc['Maximum'] = returns_series.max()
    desc['Minimum'] = returns_series.min()
    desc['Mean'] = returns_series.mean()
    desc['Median'] = returns_series.median()
    desc['Std'] = returns_series.std()

    desc['Skewness'] = returns_series.skew()
    desc['Kurtosis'] = returns_series.kurt()
    
    # Ajout d'une colonne "arima_params" et "NNAR_Params"
    desc['ARIMA params'] = str(arima_params)
    desc['NNAR params']  = str(nnar_params)
    
    return desc

def observed_volatility(returns):
    """
    Calcule la volatilité observée à partir des log rendements centrés réduits en fonction de la formule 
    dans le code du papier de recherche  à savoir : 
    rho = mu / sigma_returns
    obs_vol = abs(excess_returns) / rho
    """
    sigma_returns = returns.std()
    excess_returns = returns - returns.mean()
    abs_excess_returns = abs(excess_returns)
    mu = abs_excess_returns.mean()
    rho = mu / sigma_returns
    obs_vol = abs_excess_returns / rho
    return obs_vol


def rolling_std(returns): 
    W = 24  # fenêtre de 24 heures
    return returns.rolling(window=W).std()

#-------------- Volatilité observée par rendements minutes  ----------------
def obs_vol_by_mins(dfminutes):
    df_copy = dfminutes.copy()
    df_copy.reset_index(inplace=True)
    df_copy = df_copy[['Date', 'Price']]
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy['Min_log_return'] = np.log(df_copy['Price'] / df_copy['Price'].shift(1))
    df_copy['Minutes'] = df_copy['Date'].dt.minute
    df_copy['hourly_volatility'] = np.nan

    for i in range(len(df_copy)):
        if df_copy.loc[i, 'Minutes'] == 0 and i >= 60:  # Première minute de chaque heure
                # Calculer l'écart-type des 60 dernières minutes puis convertir sur une base horaire
            past_60_returns = df_copy.loc[(i-60)+1:i, 'Min_log_return']
            hourly_vol = np.std(past_60_returns) * np.sqrt(60)
                
                # Assigner la volatilité à la ligne correspondante
            df_copy.loc[i, 'hourly_volatility'] = hourly_vol
                
    # Retirer les lignes sans volatilité calculée
    df_copy = df_copy.dropna(subset=['hourly_volatility'])
    
    # Retourner uniquement la colonne 'hourly_volatility' comme une Series
    return df_copy.set_index('Date')['hourly_volatility']

#-------------- Recupère une valeur de la volatilité prédite pour chaque modèle ----------------
def single_vol_value(forecast_vals, method="first"):
    """
    Convertit forecast_vals (qui peut être un float, une Series, un array)
    en un float unique selon la méthode.
    
    method peut être "mean", "last", "first", etc.
    
    Globalement on utilise que la première valeur de forecast_vals pour calculer la VaR 
    Mais si on a besoin de la moyenne ou de la dernière valeur, on peut le faire.
    """

    if isinstance(forecast_vals, pd.Series):
        if method == "mean":
            return forecast_vals.mean()
        elif method == "last":
            return forecast_vals.iloc[-1]
        elif method == "first":
            return forecast_vals.iloc[0]
        else:
            raise ValueError(f"unknown method={method}")
    
    if isinstance(forecast_vals, np.ndarray):
        if method == "mean":
            return forecast_vals.mean()
        elif method == "last":
            return forecast_vals[-1]
        elif method == "first":
            return forecast_vals[0]
        else:
            raise ValueError(f"unknown method={method}")
    
    return float(forecast_vals)

def hourly_var(df):
    """Compute the hourly variance of ta price series"""
    df = df.reset_index()
    df['Timestamp'] = pd.to_datetime(df['timestamp'])
    df['Hour'] = df['Timestamp'].dt.floor('h')
    df['log_returns'] = np.log(df.iloc[:, 1] / df.iloc[:, 1].shift(1))
    variance = df.groupby('Hour').agg({'log_returns': lambda x: np.var(x.dropna(), ddof=1) * 60}).reset_index()
    variance.columns = ['Timestamp', 'Variance']
    variance.set_index('Timestamp', inplace=True)
    return variance

def hourly_cov(df):
    """Compute the hourly covariance of two series of prices"""
    df = df.reset_index()
    df['Timestamp'] = pd.to_datetime(df['timestamp'])
    df['Hour'] = df['Timestamp'].dt.floor('h')
    
    df['log_returns_1'] = np.log(df.iloc[:, 1] / df.iloc[:, 1].shift(1))
    df['log_returns_2'] = np.log(df.iloc[:, 2] / df.iloc[:, 2].shift(1))
    
    hourly_cov_df = df.groupby('Hour').apply(lambda x: x['log_returns_1'].cov(x['log_returns_2']) * 60).reset_index()
    hourly_cov_df.columns = ['Timestamp', 'Covariance']
    hourly_cov_df.set_index('Timestamp', inplace=True)
    
    return hourly_cov_df

def compute_volatility_ci(models_data, alpha_array):
    """
    Calcule l'intervalle de confiance (IC) de la volatilité pour 3 modèles (GARCH, NNAR, SS-KF).
    Renvoie un DataFrame par modèle (lower, upper) indexé par alpha_array.
    
    Paramètres
    ----------
    models_data : dict
        ex: {
            "GARCH": {"Forecast": 0.015, "RMSE": 0.007, "Shape": 8,
            "NNAR":  {"Forecast": 0.014, "RMSE": 0.006
            "SS-KF": {"Forecast": 0.016, "RMSE": 0.0065
        }
    alpha_array : array-like
        ex: np.arange(0.01, 1.01, 0.01)
    
    Retour
    ------
    dict
       {
         "GARCH": DataFrame(lower, upper),
         "NNAR":  DataFrame(lower, upper),
         "SS-KF": DataFrame(lower, upper)
       }
    """
    vol_values = []
    
    for model_name, model_dict in models_data.items():
        fvals = model_dict["Forecast_vals"]  
        vol_float = single_vol_value(fvals, method="first")
        vol_values.append(vol_float)

    mean_vol = np.mean(vol_values)
    alpha_aindex= pd.Index(alpha_array, name="alpha")
    
    ci_dict = {}
    
    for model_name, model_data in models_data.items():
        
        rmse = model_data["RMSE"] / np.sqrt(10000) # On reprend la logique du code R, un ptf de 10K 
        dist_type = "student" if model_name == "GARCH" else "normal"
        df = model_data.get("Shape", None)

        if dist_type =="student":
            zvals = np.abs(t.ppf(alpha_array/2, df))
        else:
            zvals = np.abs(norm.ppf(alpha_array/2))
        
        lower_bound = mean_vol - zvals * rmse
        upper_bound = mean_vol + zvals * rmse
        
        ci_dict[model_name] = pd.DataFrame(
            {"lower": lower_bound, "upper": upper_bound},
            index=alpha_aindex
        )
        
    return ci_dict

#----------- Version statistiquement correcte du calcul de la VaR ----------------
def compute_var_ci(vol_ci_dict, alpha_array, models_data, alpha_var=0.01, nominal=10_000):
    
    """
    Calcule un IC sur la VaR, en s'appuyant sur l'IC de volatilité 'vol_ci_dict' 
      VaR = - nominal * vol * quantile * sqrt((df - 2)/df) (pour Student)
           = - nominal * vol * quantile (pour Normal)
    
    Paramètres
    ----------
    vol_ci_dict : dict
       La sortie de compute_volatility_ci(...) => { "GARCH": DF, "NNAR": DF, "SS-KF": DF }
    alpha_array : array-like
       identique à celui utilisé pour vol_ci (index)
    models_data : dict
       ex: {
         "GARCH": {"vol_forecast":..., "rmse":..., "df":..., "dist":"student"},
         ...
       }
    alpha_var : float
       Niveau de VaR (ex: 0.01 pour 1%)
    nominal : float
       Montant du portefeuille, ex: 10,000
    
    Retour
    ------
    dict
       { "GARCH": DataFrame(lower, upper), ... }
    """
    
    var_ci_dict = {}
    alpha_index = pd.Index(alpha_array, name="alpha")
    
    for model_name, df_ci in vol_ci_dict.items():
         dist_type = "student" if model_name == "GARCH" else "normal"
         df = models_data[model_name].get("Shape", None)
         #Two-sided IC 
         
         if dist_type == "student" and df is not None:
            q = t.ppf(alpha_var/2, df)
            # Facteur de student 
            student_factor = np.sqrt((df-2)/df)
         else: 
            q = norm.ppf(alpha_var/2)
            student_factor = 1.0

         vol_lower = df_ci["lower"].values
         vol_upper = df_ci["upper"].values

         #Calcul de la VaR
         var_lower = - nominal * vol_lower * q * student_factor 
         var_upper = - nominal * vol_upper * q * student_factor 
         
         var_ci_dict[model_name] = pd.DataFrame(
             {"lower": var_lower, 
              "upper": var_upper},
             index=alpha_index
         )    
         
    return var_ci_dict

#-------------V2 AVEC DEGRES LIBERTES DU GARCH POUR PRESENTATION RESULTATS ---------------
def compute_var_ci_df(vol_ci_dict, alpha_array, models_data, alpha_var=0.01, nominal=10_000):
    
    """
    Calcule un IC sur la VaR, en s'appuyant sur l'IC de volatilité 'vol_ci_dict' 
    ET en reproduisant la logique du papier R :
    On applique les degrés de liberté du GARCH pour tous les modèles pour le calcul
    de la VaR, présentant ainsi les résultats de manière plus homogène.
    
    Paramètres
    ----------
    vol_ci_dict : dict
       La sortie de compute_volatility_ci(...) => { "GARCH": DF, "NNAR": DF, "SS-KF": DF }
    alpha_array : array-like
       identique à celui utilisé pour vol_ci (index)
    models_data : dict
       ex: {
         "GARCH": {"vol_forecast":..., "rmse":..., "df":..., "dist":"student"},
         ...
       }
    alpha_var : float
       Niveau de VaR (ex: 0.01 pour 1%)
    nominal : float
       Montant du portefeuille, ex: 10,000
    
    Retour
    ------
    dict
       { "GARCH": DataFrame(lower, upper), ... }
    """
    
    var_ci_dict = {}
    alpha_index = pd.Index(alpha_array, name="alpha")
    # Récupère la distribution et le degré de liberté
    df = models_data["GARCH"].get("Shape", None)
    
    #Two-sided IC 
    q = t.ppf(alpha_var/2, df)
    student_factor = np.sqrt((df-2)/df)

    for model_name, df_ci in vol_ci_dict.items():
         
         vol_lower = df_ci["lower"].values
         vol_upper = df_ci["upper"].values

         #Calcul de la VaR
         var_lower = - nominal * vol_lower * q * student_factor *(-1) 
         var_upper = - nominal * vol_upper * q * student_factor *(-1) 
         
         var_ci_dict[model_name] = pd.DataFrame(
             {"lower": var_lower, 
              "upper": var_upper},
             index=alpha_index
         )    
         
    return var_ci_dict