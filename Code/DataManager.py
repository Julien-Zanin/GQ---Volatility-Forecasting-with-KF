
import pandas as pd 
import numpy as np 

class DataManager:
    def __init__(self, csv_path):
        """Initialize with path to CSV."""
        self.csv_path = csv_path
        self.data = None
        self.pre_covid_data = None
        self.covid_data = None

    def load_and_clean_data(self, date_col='date', price_col='close', sep=',', rows_to_skip=1, dayfirst=False):
        """Load CSV, sort, drop duplicates, etc.
        dayfirst : il faut connaitre le format de la date pour le parser correctement
        data 1 min : False
        data du papier : True
        mes données : False
        """
        
        df = pd.read_csv(self.csv_path, skiprows=rows_to_skip)
            # On rajoute un try except pour gérer les erreurs de parsing de date DE LEUR DATASET TOUT NUL PUTAIN Y'A RIEN QUI VA 
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce',dayfirst=dayfirst)

            # On vérifie si au moins toutes les dates sont au même format
            if df[date_col].isna().any():
                raise ValueError("Date parsing failed for some rows.")

        except Exception as e:
            print(f"Warning: Date parsing failed with default settings. Attempting fallback. ({e})")
            
            # Si y'a un format différent (e.g., DD/MM/YYYY H:MM)
            try:
                df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y %H:%M', errors='coerce',dayfirst=True)
            except Exception as e:
                print(f"Error: Custom fallback also failed. ({e})")
                raise ValueError("Date parsing failed. Please check the date format in the file.")

        
        df = df.sort_values(by=date_col).drop_duplicates()
        
        # For safety, rename columns
        df.rename(columns={price_col: 'Price', date_col: 'Date'}, inplace=True)
        df.set_index('Date', inplace=True)        
        self.data = df
        
        
    def split_pre_and_post_covid(self, covid_start_date='2020-02-26 00:00:00', covid_end_date="2021-04-18 00:00:00"):
        """
        Splits the dataset into pre-covid and covid periods (or any date you want).
        The user can define the date that marks the boundary.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_clean_data first.")
        mask_pre = (self.data.index < covid_start_date) & (self.data.index > "2019-02-25 00:00:00")
        mask_covid = (self.data.index >= covid_start_date) & (self.data.index <= covid_end_date)

        self.pre_covid_data = self.data.loc[mask_pre].copy()
        self.covid_data = self.data.loc[mask_covid].copy()

    def get_data_timeperiod(self, subset='full'):
        """
        Returns the log-returns of whichever subset is requested:
        'full', 'pre', or 'covid'.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_clean_data first.")

        if subset == 'full':
            df = self.data
        elif subset == 'pre':
            df = self.pre_covid_data
        elif subset == 'covid':
            df = self.covid_data
        else:
            raise ValueError("subset must be one of 'full', 'pre', or 'covid'.")

        return df
        
#-----------Log Returns basiques et centrées réduites----------------
            
    def get_log_returns(self, df):
        df_copy = df.copy()
        df_copy['LogReturn'] = np.log(df_copy['Price'] / df_copy['Price'].shift(1))
        return df_copy['LogReturn'].dropna()
    
    def get_centred_log_returns(self,df):
        df_copy = df.copy()
        df_copy['LogReturn'] = np.log(df_copy['Price'] / df_copy['Price'].shift(1))
        mean_r = df_copy['LogReturn'].mean()
        std_r  = df_copy['LogReturn'].std()
        df_copy['CenteredReducedLogReturn'] = (df_copy['LogReturn'] - mean_r) / std_r
        return df_copy['CenteredReducedLogReturn'].dropna()        

#----------- Log Returns à partir des minutes ----------------

    def get_log_returns_from_min_freq(self, df):
        df_copy = df.copy()
        hourly_prices = df_copy['Price'].resample('H').first()
        
        # Rendement pour l'heure N : log(prix à l'heure N / prix à l'heure N-1)
        hourly_log_returns = np.log(hourly_prices / hourly_prices.shift(1))
        return hourly_log_returns.dropna()
#----------- Log Returns centrés réduits à partir des minutes ----------------    
    def get_centred_log_returns_from_min_freq(self,df):
        df_copy = df.copy()
        hourly_prices = df_copy['Price'].resample('H').first()
        df_copy['LogReturn'] = np.log(hourly_prices / hourly_prices.shift(1))
        mean_r = df_copy['LogReturn'].mean()
        std_r  = df_copy['LogReturn'].std()
        df_copy['CenteredReducedLogReturn'] = (df_copy['LogReturn'] - mean_r) / std_r
        return df_copy['CenteredReducedLogReturn'].dropna()    
    