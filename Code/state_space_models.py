from dataclasses import dataclass, field
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from typing import Union
from stats import *

####################################################################################################
################### STATE-SPACE ARIMA MODEL ########################################################
####################################################################################################

@dataclass
class SSARIMA:
    d: int  # Optimal order of integration
    phi: list  # AR coefficients
    theta: list  # MA coefficients
    sigma2: float = 1  # Model noise variance
    obs_noise: float = 1e-8  # Measurement noise variance

    Z: np.ndarray = field(init=False)
    T: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    Q: float = field(init=False)
    a1: np.ndarray = field(init=False)
    P1: np.ndarray = field(init=False)
    P1inf: np.ndarray = field(init=False)
    model: str = field(init=False)

    def __post_init__(self):
        """Initialize state space matrices."""
        p = len(self.phi) 
        q = len(self.theta)
        r = max(p, q+1) 
        m = self.d + r

        self.model = f"ARIMA({p},{self.d},{q})"

        self.phi = np.pad(self.phi, (0, r - p), 'constant')
        self.theta = np.pad(self.theta, (0, r - q), 'constant')

        # Measurement equation matrices
        self.Z = np.zeros(m)
        self.Z[:self.d + 1] = 1
        self.Z = self.Z.reshape(1, m)

        self.H = np.array([[self.obs_noise]])

        # State equation matrices
        self.T = np.zeros((m, m))
        self.T[:self.d, :self.d] = np.tri(self.d, self.d, 0, dtype=float).T
        self.T[:self.d, self.d] = 1
        self.T[self.d:, self.d] = self.phi
        for i in range(1, r):
            self.T[self.d+i-1, self.d + i] = 1

        # R comme matrice colonne (m x 1)
        self.R = np.zeros((m, 1))
        self.R[self.d] = 1
        for i in range(r-1):
            self.R[self.d + i + 1] = self.theta[i]

        self.Q = np.array([[self.sigma2]])

        # Initial state
        self.a1 = np.zeros(m).reshape(m,1)
        self.P1 = np.zeros((m, m))
        self.P1[self.d:, self.d:] = np.eye(m - self.d)
        self.P1inf = np.zeros((m, m))
        self.P1inf[:self.d, :self.d] = np.eye(self.d)
    
    def summary(self):
        """Dislay the summary of the State-Space Model."""
        print("State-Space ARIMA Model:")
        print("*"*30)
        print(self.model)
        print(f"AR coefficients (phi): {self.phi}")
        print(f"MA coefficients (theta): {self.theta}")
        print(f"Variance of noise (sigma^2): {self.sigma2}")
        print("*"*30)
        print("Matrices:")
        print(f"Z: \n{self.Z}\n")
        print("-"*25)
        print(f"H: \n{self.H}\n")
        print("-"*25)
        print(f"T: \n{self.T}\n")
        print("-"*25)
        print(f"R: \n{self.R}\n")
        print("-"*25)
        print(f"Q: \n{self.Q}\n")
        print("-"*25)
        print(f"a1: \n{self.a1}\n")
        print("-"*25)
        print(f"P1: \n{self.P1}\n")
        print("-"*25)
        print(f"P1inf: \n{self.P1inf}\n")


####################################################################################################
################### STATE-SPACE VECH MODEL #########################################################
####################################################################################################

@dataclass
class SSVECH:
    n: int
    A: Union[list, np.ndarray]
    B: Union[list, np.ndarray]
    gamma: np.ndarray
    Q: np.ndarray = None  # Model noise variance
    observation_noise: int = 1 # Measurement noise variance

    Z: np.ndarray = field(init=False)
    T: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    a1: np.ndarray = field(init=False)
    P1: np.ndarray = field(init=False)
    P1inf: np.ndarray = field(init=False)
    model: str = field(init=False)

    def __post_init__(self):
        self.m = self.n * (self.n + 1) // 2
        self.A = self.A if isinstance(self.B, list) else list(self.A)
        self.B = self.B if isinstance(self.B, list) else list(self.B)
        self.q = len(self.A)
        self.p = len(self.B)
        self.dim = self.m * (self.p +self.q) + 1
        if self.Q is None:
            self.Q = 10 * np.eye(self.dim)
            self.Q[0, 0] = 0
        self.H = self.observation_noise * np.eye(self.m)

        self.T = self._load_T()
        self.R = self._load_R()
        self.Z = self._load_Z()

        self.a1 = np.zeros((self.dim,1))
        self.P1 = np.eye(self.dim)

    def _load_T(self):
        T = np.zeros((self.dim, self.dim))

        T[0, 0] = 1

        for i in range(self.m):
            T[1 + i, 0] = self.gamma[i]
            for j, B in enumerate(self.B):
                T[1 + i, 1 + j * self.m: 1 + (j + 1) * self.m] = B[i]

            for j, A in enumerate(self.A):
                T[1 + i, 1 + self.p * self.m + j * self.m: 1 + self.p * self.m + (j + 1) * self.m] = A[i]

        for i in range(self.p - 1):
            start_row = 1 + (i + 1) * self.m
            end_row = start_row + self.m
            T[start_row:end_row, 1 + i * self.m: 1 + (i + 1) * self.m] = np.eye(self.m)

        for i in range(self.q - 1):
            start_row = 1 + (1 + self.p + i) * self.m
            start_col = 1 + self.p * self.m + i * self.m
            T[start_row:start_row+self.m, start_col:start_col+self.m] = np.eye(self.m)

        return T

    def _load_R(self):
        R = np.zeros((self.dim, self.dim * (self.dim + 1) // 2))

        start_row = 1 + self.p * self.m
        start_col = self.dim

        R[start_row:start_row+self.m,start_col:start_col+self.m] = np.eye(self.m)

        return R
    
    def _load_Z(self):
        Z = np.zeros((self.m, self.dim))
        Z[0:self.m,1:self.m+1]= np.eye(self.m)
        return Z
    
    def summary(self):
        """Dislay the summary of the State-Space Model."""
        print("State-Space VECH Model:")
        print("*"*30)
        print("AR matrices (B):")
        for i,mat in enumerate(self.B):
            print(f"B_{i}:")
            print(mat)
        print("-"*25)
        print("MA matrices (A):")
        for i,mat in enumerate(self.A):
            print(f"A_{i}:")
            print(mat)
        print("-"*25)
        print("Constants:")
        print(self.gamma)
        print("*"*30)
        print("Matrices:")
        print(f"Z: \n{self.Z}\n")
        print("-"*25)
        print(f"H: \n{self.H}\n")
        print("-"*25)
        print(f"T: \n{self.T}\n")
        print("-"*25)
        print(f"R: \n{self.R}\n")
        print("-"*25)
        print(f"Q: \n{self.Q}\n")
        print("-"*25)
        print(f"a1: \n{self.a1}\n")
        print("-"*25)
        print(f"P1: \n{self.P1}\n")

    def state_transform(self, sigma_point):
        return self.T @ sigma_point
        
    def residuals_transform(self, eta):
        return self.R @ (vech(np.outer(eta,eta)))
    
    def measurement_transform(self, sigma_point):
        return self.Z @ sigma_point