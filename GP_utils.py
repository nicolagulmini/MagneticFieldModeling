import numpy as np
import matplotlib.pyplot as plt

class GaussianProcessRegressionUtils():
    
    def __init__(n, m, D, Omega):
        self.n = n 
        self.m = m 
        self.X = np.array([el[0] for el in D])
        self.Y = np.array([el[1] for el in D])
        self.Omega = Omega
    
    def matrix_n(m):
        return np.ones(shape=(m,3))