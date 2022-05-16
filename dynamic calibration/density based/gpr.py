import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as gpr

class gaussian_process_regressor:
    
    def __init__(self, alpha, points, measures):
        self.alpha = alpha
        self.points = points # shape = (number of points, 6), which means 3 for position and 3 for orientation
        self.measures = measures # shape = (number of points, 8), which means one uniaxial measure per coil
        self.kernel = RBF()
        self.regressor = gpr(kernel=self.kernel, 
                                alpha=self.alpha, 
                                optimizer='fmin_l_bfgs_b', 
                                normalize_y=False, 
                                copy_X_train=False, 
                                random_state=None)
    def fit(self):
        self.regressor.fit(self.points, self.measures)
        
    def score(self, points, measures):
        return self.regressor.score(points, measures)
        
    def predict(self, points):  
        return self.regressor.predict(points, return_std=True)