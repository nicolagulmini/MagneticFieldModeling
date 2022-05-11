import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def produce_basis_vectors_for_prediction(n):
    to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
    to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
    to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
    return to_pred_x, to_pred_y, to_pred_z

class custom_radial_basis_function_interpolator:
    
    def __init__(self, gamma, sigma, points, measures, stack_grid):
        self.gamma = gamma
        self.sigma = sigma
        self.points = points # shape = (number of points, 6), which means 3 for position and 3 for orientation
        self.measures = measures # shape = (number of points, 8), which means one uniaxial measure per coil
        
        self.k = self.sigma*np.eye(self.points.shape[0])+self.produce_kernel(self.points, self.points)
        self.w = np.linalg.solve(self.k, self.measures) 
        self.stack_grid = stack_grid
        self.pred_kernel = self.produce_kernel(self.stack_grid, self.points) # kernel between training points and stack grid
        self.diag_on_grid_for_cholensky = np.diag(self.produce_kernel(self.stack_grid, self.stack_grid))
        
    def produce_kernel(self, X, Y):
        return rbf_kernel(X[:, :3], Y[:, :3], gamma=self.gamma) * np.tensordot(X[:, 3:], Y[:, 3:], axes=(1, 1))
        
    def predict(self):  
        return np.matmul(self.pred_kernel, self.w)
            
    def uncertainty(self):
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(self.pred_kernel))
        stdv = np.sqrt(self.diag_on_grid_for_cholensky-np.sum(Lk**2, axis=0))
        return stdv