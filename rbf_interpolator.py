import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class rbf_interpolator:
    
    def __init__(self, gamma=.0005):
        self.pointset = None
        self.measures = None
    
    def __init__(self, pointset, measures, gamma=.0005):
        self.pointset = None
        self.measures = None
        if pointset.shape[0] == measures.shape[0]:
            self.pointset = pointset
            self.measures = measures
        else:
            print("Warning: there is no correspondence between the points and the measures.")
        self.gamma = gamma
        
    def update_sets(self, new_points, new_measures):
        if not new_points.shape[0] == new_measures.shape[0]:
            print("Error: points and measures must correspond! Return.")
            return
        if self.pointset is None: # if pointset is None also measures must be None
            self.pointset = new_points
            self.measures = new_measures
        else:
            self.pointset = np.insert(self.pointset, 0, new_points, axis=0)
            self.measures = np.insert(self.measures, 0, new_measures, axis=0)
        
    def compute_weights(self):
        self.k = rbf_kernel(self.pointset, gamma=self.gamma)
        self.weights = np.linalg.solve(self.k, self.measures)
    
    def predict(self, points, uncertainty=False):
        pred_kernel = rbf_kernel(points, self.pointset, self.gamma)
        to_return = np.matmul(pred_kernel, self.weights)
        if uncertainty is False:
            return to_return
        L = np.linalg.cholesky(self.k + 1e-10*np.eye(self.k.shape[0]))
        Lk = np.linalg.solve(L, np.transpose(pred_kernel))
        stdv = np.sqrt(np.diag(rbf_kernel(points, gamma=self.gamma))-np.sum(Lk**2, axis=0))
        return to_return, stdv
    
