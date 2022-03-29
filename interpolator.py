import numpy as np
#from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
#from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE



class cube:
    
    def __init__(self, origin, side_length=4.):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
    
    def corners(self):
        return [self.origin_corner, 
                self.origin_corner + self.side_length * np.array([0., 0., 1.]),
                self.origin_corner + self.side_length * np.array([0., 1., 0.]),
                self.origin_corner + self.side_length * np.array([0., 1., 1.]),
                self.origin_corner + self.side_length * np.array([1., 0., 0.]),
                self.origin_corner + self.side_length * np.array([1., 0., 1.]),
                self.origin_corner + self.side_length * np.array([1., 1., 0.]),
                self.origin_corner + self.side_length * np.ones(3),
                ]
    
    def is_inside(self, point):
        # a 3-dim point
        if point[0] >= self.origin_corner[0] and point[1] >= self.origin_corner[1] and point[2] >= self.origin_corner[2]:
            if point[0] <= self.origin_corner[0] + self.side_length and point[1] <= self.origin_corner[1] + self.side_length and point[2] <= self.origin_corner[2] + self.side_length:
                return True
        return False
    
    def __str__(self):
        return "Cube of size %.1f x %.1f x %.1f, origin in (%.1f, %.1f, %.1f)" % (self.side_length, 
                                                                    self.side_length, 
                                                                    self.side_length, 
                                                                    self.origin_corner[0], 
                                                                    self.origin_corner[1], 
                                                                    self.origin_corner[2]) 
    
     

class rbf_interpolator:
    
    def __init__(self, pointset, measures, gamma=.0005):
        self.pointset = pointset
        self.measures = measures
        self.gamma = gamma
        
    def update_sets(self, new_points, new_measures):
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
