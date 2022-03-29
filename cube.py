import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class rbf_interpolator:
    
    def __init__(self, pointset=None, measures=None, gamma=.0005):
        self.pointset = np.array([])
        self.measures = np.array([])
        if pointset is not None:
            if pointset.shape[0] == measures.shape[0]:
                self.pointset = pointset
                self.measures = measures
            else:
                print("Warning: there is no correspondence between the points and the measures.")
        self.gamma = gamma
        print(self.pointset)
        
    def update_sets(self, new_points, new_measures):
        if not new_points.shape[0] == new_measures.shape[0]:
            print("Error: points and measures must correspond! Return.")
            return
        print("prova")
        print(self.pointset)
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

class cube:
    
    def __init__(self, origin, side_length=4.):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        self.interpolator = rbf_interpolator.rbf_interpolator()
    
    def set_points(self, points, measures):
        print(self.interpolator)
        self.interpolator.update_sets(points, measures)
        
    def interpolate(self):
        self.interpolator.compute_weights()
        
    def uncertainty_cloud(self, grid_points):
        return self.interpolator.predict(grid_points, uncertainty=True)[1]
    
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
