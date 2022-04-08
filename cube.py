import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class rbf_interpolator:
    def __init__(self, points=None, measures=None, gamma=.0005, sigma=1e-10):
        self.pointset = points
        self.measures = measures
        if points is None:
            self.pointset = np.array([])
        if measures is None:
            self.pointset = np.array([])
        self.gamma = gamma
        self.sigma = sigma
        self.k = np.array([])
        
    def update_points(self, new_points):
        if self.pointset.shape[0] == 0:
            self.pointset = new_points
        else:
            self.pointset = np.insert(self.pointset, 0, new_points, axis=0)
        
    def update(self, new_points, new_measures=None):
        if new_measures is None:
            self.update_points(new_points)
            return
        if not new_points.shape[0] == new_measures.shape[0]:
            print("Error: points and measures must correspond! Return.")
            return
        if self.pointset.shape[0] == 0: # if pointset is None also measures must be None
            self.pointset = new_points
            self.measures = new_measures
        else:
            self.pointset = np.insert(self.pointset, 0, new_points, axis=0)
            self.measures = np.insert(self.measures, 0, new_measures, axis=0)
        
    def set_kernel(self, points=True):
        self.k = self.sigma*np.eye(self.pointset.shape[0])+rbf_kernel(self.pointset, gamma=self.gamma)
        
    def set_weights(self):
        self.weights = np.linalg.solve(self.k, self.measures)
    
    def predict(self, points, uncertainty=False):
        pred_kernel = rbf_kernel(points, self.pointset, self.gamma)
        if not pred_kernel.shape[1] == self.weights.shape[0]:
            self.compute_weights()
        to_return = np.matmul(pred_kernel, self.weights)
        if uncertainty is False:
            return to_return
        stdv = self.uncertainty(points, pred_kernel)
        return to_return, stdv
    
    def predict_with_uncertainty(self, points):
        return self.predict(points, uncertainty=True)
    
    def uncertainty(self, points, pred_kernel=None):
        if pred_kernel is None:
            pred_kernel = rbf_kernel(points, self.pointset, self.gamma)
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(pred_kernel))
        stdv = np.sqrt(np.diag(rbf_kernel(points, gamma=self.gamma))-np.sum(Lk**2, axis=0))
        return stdv
    
class uniaxial_rbf_interpolator(rbf_interpolator):
    
    def __init__(self, points=None, measures=None, gamma=.0005, sigma=1e-10):
        super().__init__(points=points, measures=measures, gamma=gamma, sigma=sigma)
        # but now the points of the pointset are 6-dim, and the measures are 1-dim
    
    def produce_kernel(self, X, Y):
        return rbf_kernel(X[:, :3], Y[:, :3], gamma=self.gamma) * np.tensordot(X[:, 3:], Y[:, 3:], axes=(1, 1))
    
    def set_kernel(self):
        self.k = self.sigma*np.eye(self.pointset.shape[0])+self.produce_kernel(self.pointset, self.pointset)
                    
    def predict(self, points, uncertainty=False):
        basis_vectors_x, basis_vectors_y, basis_vectors_z = uniaxial_rbf_interpolator.produce_basis_vectors_for_prediction(points.shape[0])
        high_dim_x = np.transpose(np.concatenate((np.transpose(points), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(points), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(points), basis_vectors_z)))
        stack_together = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
        
        pred_kernel = self.produce_kernel(stack_together, self.pointset)
        self.set_kernel()
        self.compute_weights()
            
        mul = np.matmul(pred_kernel, self.weights)
        dims = int(mul.shape[0]/3)
        predictions_x = np.reshape(mul[:dims], (dims, 1, 8))
        predictions_y = np.reshape(mul[dims:int(2*dims)], (dims, 1, 8))
        predictions_z = np.reshape(mul[int(2*dims):], (dims, 1, 8))
        
        to_return = np.concatenate((predictions_x, predictions_y, predictions_z), axis=1)
        
        # uncertainty computation for all the components giving stack_together instead of points
        stdv = self.uncertainty(stack_together, pred_kernel)
            
        return to_return, stdv
    
    def uncertainty(self, points, pred_kernel=None):
        if pred_kernel is None:
            pred_kernel = self.produce_kernel(points, self.pointset)
        self.set_kernel()
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(pred_kernel))
        stdv = np.sqrt(np.diag(rbf_kernel(points, gamma=self.gamma))-np.sum(Lk**2, axis=0))
        return stdv
       
    @staticmethod
    def produce_basis_vectors_for_prediction(n):
        to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
        to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
        to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
        return to_pred_x, to_pred_y, to_pred_z
    
class cube:
    
    def __init__(self, origin, side_length=40., uniaxial=False):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        if uniaxial:
            self.interpolator = uniaxial_rbf_interpolator()
        else:
            self.interpolator = rbf_interpolator()
            
    def set_grid(self, point_density=10.): 
        # point density means that every 1 centimeter there is a point of the grid
        x = np.linspace(self.origin_corner[0], self.origin_corner[0]+self.side_length, int(self.side_length/point_density)+1) # i think
        y = np.linspace(self.origin_corner[1], self.origin_corner[1]+self.side_length, 5)
        z = np.linspace(self.origin_corner[2], self.origin_corner[2]+self.side_length, 5)

        # is there a better method to do so?
        grid = np.zeros((int(x.shape[0]*y.shape[0]*z.shape[0]), 3))
        c = 0
        for i in z:
            for j in y:
                for k in x:
                    grid[c] = np.array([k, j, i])
                    c += 1
        self.grid = grid
        
        if isinstance(self.interpolator, uniaxial_rbf_interpolator):
            basis_vectors_x, basis_vectors_y, basis_vectors_z = uniaxial_rbf_interpolator.produce_basis_vectors_for_prediction(self.grid.shape[0])
            high_dim_x = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_x)))
            high_dim_y = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_y)))
            high_dim_z = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_z)))
            self.stack_grid = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
        
    def update(self, points, measures=None):
        self.interpolator.update(points, measures)
            
    def interpolate(self):
        self.interpolator.set_weights()
        
    def uncertainty_cloud(self):
        points = self.grid
        if isinstance(self.interpolator, uniaxial_rbf_interpolator):
            points = self.stack_grid
        to_ret = self.interpolator.uncertainty(points)
        return to_ret
        
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

