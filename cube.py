import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class rbf_interpolator:
    
    def __init__(self, pointset=None, measures=None, gamma=.0005, sigma=1e-10):
        self.pointset = np.array([])
        self.measures = np.array([])
        if pointset is not None:
            if pointset.shape[0] == measures.shape[0]:
                self.pointset = pointset
                self.measures = measures
            else:
                print("Warning: there is no correspondence between the points and the measures.")
        self.gamma = gamma
        self.sigma = sigma
        
    def update_sets(self, new_points, new_measures):
        if not new_points.shape[0] == new_measures.shape[0]:
            print("Error: points and measures must correspond! Return.")
            return
        if self.pointset.shape[0] == 0: # if pointset is None also measures must be None
            self.pointset = new_points
            self.measures = new_measures
        else:
            self.pointset = np.insert(self.pointset, 0, new_points, axis=0)
            self.measures = np.insert(self.measures, 0, new_measures, axis=0)
        
    def compute_weights(self):
        self.k = self.sigma*np.eye(self.pointset.shape[0])+rbf_kernel(self.pointset, gamma=self.gamma)
        self.weights = np.linalg.solve(self.k, self.measures)
    
    def predict(self, points, uncertainty=False):
        pred_kernel = rbf_kernel(points, self.pointset, self.gamma)
        if not pred_kernel.shape[1] == self.weights.shape[0]:
            self.compute_weights()
        to_return = np.matmul(pred_kernel, self.weights)
        if uncertainty is False:
            return to_return
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(pred_kernel))
        stdv = np.sqrt(np.diag(rbf_kernel(points, gamma=self.gamma))-np.sum(Lk**2, axis=0))
        return to_return, stdv
    
class uniaxial_rbf_interpolator(rbf_interpolator):
    
    def __init__(self, pointset=None, measures=None, gamma=.0005, sigma=1e-10):
        super().__init__(pointset=pointset, measures=measures, gamma=gamma, sigma=sigma)
        # but now the points of the pointset are 6-dim, and the measures are 1-dim
        
    def compute_kernel(self, high_dim_x1, high_dim_x2):
        pos_1 = high_dim_x1[:3]
        pos_2 = high_dim_x2[:3]
        or_1 = high_dim_x1[3:]
        or_2 = high_dim_x2[3:]
        return np.dot(or_1, or_2) * np.exp(-self.gamma*(np.linalg.norm(pos_1-pos_2)**2))
    
    def produce_kernel(self, X, Y):
        matrix = np.zeros(shape=(X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                matrix[i][j] = self.compute_kernel(X[i], Y[j])
        return matrix
        
    def compute_weights(self):
        self.k = self.sigma*np.eye(self.pointset.shape[0])+self.produce_kernel(self.pointset, self.pointset)
        self.weights = np.linalg.solve(self.k, self.measures)
            
    def predict(self, points, uncertainty=False):
        basis_vectors_x, basis_vectors_y, basis_vectors_z = uniaxial_rbf_interpolator.produce_basis_vectors_for_prediction(points.shape[0])
        high_dim_x = np.transpose(np.concatenate((np.transpose(points), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(points), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(points), basis_vectors_z)))
        stack_together = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
        
        pred_kernel = self.produce_kernel(stack_together, self.pointset)

        if not pred_kernel.shape[1] == self.weights.shape[0]:
            self.compute_weights()
            
        mul = np.matmul(pred_kernel, self.weights)
        dims = int(mul.shape[0]/3)
        predictions_x = np.reshape(mul[:dims], (dims, 1, 8))
        predictions_y = np.reshape(mul[dims:int(2*dims)], (dims, 1, 8))
        predictions_z = np.reshape(mul[int(2*dims):], (dims, 1, 8))
        
        to_return = np.concatenate((predictions_x, predictions_y, predictions_z), axis=1)
        if uncertainty is False:
            return to_return
        
        # uncertainty computation for all the components
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(pred_kernel))
        stdv = np.sqrt(np.diag(self.produce_kernel(stack_together, stack_together))-np.sum(Lk**2, axis=0))
        return to_return, stdv
    
    @staticmethod
    def produce_basis_vectors_for_prediction(n):
        to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
        to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
        to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
        return to_pred_x, to_pred_y, to_pred_z

class cube:
    
    def __init__(self, origin, side_length=4., uniaxial=False):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        if not uniaxial:
            self.interpolator = rbf_interpolator()
        if uniaxial:
            self.interpolator = uniaxial_rbf_interpolator()
        
    def set_points(self, points, measures):
        if points.shape[0] == measures.shape[0]:
            self.interpolator.pointset = points
            self.interpolator.measures = measures
        else:
            print("Error: points and measures must correspond! Return.")
            return
    
    def add_points(self, points, measures):
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

'''
class multicube:
    
    def __init__(self):
        self.current_position = np.zeros(3)
        self.current_cube = cube(origin=self.current_position)
        self.cubes = [self.current_cube]
        
    def move_sensor(self, new_position):
        if self.current_cube.is_inside(new_position): 
            return self.current_cube
        for cube in self.cubes:
            if cube.is_inside(new_position):
                self.current_cube = cube
                return self.current_cube
        # define a new cube
        new_cube = cube(origin=np.zeros(3)) # wrong: the new cube must have origin = opposite corner of another cube, and it must contain the new point
        self.current_cube = new_cube
        self.cubes.append(self.current_cube)
'''