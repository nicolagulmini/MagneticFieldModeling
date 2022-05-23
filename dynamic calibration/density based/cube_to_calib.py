import numpy as np
import crbfi

class cube_to_calib:
    
    def __init__(self, origin, side_length=40., gamma=0.0005, sigma=1e-10, point_density=10., minimum_number_of_points=5):
        
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        
        self.points = np.array([]) # shape = (number of points, 6), which means 3 for position and 3 for orientation
        self.measures = np.array([]) # shape = (number of points, 8), which means one measure per coil
        
        self.gamma = gamma
        self.sigma = sigma
        
        self.side_length = side_length # in millimiters
        self.minimum_number_of_points = minimum_number_of_points # needed points to consider a grid point properly covered, along an orientation
        self.point_density = point_density
        self.number_of_points_along_an_axis = int(self.side_length/point_density)+1
        x = np.linspace(self.origin_corner[0], self.origin_corner[0]+self.side_length, self.number_of_points_along_an_axis)
        y = np.linspace(self.origin_corner[1], self.origin_corner[1]+self.side_length, self.number_of_points_along_an_axis)
        z = np.linspace(self.origin_corner[2], self.origin_corner[2]+self.side_length, self.number_of_points_along_an_axis)

        grid = np.zeros((self.number_of_points_along_an_axis**3, 3))
        c = 0
        for i in z:
            for j in y:
                for k in x:
                    grid[c] = np.array([k, j, i])
                    c += 1
        self.grid = grid
        self.xline = self.grid.T[0]
        self.yline = self.grid.T[1]
        self.zline = self.grid.T[2]
        self.contributions = np.zeros((self.number_of_points_along_an_axis**3, 3)) # here the contributions of the sampled points in terms of the amount of covered volume
        # then it is sufficient to call self.contributions to obtain the "uncertainty" values
        
        basis_vectors_x = np.array([np.ones(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0]))])
        basis_vectors_y = np.array([np.zeros(shape=(self.grid.shape[0])), np.ones(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0]))])
        basis_vectors_z = np.array([np.zeros(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0])), np.ones(shape=(self.grid.shape[0]))])
        
        high_dim_x = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_z)))
        
        self.stack_grid = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
        self.interpolator = None #uniaxial_to_calib(gamma, sigma, self.stack_grid)
        
    def add_batch(self, points):
        self.update_points(points[:, :6], points[:, 6:])
        query = np.round_((points[:, :3] - np.array([self.origin_corner for _ in range(points.shape[0])])) / self.point_density, decimals = 0)
        index = sum([query[:, i] * self.number_of_points_along_an_axis ** i for i in range(3)]).astype(int) # shape = shape.points[0] i.e. one index for each point of the batch
        self.contributions[index] += abs(points[:, 3:6]) / self.minimum_number_of_points # cause points are (x, y, z, orientation_x, orientation_y, orientation_z, measurements...)
        # should ignore points
        
    def update_points(self, new_points, new_measures):
        if self.points.shape[0] == 0:
            self.points = new_points
            self.measures = new_measures
        else:
            self.points = np.concatenate((self.points, new_points))
            self.measures = np.concatenate((self.measures, new_measures))
    
    def interpolation(self):
        self.interpolator = crbfi.custom_radial_basis_function_interpolator(gamma=self.gamma, sigma=self.sigma, points=self.points, measures=self.measures, stack_grid=self.stack_grid)
        return self.interpolator.predict(), self.interpolator.uncertainty()