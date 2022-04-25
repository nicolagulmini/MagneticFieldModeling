import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import zeros
from queue import Queue
# import pyigtl
# from PyDAQmx import *
import threading
import platform
from ctypes import byref
from sklearn.metrics.pairwise import rbf_kernel
import sys
np.set_printoptions(threshold=sys.maxsize)

class cube_to_calib:
    
    def __init__(self, origin, side_length=40., gamma=0.0005, sigma=1e-10, point_density=10., minimum_number_of_points=5):
        
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        
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
        self.contributions = np.zeros((self.number_of_points_along_an_axis**3, 3)) # here the contributions of the sampled points in terms of the amount of covered volume
                
        basis_vectors_x = np.array([np.ones(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0]))])
        basis_vectors_y = np.array([np.zeros(shape=(self.grid.shape[0])), np.ones(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0]))])
        basis_vectors_z = np.array([np.zeros(shape=(self.grid.shape[0])), np.zeros(shape=(self.grid.shape[0])), np.ones(shape=(self.grid.shape[0]))])
        
        high_dim_x = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_z)))
        
        self.stack_grid = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
        self.interpolator = None #uniaxial_to_calib(gamma, sigma, self.stack_grid)
        
    def add_point(self, point):
        print(point)
        query = np.round_((point[:3] - self.origin_corner) / self.point_density, decimals = 0)
        index = int(sum([query[i] * self.number_of_points_along_an_axis ** i for i in range(3)]))
        print(self.grid[index])
        self.contributions[index] += point[3:6] / self.minimum_number_of_points # cause points are (x, y, z, orientation_x, orientation_y, orientation_z, measurements...)
        print(self.contributions[index])
        
    # def update_uncert_vis(self, new_points, new_measures):
    #     self.interpolator.update_kernel(new_points, new_measures)
    #     return self.interpolator.uncertainty(new_points, new_measures)