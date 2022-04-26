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
        
        self.points = np.array([]) # shape = (number of points, 6), which means 3 for position and 3 for orientation
        self.measures = np.array([]) # shape = (number of points, 8), which means one measure per coil
        
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
        self.contributions[index] += points[:, 3:6] / self.minimum_number_of_points # cause points are (x, y, z, orientation_x, orientation_y, orientation_z, measurements...)
        # dont ignore points... unless it becomes slow
        
    def update_points(self, new_points, new_measures):
        if self.points.shape[0] == 0:
            self.points = new_points
            self.measures = new_measures
        else:
            self.points = np.concatenate((self.points, new_points))
            self.measures = np.concatenate((self.measures, new_measures))
    
    def interpolation(self):
        # define an interpolator which could be trained at the end of the calibration, with the gathered points
        # could be an rbf or a gaussian process or a neural network...
        # notice that sigma and gamma are already defined
        return
    

class CoilModel:

    def __init__(self, module_config={'centers_x': None, 'centers_y': None}):
        '''
            The init function should initialise and prepare all data necessary for 'coil_field' to operate.
        '''
        x_centres = np.array(module_config['centers_x'])
        x_centres = x_centres * 1.0e-3
        y_centres = np.array(module_config['centers_y'])
        y_centres = y_centres * 1.0e-3
        
        x_points_a = np.array([-61.5e-3/np.sqrt(2), 0, 61.5e-3 / np.sqrt(2),0, - 61.5e-3 / np.sqrt(2)])
        y_points_a = np.array([0, - 61.5e-3 / np.sqrt(2), 0, 61.5e-3 / np.sqrt(2), 0])
        z_points_a = np.zeros([1, 5]) -1.6e-3 / 2
        
        x_points_v = [-61.5e-3 / 2, 61.5e-3 / 2, 61.5e-3 / 2, - 61.5e-3 / 2, - 61.5e-3 / 2]
        y_points_v = [-61.5e-3 / 2, - 61.5e-3 / 2, 61.5e-3 / 2, 61.5e-3 / 2, - 61.5e-3 / 2]
        z_points_v = np.zeros([1, 5]) - 1.6e-3 / 2
        
        x_points_1 = x_points_v + x_centres[0]
        x_points_2 = x_points_a + x_centres[1]
        x_points_3 = x_points_v + x_centres[2]
        x_points_4 = x_points_a + x_centres[3]
        x_points_5 = x_points_a + x_centres[4]
        x_points_6 = x_points_v + x_centres[5]
        x_points_7 = x_points_a + x_centres[6]
        x_points_8 = x_points_v + x_centres[7]
        
        y_points_1 = y_points_v + y_centres[0]
        y_points_2 = y_points_a + y_centres[1]
        y_points_3 = y_points_v + y_centres[2]
        y_points_4 = y_points_a + y_centres[3]
        y_points_5 = y_points_a + y_centres[4]
        y_points_6 = y_points_v + y_centres[5]
        y_points_7 = y_points_a + y_centres[6]
        y_points_8 = y_points_v + y_centres[7]
        
        z_points_1 = z_points_v
        z_points_2 = z_points_a
        z_points_3 = z_points_v
        z_points_4 = z_points_a
        z_points_5 = z_points_a
        z_points_6 = z_points_v
        z_points_7 = z_points_a
        z_points_8 = z_points_v
        
        self.x_points = np.array(
            [x_points_1, x_points_2, x_points_3, x_points_4, x_points_5, x_points_6, x_points_7, x_points_8])
        self.y_points = np.array(
            [y_points_1, y_points_2, y_points_3, y_points_4, y_points_5, y_points_6, y_points_7, y_points_8])
        self.z_points = np.array(
            [z_points_1, z_points_2, z_points_3, z_points_4, z_points_5, z_points_6, z_points_7, z_points_8])
        
    def coil_field_total(self, px, py, pz):
        
        '''
            must be defined. 
            A User defined function which returns the magnetic field intensity H at a point in space P.
            This function must be defined as accepting a cartesian coordinate (x,y,z),
            and returning a three magnetic intensity values Hx, Hy and Hz
        '''

        I = 1
        numPoints = self.x_points.shape[1]

        # matrix conversions

        x_points = np.matrix(self.x_points)
        y_points = np.matrix(self.y_points)
        z_points = np.matrix(self.z_points)

        ax = x_points[:, 1:numPoints] - x_points[:, 0:(numPoints - 1)]
        ay = y_points[:, 1:numPoints] - y_points[:, 0:(numPoints - 1)]
        az = z_points[:, 1:numPoints] - z_points[:, 0:(numPoints - 1)]

        bx = x_points[:, 1:numPoints] - px
        by = y_points[:, 1:numPoints] - py
        bz = z_points[:, 1:numPoints] - pz

        cx = x_points[:, 0:(numPoints - 1)] - px
        cy = y_points[:, 0:(numPoints - 1)] - py
        cz = z_points[:, 0:(numPoints - 1)] - pz

        c_mag = np.sqrt(np.square(cx) + np.square(cy) + np.square(cz))
        b_mag = np.sqrt(np.square(bx) + np.square(by) + np.square(bz))

        a_dot_b = np.multiply(ax, bx) + np.multiply(ay, by) + np.multiply(az, bz)
        a_dot_c = np.multiply(ax, cx) + np.multiply(ay, cy) + np.multiply(az, cz)

        c_cross_a_x = np.multiply(az, cy) - np.multiply(ay, cz)
        c_cross_a_y = np.multiply(ax, cz) - np.multiply(az, cx)
        c_cross_a_z = np.multiply(ay, cx) - np.multiply(ax, cy)

        c_cross_a_mag_squared = np.square(c_cross_a_x) + np.square(c_cross_a_y) + np.square(c_cross_a_z)

        scalar = np.divide((np.divide(a_dot_c, c_mag) - np.divide(a_dot_b, b_mag)), c_cross_a_mag_squared)

        hx_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_x, scalar)
        hy_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_y, scalar)
        hz_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_z, scalar)

        hx = np.sum(hx_dum, axis=1)
        hy = np.sum(hy_dum, axis=1)
        hz = np.sum(hz_dum, axis=1)

        return hx, hy, hz

global coil_model
coil_model = CoilModel(module_config={'centers_x': [-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], 
                                      'centers_y': [93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]}) # mm

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp)    

def calib_simulation(origin=np.array([0., 0., 50.]), side_length=40., AMOUNT_OF_NEW_POINTS=5, interval=100, EPSILON=1):
            
    plt.close('all')
    fig = plt.figure("Three components")
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ay = fig.add_subplot(1, 3, 2, projection='3d')
    az = fig.add_subplot(1, 3, 3, projection='3d')
    
    global cube
    cube = cube_to_calib(origin, side_length, point_density=20., minimum_number_of_points=5)
    
    zline = cube.grid.T[2]
    yline = cube.grid.T[1]
    xline = cube.grid.T[0]
        
    q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
    
    def animate(k):
        
        while len(q.queue) < AMOUNT_OF_NEW_POINTS:
            pos, ori = cube.side_length*np.random.random(3), np.random.random(3)
            tmp = get_theoretical_field(coil_model, pos, ori)
            q.put(np.concatenate((pos, ori, tmp.A1), axis=0))

        new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
        #COUNTER += 1
        # and everytime it is necessary to reset the settings of the plot
        ax.clear()
        # ax.set_title("\nx component")
        # ax.set_xlabel("x (mm)")
        # ax.set_ylabel("y (mm)")
        # ax.set_zlabel("z (mm)")
        ax.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        ax.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        ax.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        ay.clear()
        # ay.set_title("\ny component")
        # ay.set_xlabel("x (mm)")
        # ay.set_ylabel("y (mm)")
        # ay.set_zlabel("z (mm)")
        ay.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        ay.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        ay.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        az.clear()
        # az.set_title("\nz component")
        # az.set_xlabel("x (mm)")
        # az.set_ylabel("y (mm)")
        # az.set_zlabel("z (mm)")
        az.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        az.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        az.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        cube.add_batch(new_raw_points)
          
        c_x = cube.contributions.T[0][np.newaxis] 
        c_y = cube.contributions.T[1][np.newaxis]
        c_z = cube.contributions.T[2][np.newaxis]
        
        unc_x = 1.-max(c_x, 1) # risolvi sta roba
        unc_y = 1.-max(c_y, 1)
        unc_z = 1.-max(c_z, 1)
        
        color_vec_x = np.concatenate((unc_x, 1-unc_x, np.zeros(unc_x.shape)), axis=0).T
        color_vec_y = np.concatenate((unc_y, 1-unc_y, np.zeros(unc_y.shape)), axis=0).T
        color_vec_z = np.concatenate((unc_z, 1-unc_z, np.zeros(unc_z.shape)), axis=0).T

        for i in np.arange(.1, 1., .2):
            ax.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x)**2, alpha = .05/i, c = color_vec_x)
            ay.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_y)**2, alpha = .05/i, c = color_vec_y)
            az.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_z)**2, alpha = .05/i, c = color_vec_z)
            
        pos_sensor = np.flip(new_raw_points[:, :6], 0)[:, :3]
        or_sensor = np.flip(new_raw_points[:, :6], 0)[:, 3:]

        ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            
    global ani
    ani = FuncAnimation(plt.gcf(), animate, interval=interval)

calib_simulation()