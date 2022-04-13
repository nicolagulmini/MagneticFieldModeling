import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from queue import Queue
from scipy import interpolate
import time
import sys
import pyigtl
from sklearn.metrics.pairwise import rbf_kernel
#np.set_printoptions(threshold=sys.maxsize)

X_CENTERS = [-93.543, 0., 93.543, -68.55, 68.55, -93.543, 0., 93.543]
Y_CENTERS = [93.543, 68.55, 93.543, 0., 0., -93.543, -68.55, -93.543]

class CoilModel:

    def __init__(self, module_config={'centers_x': None, 'centers_y': None}):
        '''
            The init function should initialise and prepare all data necessary for 'coil_field' to operate.
        '''
        x_centres = np.array(module_config['centers_x'])
        x_centres = x_centres * 1.0e-3
        y_centres = np.array(module_config['centers_y'])
        y_centres = y_centres * 1.0e-3
        
        num_coils = x_centres.shape[0]
        
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

class tri_uniaxial_to_calib:
    
    def __init__(self, gamma, sigma, grid_points):
        self.gamma = gamma
        self.sigma = sigma
        self.points = np.array([])
        self.measures = np.array([])
        self.sensor_1_kernel = None
        self.sensor_2_kernel = None
        self.sensor_3_kernel = None
        self.grid_points = grid_points
        self.pred_kernel_on_grid = None
        self.diag_on_grid_for_cholensky = None
        self.w = None # w are the weights and their dim is (3m, 8) where m is the number of grid points and 8 = coils
        
    def update_points(self, new_points, new_measures): 
        # shape of new_points must be (number_of_new_points, 12)
        # where each point has (x, y, z, n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z)
        if self.points.shape[0] == 0:
            self.points = new_points
            self.measures = new_measures
        else:
            self.points = np.concatenate((self.points, new_points))
            self.measures = np.concatenate((self.measures, new_measures))
        
    def kernels_rows(self, x):
        row = rbf_kernel(x[:, :3], self.grid_points, self.gamma)
        rows_for_kernel_1, rows_for_kernel_2, rows_for_kernel_3 = [np.zeros((row.shape[0], self.grid_points.shape[0]*3)) for _ in range(3)]
        for i in range(row.shape[0]):
            tmp = row[i]
            rows_for_kernel_1[i] = np.concatenate((x[i][3]*tmp, x[i][4]*tmp, x[i][5]*tmp))
            rows_for_kernel_2[i] = np.concatenate((x[i][6]*tmp, x[i][7]*tmp, x[i][8]*tmp))
            rows_for_kernel_3[i] = np.concatenate((x[i][9]*tmp, x[i][10]*tmp, x[i][11]*tmp))
        return rows_for_kernel_1, rows_for_kernel_2, rows_for_kernel_3
        
    def update_state(self, new_points, new_measures): # no prediction
        self.update_points(new_points, new_measures)
        if self.sensor_1_kernel is None: # if this is None then also the others has to be None
            self.sensor_1_kernel, self.sensor_2_kernel, self.sensor_3_kernel = self.kernels_rows(new_points)
            return
        self.k = self.k - self.sigma*np.eye(self.k.shape[0])
        K_ = self.produce_kernel(new_points, self.points)
        self.k = np.concatenate((self.k, K_[:, :self.k.shape[0]]))
        self.k = np.concatenate((self.k, K_.T), axis=1)
        self.k = self.k + self.sigma*np.eye(self.k.shape[0])
        
    def uncertainty(self, stack_grid, new_points):
        pred_kernel_on_new_points = self.produce_kernel(stack_grid, new_points)
        if self.pred_kernel is None:
            self.pred_kernel = pred_kernel_on_new_points
        else:
            self.pred_kernel = np.concatenate((self.pred_kernel, pred_kernel_on_new_points), axis=1)
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(self.pred_kernel))
        if self.diag_on_grid_for_cholensky is None:
            self.diag_on_grid_for_cholensky = np.diag(rbf_kernel(stack_grid, gamma=self.gamma))
        stdv = np.sqrt(self.diag_on_grid_for_cholensky-np.sum(Lk**2, axis=0))
        return stdv
    
class cube_to_calib:
    
    def __init__(self, origin, side_length=40., gamma=0.0005, sigma=0., point_density=10.):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        self.interpolator = tri_uniaxial_to_calib(gamma, sigma)
        
        x = np.linspace(self.origin_corner[0], self.origin_corner[0]+self.side_length, int(self.side_length/point_density)+1) # i think
        y = np.linspace(self.origin_corner[1], self.origin_corner[1]+self.side_length, int(self.side_length/point_density)+1)
        z = np.linspace(self.origin_corner[2], self.origin_corner[2]+self.side_length, int(self.side_length/point_density)+1)

        # is there a better method to do so?
        grid = np.zeros((int(x.shape[0]*y.shape[0]*z.shape[0]), 3))
        c = 0
        for i in z:
            for j in y:
                for k in x:
                    grid[c] = np.array([k, j, i])
                    c += 1
        self.grid = grid
        
        high_dim_x = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_z)))
        self.stack_grid = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
    
    def update_uncert_vis(self, new_points):
        self.interpolator.update_kernel(new_points)
        return self.interpolator.uncertainty(self.stack_grid, new_points)

def uniaxial_dynamic_cal(client, EPSILON=1, AMOUNT_OF_NEW_POINTS=10, NUMBER_OF_PAST_POINTS_TO_VIS=10, origin=np.array([-20., -20., 105.]), side_length=40., GAMMA=.0005, SIGMA=(2.5e-3)**2, interval=300):
    # mm of tolerance for the visualization of the cube
    if AMOUNT_OF_NEW_POINTS < 4:
        AMOUNT_OF_NEW_POINTS = 4 # for the interpolation of the curve
    # if >= amount_of_new_points then number_of_past_points_to_visualize = amount_of_new_points
            
    # for visualization
    plt.close('all')
    fig = plt.figure("Three components")
    
    # three plots for the three components
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ay = fig.add_subplot(1, 3, 2, projection='3d')
    az = fig.add_subplot(1, 3, 3, projection='3d')
    
    # define the cube
    #cube = cube_to_calib(origin=np.array([-20., -20., 105.]), sigma=SIGMA)
    cube = cube_to_calib(origin=origin, side_length=side_length, sigma=SIGMA)

    zline = cube.grid.T[2]
    yline = cube.grid.T[1]
    xline = cube.grid.T[0]
        
    q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
    
    coil_model = CoilModel(module_config={'centers_x': X_CENTERS, 
                                          'centers_y': Y_CENTERS})
    
    def animate(k):
        global COUNTER # to allow the modification of this external variable also inside this method
        for _ in range(AMOUNT_OF_NEW_POINTS):
            message = client.wait_for_message("SensorTipToFG", timeout=5)
            if message is not None:
                pos = message.matrix.T[3][:3]
                n_matrix = message.matrix.T[:3, :3]
                # define the transformations
                # produce the magnetic field measurements from the code which simulates them
                q.put(np.concatenate((pos, ori), axis=0))

        if len(q.queue) >= AMOUNT_OF_NEW_POINTS: 
            new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
            
            ax.clear()
            ay.clear()
            az.clear()
            
            # and everytime it is necessary to reset the settings of the plot
            #ax.set_title("\nx component")#" (sampled points = %i)"%(COUNTER))
            #ax.set_xlabel("x (mm)")
            #ax.set_ylabel("y (mm)")
            #ax.set_zlabel("z (mm)")
            ax.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            ax.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            ax.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            #ay.set_title("\ny component")#" (sampled points = %i)"%(COUNTER))
            #ay.set_xlabel("x (mm)")
            #ay.set_ylabel("y (mm)")
            #ay.set_zlabel("z (mm)")
            ay.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            ay.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            ay.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            #az.set_title("\nz component")#" (sampled points = %i)"%(COUNTER))
            #az.set_xlabel("x (mm)")
            #az.set_ylabel("y (mm)")
            #az.set_zlabel("z (mm)")
            az.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            az.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            az.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            new_points = new_raw_points[:, :6] # shape = (AMOUNT_OF_NEW_POINTS, 6)
            unc = cube.update_uncert_vis(new_points)
            
            dims = int(unc.shape[0]/3)
            unc_x = unc[:dims]
            unc_y = unc[dims:int(2*dims)]
            unc_z = unc[int(2*dims):]
            unc_x_ = unc_x[np.newaxis] / max(unc_x)
            unc_y_ = unc_y[np.newaxis] / max(unc_y)
            unc_z_ = unc_z[np.newaxis] / max(unc_z)
            color_vec_x = np.concatenate((unc_x_, 1-unc_x_, np.zeros(unc_x_.shape)), axis=0).T
            color_vec_y = np.concatenate((unc_y_, 1-unc_y_, np.zeros(unc_y_.shape)), axis=0).T
            color_vec_z = np.concatenate((unc_z_, 1-unc_x_, np.zeros(unc_z_.shape)), axis=0).T
        
            for i in np.arange(.1, 1., .2):
                ax.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_x)
                ay.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_y)
                az.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_z)
                
            # probabilmente va sistemata anche questa cosa in modo da poter visualizzare più punti di quelli raccolti con l'ultimo aggiornamento, quindi forse sarà necessario definire una seconda coda, apposita
            pos_sensor = np.flip(new_points, 0)[:, :3]
            or_sensor = np.flip(new_points, 0)[:, 3:]
            #ax.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
            #ay.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
            #az.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')

            #tck, u = interpolate.splprep([pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2]], s=2)
            #u_fine = np.linspace(0, 1, 100)
            #x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            
            #ax.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            #ax.plot(pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2], alpha=.3, color='blue')
            ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            #ay.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            #ay.plot(pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2], alpha=.3, color='blue')
            ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            #az.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            #az.plot(pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2], alpha=.3, color='blue')
            az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')

    global ani
    ani = FuncAnimation(plt.gcf(), animate, interval=interval)
    plt.tight_layout()
    
client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)
uniaxial_dynamic_cal(client, AMOUNT_OF_NEW_POINTS=15, origin=np.array([0., 0., 0.]), side_length=40., SIGMA=1e-10)
