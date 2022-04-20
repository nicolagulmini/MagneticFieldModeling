import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from queue import Queue
#import pyigtl
from sklearn.metrics.pairwise import rbf_kernel
import sys
from time import sleep
np.set_printoptions(threshold=sys.maxsize)

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
coil_model = CoilModel(module_config={'centers_x': [-93.543, 0., 93.543, -68.55, 68.55, -93.543, 0., 93.543], 
                                      'centers_y': [93.543, 68.55, 93.543, 0., 0., -93.543, -68.55, -93.543]})

def theoretical_field(model, point):
    return np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T # (3, 8)
    
def produce_basis_vectors_for_prediction(n):
    to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
    to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
    to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
    return to_pred_x, to_pred_y, to_pred_z

class uniaxial_to_calib:
    
    def __init__(self, gamma, sigma, stack_grid):
        self.gamma = gamma
        self.sigma = sigma
        self.points = np.array([]) # shape = (number of points, 6), which means 3 for position and 3 for orientation
        self.measures = np.array([]) # shape = (number of points, 8), which means one uniaxial measure per coil
        
        self.k = None
        self.w = None 
        self.pred_kernel = None # kernel between training points and stack grid
        
        self.stack_grid = stack_grid
        self.diag_on_grid_for_cholensky = np.diag(self.produce_kernel(self.stack_grid, self.stack_grid))
        
    def update_points(self, new_points, new_measures):
        if self.points.shape[0] == 0:
            self.points = new_points
            self.measures = new_measures
        else:
            self.points = np.concatenate((self.points, new_points))
            self.measures = np.concatenate((self.measures, new_measures))
        
    def produce_kernel(self, X, Y):
        return rbf_kernel(X[:, :3], Y[:, :3], gamma=self.gamma) * np.tensordot(X[:, 3:], Y[:, 3:], axes=(1, 1))
    
    def set_kernel(self):
        self.k = self.sigma*np.eye(self.points.shape[0])+self.produce_kernel(self.points, self.points)
        
    def update_kernel(self, new_points, new_measures):
        self.update_points(new_points, new_measures)
        if self.k is None:
            self.set_kernel()
            return
        self.k = self.k - self.sigma*np.eye(self.k.shape[0])
        K_ = self.produce_kernel(new_points, self.points)
        self.k = np.concatenate((self.k, K_[:, :self.k.shape[0]]))
        self.k = np.concatenate((self.k, K_.T), axis=1)
        self.k = self.k + self.sigma*np.eye(self.k.shape[0])
        
    def set_weights(self):
        if self.k.shape[0] != self.measures.shape[0]:
            print("Error. Return")
            return
        self.w = np.linalg.solve(self.k, self.measures) # it has to be (number_of_grid_points, 8)
        
    def predict(self):  
        print('weights:', self.w.shape)
        return np.matmul(self.pred_kernel, self.w)
            
    def update_pred_kernel(self, new_points, new_measures):
        #self.update_points(new_points, new_measures) # otherwise the points are set twice
        pred_kernel_on_new_points = self.produce_kernel(self.stack_grid, new_points)
        if self.pred_kernel is None:
            self.pred_kernel = pred_kernel_on_new_points
        else:
            self.pred_kernel = np.concatenate((self.pred_kernel, pred_kernel_on_new_points), axis=1)
        
    def uncertainty(self, new_points, new_measures):
        self.update_pred_kernel(new_points, new_measures)
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(self.pred_kernel))
        stdv = np.sqrt(self.diag_on_grid_for_cholensky-np.sum(Lk**2, axis=0))
        return stdv
    
class cube_to_calib:
    
    def __init__(self, origin, side_length=40., gamma=0.0005, sigma=0., point_density=10.):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        
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
        
        basis_vectors_x, basis_vectors_y, basis_vectors_z = produce_basis_vectors_for_prediction(self.grid.shape[0])
        high_dim_x = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_z)))
        self.stack_grid = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
        self.interpolator = uniaxial_to_calib(gamma, sigma, self.stack_grid)
    
    def update_uncert_vis(self, new_points, new_measures):
        self.interpolator.update_kernel(new_points, new_measures)
        return self.interpolator.uncertainty(new_points, new_measures)

def uniaxial_dynamic_cal(client, origin, side_length, GAMMA=.0005, SIGMA=(2.5e-3)**2, interval=300, EPSILON=1, AMOUNT_OF_NEW_POINTS=15):
    # EPSILON is mm of tolerance for the visualization of the cube
            
    plt.close('all')
    fig = plt.figure("Three components")
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ay = fig.add_subplot(1, 3, 2, projection='3d')
    az = fig.add_subplot(1, 3, 3, projection='3d')
    
    global cube
    cube = cube_to_calib(origin=origin, side_length=side_length, sigma=SIGMA)
    
    zline = cube.grid.T[2]
    yline = cube.grid.T[1]
    xline = cube.grid.T[0]
        
    q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
    
    def animate(k):
        
        for _ in range(AMOUNT_OF_NEW_POINTS):
            #message = client.wait_for_message("SensorTipToFG", timeout=5)
            #if message is not None:
            if True:
                #pos = message.matrix.T[3][:3]
                #ori = message.matrix.T[2][:3]
                pos, ori = cube.side_length*np.random.random(3), np.array([0., 0., 1.])
                tmp = np.dot(ori, theoretical_field(coil_model, pos))
                q.put(np.concatenate((pos, ori, tmp.A1), axis=0))
            #sleep(.1)

        if len(q.queue) >= AMOUNT_OF_NEW_POINTS: 

            new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
            
            # and everytime it is necessary to reset the settings of the plot
            ax.clear()
            ax.set_title("\nx component")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("z (mm)")
            ax.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            ax.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            ax.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            ay.clear()
            ay.set_title("\ny component")
            ay.set_xlabel("x (mm)")
            ay.set_ylabel("y (mm)")
            ay.set_zlabel("z (mm)")
            ay.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            ay.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            ay.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            az.clear()
            az.set_title("\nz component")
            az.set_xlabel("x (mm)")
            az.set_ylabel("y (mm)")
            az.set_zlabel("z (mm)")
            az.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            az.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            az.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            new_points = new_raw_points[:, :6] # shape = (AMOUNT_OF_NEW_POINTS, 6)
            new_measures = new_raw_points[:, 6:] # shape = (AMOUNT_OF_NEW_POINTS, 8)
            unc = cube.update_uncert_vis(new_points, new_measures)
            
            dims = int(unc.shape[0]/3)
            unc_x = unc[:dims]
            unc_y = unc[dims:int(2*dims)]
            unc_z = unc[int(2*dims):]
            unc_x_ = unc_x[np.newaxis] / max(unc_x)
            unc_y_ = unc_y[np.newaxis] / max(unc_y)
            unc_z_ = unc_z[np.newaxis] / max(unc_z)
            color_vec_x = np.concatenate((unc_x_, 1-unc_x_, np.zeros(unc_x_.shape)), axis=0).T
            color_vec_y = np.concatenate((unc_y_, 1-unc_y_, np.zeros(unc_y_.shape)), axis=0).T
            color_vec_z = np.concatenate((unc_z_, 1-unc_z_, np.zeros(unc_z_.shape)), axis=0).T
        
            for i in np.arange(.1, 1., .2):
                ax.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_x)
                ay.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_y_)**2, alpha = .05/i, c = color_vec_y)
                az.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_z_)**2, alpha = .05/i, c = color_vec_z)
                
            pos_sensor = np.flip(new_points, 0)[:, :3]
            or_sensor = np.flip(new_points, 0)[:, 3:]

            ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            
            #if cube.interpolator.w is not None:
            #    if cube.interpolator.w.shape[0] % AMOUNT_OF_NEW_POINTS == 0:
            cube.interpolator.set_weights()
            grid_field = cube.interpolator.predict()

            tmp = theoretical_field(coil_model, cube.grid[0])
            print(tmp[0].A1[0], tmp[1].A1[0], tmp[2].A1[0])
            print(grid_field[0][0], grid_field[0+125][0], grid_field[0+125*2][0])
            
    global ani
    ani = FuncAnimation(plt.gcf(), animate, interval=interval)
    plt.tight_layout()
    
#client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)
client = None
uniaxial_dynamic_cal(client, AMOUNT_OF_NEW_POINTS=40, origin=np.array([0., 0., 0.]), side_length=40., SIGMA=1e-10)