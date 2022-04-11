import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from queue import Queue
from scipy import interpolate
import time
import sys
import pyigtl
from sklearn.metrics.pairwise import rbf_kernel
np.set_printoptions(threshold=sys.maxsize)

def produce_basis_vectors_for_prediction(n):
    to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
    to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
    to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
    return to_pred_x, to_pred_y, to_pred_z

class uniaxial_to_calib:
    
    def __init__(self, gamma, sigma):
        self.gamma = gamma
        self.sigma = sigma
        self.points = np.array([])
        self.k = None
        
    def update_points(self, new_points):
        if self.points.shape[0] == 0:
            self.points = new_points
        else:
            self.points = np.insert(self.points, 0, new_points, axis=0)
        
    def produce_kernel(self, X, Y):
        return rbf_kernel(X[:, :3], Y[:, :3], gamma=self.gamma) * np.tensordot(X[:, 3:], Y[:, 3:], axes=(1, 1))
    
    def set_kernel(self):
        self.k = self.sigma*np.eye(self.points.shape[0])+self.produce_kernel(self.points, self.points)
        
    def update_kernel(self, new_points):
        self.update_points(new_points)
        if self.k is None:
            self.set_kernel()
            return
        print('self.k prima di rimuovere sigma:', self.k)
        self.k = self.k - self.sigma*np.eye(self.k.shape[0])
        print('self.k dopo aver rimosso sigma:', self.k)
        K_ = self.produce_kernel(new_points, self.points)
        print('K_:', K_)
        self.k = np.concatenate((self.k, K_[:, :self.k.shape[0]]))
        print('self.k dopo chegli ho aggiunto le righe:', self.k)
        self.k = np.concatenate((self.k.T, K_))
        print('self.k dopo chegli ho aggiunto le colonne:', self.k)
        self.k = self.k + self.sigma*np.eye(self.k.shape[0])
        print('self.k dopo chegli ho aggiunto sigma:', self.k)
        self.k = self.sigma*np.eye(self.points.shape[0])+self.produce_kernel(self.points, self.points)
        print('self.k target:', self.k)
        
    def uncertainty(self, stack_grid):
        pred_kernel = self.produce_kernel(stack_grid, self.points)
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(pred_kernel))
        stdv = np.sqrt(np.diag(rbf_kernel(stack_grid, gamma=self.gamma))-np.sum(Lk**2, axis=0))
        return stdv
    
class cube_to_calib:
    
    def __init__(self, origin, side_length=40., gamma=0.0005, sigma=0., point_density=10.):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        self.interpolator = uniaxial_to_calib(gamma, sigma)
        
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
        
        basis_vectors_x, basis_vectors_y, basis_vectors_z = produce_basis_vectors_for_prediction(self.grid.shape[0])
        high_dim_x = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_z)))
        self.stack_grid = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
    
    def update_uncert_vis(self, new_points):
        self.interpolator.update_kernel(new_points)
        return self.interpolator.uncertainty(self.stack_grid)

def uniaxial_dynamic_cal(client, EPSILON=1, AMOUNT_OF_NEW_POINTS=10, NUMBER_OF_PAST_POINTS_TO_VIS=10, GAMMA=.0005, SIGMA=(2.5e-3)**2, interval=300):
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
    cube = cube_to_calib(origin=np.array([-20., -20., 105.]), sigma=SIGMA)

    zline = cube.grid.T[2]
    yline = cube.grid.T[1]
    xline = cube.grid.T[0]
        
    q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
    
    def animate(k):
        global COUNTER # to allow the modification of this external variable also inside this method
        for _ in range(AMOUNT_OF_NEW_POINTS):
            message = client.wait_for_message("SensorTipToFG", timeout=5)
            if message is not None:
                pos = message.matrix.T[3][:3]
                ori = message.matrix.T[2][:3]
                q.put(np.concatenate((pos, ori), axis=0))

        if len(q.queue) >= AMOUNT_OF_NEW_POINTS: 
            new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
            
            ax.clear()
            ay.clear()
            az.clear()
            
            # and everytime it is necessary to reset the settings of the plot
            ax.set_title("\nx component")#" (sampled points = %i)"%(COUNTER))
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("z (mm)")
            ax.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            ax.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            ax.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            ay.set_title("\ny component")#" (sampled points = %i)"%(COUNTER))
            ay.set_xlabel("x (mm)")
            ay.set_ylabel("y (mm)")
            ay.set_zlabel("z (mm)")
            ay.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
            ay.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
            ay.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
            
            az.set_title("\nz component")#" (sampled points = %i)"%(COUNTER))
            az.set_xlabel("x (mm)")
            az.set_ylabel("y (mm)")
            az.set_zlabel("z (mm)")
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
        
            for i in np.arange(.1, 1., .1):
                ax.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_x)
                ay.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_y)
                az.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_z)
                
            # probabilmente va sistemata anche questa cosa in modo da poter visualizzare più punti di quelli raccolti con l'ultimo aggiornamento, quindi forse sarà necessario definire una seconda coda, apposita
            pos_sensor = np.flip(new_points, 0)[:, :3]
            or_sensor = np.flip(new_points, 0)[:, 3:]
            ax.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
            ay.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
            az.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')

            #tck, u = interpolate.splprep([pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2]], s=2)
            #u_fine = np.linspace(0, 1, 100)
            #x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            
            #ax.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            #ay.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            #az.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')

    global ani
    ani = FuncAnimation(plt.gcf(), animate, interval=interval)
    plt.tight_layout()
    
client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)
uniaxial_dynamic_cal(client, AMOUNT_OF_NEW_POINTS=10)
