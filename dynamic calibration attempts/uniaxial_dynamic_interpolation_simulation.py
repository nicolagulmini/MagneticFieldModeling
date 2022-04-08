import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import numpy as np
import cube
#from queue import Queue
from scipy import interpolate

EPSILON = 1 # mm of tolerance for the visualization of the cube
AMOUNT_OF_NEW_POINTS = 10
if AMOUNT_OF_NEW_POINTS < 4:
    AMOUNT_OF_NEW_POINTS = 4
NUMBER_OF_PAST_POINTS_TO_VIS = AMOUNT_OF_NEW_POINTS # if >= amount_of_new_points then number_of_past_points_to_visualize = amount_of_new_points
GAMMA = .0005
SIGMA = (2.5e-3)**2

# reading AMOUNT_OF_NEW_POINTS points at time directly from the matrix
# also, there is some noise
# just for the simulation
grid = np.transpose(loadmat('./fluxes_biot_5_cube5cm.mat')['PP_test_grd'])
uniaxial_mat = loadmat('./fluxes_biot_300_cube5cm.mat')
positions = uniaxial_mat['PP_test_rnd']
measures = uniaxial_mat['fluxes_biot_rnd_SNR60']
measures = np.reshape(measures, (measures.shape[0], measures.shape[2]))
orientations = uniaxial_mat['n_test_rnd']
high_dim_x = np.transpose(np.concatenate((positions, orientations)))

# for visualization
plt.close('all')
fig = plt.figure("Three components")

# three plots for the three components
ax = fig.add_subplot(1, 3, 1, projection='3d')
ay = fig.add_subplot(1, 3, 2, projection='3d')
az = fig.add_subplot(1, 3, 3, projection='3d')

zline = grid.T[2]
yline = grid.T[1]
xline = grid.T[0]

# define the cube
cube = cube.cube(origin=np.array([-2., -2., 10.5]), uniaxial=True)
cube.interpolator.set_sigma(SIGMA)
cube.interpolator.set_gamma(GAMMA)

COUNTER = 0
# maybe in the dynamic scenario a queue would be a more suitable structure to record the sensor's past positions to visualize

def animate(k):
    global COUNTER # to allow the modification of this external variable also inside this method
    if COUNTER <= 124-124%AMOUNT_OF_NEW_POINTS:
        
        # clear the axis, otherwise new plots are shown over the past ones
        #plt.cla() # useless now
        ax.clear()
        ay.clear()
        az.clear()
        
        # and everytime it is necessary to reset the settings of the plot
        ax.set_title("\nx component (sampled points = %i)"%(COUNTER))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_xlim(10*cube.origin_corner[0]-EPSILON, 10*(cube.origin_corner[0]+cube.side_length)+EPSILON)
        ax.set_ylim(10*cube.origin_corner[1]-EPSILON, 10*(cube.origin_corner[1]+cube.side_length)+EPSILON)
        ax.set_zlim(10*cube.origin_corner[2]-EPSILON, 10*(cube.origin_corner[2]+cube.side_length)+EPSILON)
        
        ay.set_title("\ny component (sampled points = %i)"%(COUNTER))
        ay.set_xlabel("x (mm)")
        ay.set_ylabel("y (mm)")
        ay.set_zlabel("z (mm)")
        ay.set_xlim(10*cube.origin_corner[0]-EPSILON, 10*(cube.origin_corner[0]+cube.side_length)+EPSILON)
        ay.set_ylim(10*cube.origin_corner[1]-EPSILON, 10*(cube.origin_corner[1]+cube.side_length)+EPSILON)
        ay.set_zlim(10*cube.origin_corner[2]-EPSILON, 10*(cube.origin_corner[2]+cube.side_length)+EPSILON)
        
        az.set_title("\nz component (sampled points = %i)"%(COUNTER))
        az.set_xlabel("x (mm)")
        az.set_ylabel("y (mm)")
        az.set_zlabel("z (mm)")
        az.set_xlim(10*cube.origin_corner[0]-EPSILON, 10*(cube.origin_corner[0]+cube.side_length)+EPSILON)
        az.set_ylim(10*cube.origin_corner[1]-EPSILON, 10*(cube.origin_corner[1]+cube.side_length)+EPSILON)
        az.set_zlim(10*cube.origin_corner[2]-EPSILON, 10*(cube.origin_corner[2]+cube.side_length)+EPSILON)
        
        new_points = high_dim_x[COUNTER:COUNTER+AMOUNT_OF_NEW_POINTS]
        new_measures = measures[COUNTER:COUNTER+AMOUNT_OF_NEW_POINTS]
        cube.add_points(new_points, new_measures)
        cube.interpolate()
        unc = cube.uncertainty_cloud(grid)
        
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
    
        for i in np.arange(.1, 1., .15):
            ax.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_x)
            ay.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_y)
            az.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_z)
            
        # also the visualization for the uniaxial is different
        pos_sensor = np.flip(new_points, 0)[:NUMBER_OF_PAST_POINTS_TO_VIS, :3]
        or_sensor = np.flip(new_points, 0)[:NUMBER_OF_PAST_POINTS_TO_VIS, 3:]
        ax.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
        ay.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
        az.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
        
        tck, u = interpolate.splprep([pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2]], s=2)
        u_fine = np.linspace(0, 1, 100)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        
        ax.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
        ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        ay.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
        ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        az.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
        az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        
        COUNTER += AMOUNT_OF_NEW_POINTS
        
ani = FuncAnimation(plt.gcf(), animate, interval=300)
plt.tight_layout()