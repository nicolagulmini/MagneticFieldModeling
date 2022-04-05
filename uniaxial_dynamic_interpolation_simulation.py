import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import numpy as np
import cube
from queue import Queue

EPSILON = 5 # mm of tolerance

# reading one point at time directly from the matrix
# also, there is some noise

grid = np.transpose(loadmat('./MagneticFieldModeling/simulated_data/fluxes_biot_5_cube5cm.mat')['PP_test_grd'])
uniaxial_mat = loadmat('./MagneticFieldModeling/simulated_data/noisy/fluxes_biot_300_cube5cm.mat')
positions = uniaxial_mat['PP_test_rnd']
measures = uniaxial_mat['fluxes_biot_rnd_SNR60']
measures = np.reshape(measures, (measures.shape[0], measures.shape[2]))
orientations = uniaxial_mat['n_test_rnd']
high_dim_x = np.transpose(np.concatenate((positions, orientations)))

plt.close('all')
fig = plt.figure()
ax = plt.axes(projection='3d')
zline = grid.T[2]
yline = grid.T[1]
xline = grid.T[0]

cube = cube.cube(origin=np.array([-2., -2., 10.5]), uniaxial=True)

COUNTER = 0
amount_of_new_points = 10
number_of_past_points_to_visualize = 10 # if >= amount_of_new_points then number_of_past_points_to_visualize = amount_of_new_points
# maybe in the dynamic scenario a queue would be a more suitable structure to record the sensor's past positions to visualize

def animate(k):
    global COUNTER
    global queue_for_position_visualization
    if COUNTER <= 124-124%amount_of_new_points:
        plt.cla()
        
        ax.set_title("\nx component (sampled points = %i)"%(COUNTER))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_xlim(10*cube.origin_corner[0]-EPSILON, 10*(cube.origin_corner[0]+cube.side_length)+EPSILON)
        ax.set_ylim(10*cube.origin_corner[1]-EPSILON, 10*(cube.origin_corner[1]+cube.side_length)+EPSILON)
        ax.set_zlim(10*cube.origin_corner[2]-EPSILON, 10*(cube.origin_corner[2]+cube.side_length)+EPSILON)
        
        new_points = high_dim_x[COUNTER:COUNTER+amount_of_new_points]
        new_measures = measures[COUNTER:COUNTER+amount_of_new_points]
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
            ax.scatter3D(xline, yline, zline, 
                         lw = 0, 
                         s = (60*i*unc_x_)**2, 
                         alpha = .05/i,
                         c = color_vec_x)
            
        # also the visualization for the uniaxial is different
        print(new_points)
        pos_sensor = np.flip(new_points, 0)[:number_of_past_points_to_visualize, :3]
        print(pos_sensor)
        or_sensor = np.flip(new_points, 0)[:number_of_past_points_to_visualize, 3:]
        ax.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
        ax.plot(pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2], alpha=.3, color='blue')
        ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        COUNTER += amount_of_new_points
        
ani = FuncAnimation(plt.gcf(), animate, interval=300)
plt.tight_layout()