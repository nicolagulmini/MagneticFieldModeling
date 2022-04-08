import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import numpy as np
import cube as cb
from queue import Queue
from scipy import interpolate
import pyigtl

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
    cube = cb.cube(origin=np.array([-20., -20., 105.]), uniaxial=True)
    cube.interpolator.sigma = SIGMA
    cube.interpolator.gamma = GAMMA
    cube.set_grid()
    
    zline = cube.grid.T[2]
    yline = cube.grid.T[1]
    xline = cube.grid.T[0]
        
    q = Queue(maxsize = 15)
    
    def animate(k):
        global COUNTER # to allow the modification of this external variable also inside this method
        for _ in range(AMOUNT_OF_NEW_POINTS):
            message = client.wait_for_message("SensorTipToFG", timeout=5)
            if message is not None:
                pos = message.matrix.T[3][:3]
                ori = message.matrix.T[2][:3]
                new_data = np.concatenate((pos, ori), axis=0)
                print(new_data)
                q.put(new_data)

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
            
            cube.update(new_points)
            unc = cube.uncertainty_cloud()
            
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
                
            # probabilmente va sistemata anche questa cosa in modo da poter visualizzare più punti di quelli raccolti con l'ultimo aggiornamento, quindi forse sarà necessario definire una seconda coda, apposita
            pos_sensor = np.flip(new_points, 0)[:NUMBER_OF_PAST_POINTS_TO_VIS, :3]
            or_sensor = np.flip(new_points, 0)[:NUMBER_OF_PAST_POINTS_TO_VIS, 3:]
            ax.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
            ay.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
            az.scatter(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], s=20, color='blue')
            
            '''
            tck, u = interpolate.splprep([pos_sensor.T[0], pos_sensor.T[1], pos_sensor.T[2]], s=2)
            u_fine = np.linspace(0, 1, 100)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            
            ax.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            ay.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            az.plot(x_fine, y_fine, z_fine, alpha=.3, color='blue')
            az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            '''

    global ani
    ani = FuncAnimation(plt.gcf(), animate, interval=interval)
    plt.tight_layout()
    
client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)
uniaxial_dynamic_cal(client)
