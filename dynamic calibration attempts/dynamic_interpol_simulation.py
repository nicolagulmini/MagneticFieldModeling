import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import numpy as np
import cube
from queue import Queue

namefile = 'random_cloud.csv'

grid = np.transpose(loadmat('./MagneticFieldModeling/simulated_data/fluxes_biot_5_cube5cm.mat')['PP_test_grd'])

fig = plt.figure()
ax = plt.axes(projection='3d')
zline = grid.T[2]
yline = grid.T[1]
xline = grid.T[0]
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")

cube = cube.cube(origin=np.array([-2., -2., 10.5]))

ax.set_xlim(cube.origin_corner[0], cube.origin_corner[0]+cube.side_length)
ax.set_ylim(cube.origin_corner[1], cube.origin_corner[1]+cube.side_length)
ax.set_zlim(cube.origin_corner[2], cube.origin_corner[2]+cube.side_length)

'''
def animate(k):
    data = pd.read_csv(namefile)
    plt.cla()
    if len(data) != 0:
        positions = data[data.columns[:3]].to_numpy()
        measures = data[data.columns[3:]].to_numpy()
        cube.set_points(positions, measures)
        cube.interpolate()
        unc = cube.uncertainty_cloud(grid)
    
        unc_ = unc[np.newaxis] / max(unc)
        color_vec = np.concatenate((unc_, 1-unc_, np.zeros(unc_.shape)), axis=0).T
    
        for i in np.arange(.1, 1., .15): # np.arange(.1, 1.01, .1) for a nicer visualization
            ax.scatter3D(xline, yline, zline, 
                         lw = 0, 
                         s = (100*i*unc_)**2, 
                         alpha = .05/i,
                         c = color_vec)

ani = FuncAnimation(plt.gcf(), animate, interval=100)
plt.tight_layout()
'''

# to obtain a faster update
COUNTER = 0
data = pd.read_csv(namefile)
snake_positions = Queue(maxsize = 3)
for _ in range(snake_positions.maxsize):
    snake_positions.put(np.zeros(3))

def animate(k):
    global COUNTER
    global snake_positions
    if COUNTER <= 124:
        plt.cla()
        positions = np.array([data[data.columns[:3]].to_numpy()[COUNTER]])
        measures = np.array([data[data.columns[3:]].to_numpy()[COUNTER]])
        COUNTER += 1
        cube.add_points(positions, measures)
        cube.interpolate()
        unc = cube.uncertainty_cloud(grid)
    
        unc_ = unc[np.newaxis] / max(unc)
        color_vec = np.concatenate((unc_, 1-unc_, np.zeros(unc_.shape)), axis=0).T
    
        for i in np.arange(.1, 1., .15):
            ax.scatter3D(xline, yline, zline, 
                         lw = 0, 
                         s = (60*i*unc_)**2, 
                         alpha = .05/i,
                         c = color_vec)
        if snake_positions.full():
            snake_positions.get()
        snake_positions.put(positions.reshape(-1))
        ax.scatter(positions[0][0], positions[0][1], positions[0][2], s=20, color='blue')
        ax.plot([el[0] for el in snake_positions.queue], [el[1] for el in snake_positions.queue], [el[2] for el in snake_positions.queue], alpha=.3, color='blue')

ani = FuncAnimation(plt.gcf(), animate, interval=300)
plt.tight_layout()
