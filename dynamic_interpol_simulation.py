import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
import numpy as np
import cube

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

cube = cube.cube(origin=np.array([-2., -2., 10.5])) # qua lo ridefinisco ogni volta


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
#ani.save('es') # understand how to save the animation