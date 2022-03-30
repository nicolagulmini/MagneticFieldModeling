from scipy.io import loadmat
import numpy as np
import cube

from matplotlib import pyplot as plt
#from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

mat = loadmat('./MagneticFieldModeling/simulated_data/fluxes_biot_5_cube5cm.mat')

random_cloud = np.transpose(mat['PP_test_rnd'])
test_points = np.transpose(mat['PP_test_grd'])

# points (batch)
fluxes_biot_rnd = np.swapaxes(mat['fluxes_biot_rnd'], 1, 2)
fluxes_biot_rnd = fluxes_biot_rnd.reshape((fluxes_biot_rnd.shape[0], 24))

# definition of the cube
cube = cube.cube(origin=np.array([-2., -2., 10.5]))

# set the points inside the cube
cube.add_points(random_cloud, fluxes_biot_rnd)
cube.interpolate()
unc = cube.uncertainty_cloud(test_points)
print(unc)
  
fig = plt.figure()
ax = plt.axes(projection='3d')
zline = test_points.T[2]
yline = test_points.T[1]
xline = test_points.T[0]

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")

unc_ = unc[np.newaxis] / max(unc)
print(unc_)
color_vec = np.concatenate((unc_, 1-unc_, np.zeros(unc_.shape)), axis=0).T

for i in np.arange(.1, 1., .15): # np.arange(.1, 1.01, .1) for a nicer visualization
    ax.scatter3D(xline, yline, zline, 
                 lw = 0, 
                 s = (100*i*unc_)**2, 
                 alpha = .05/i,
                 c = color_vec)