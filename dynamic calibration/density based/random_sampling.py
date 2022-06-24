from matplotlib import pyplot as plt
import numpy as np
import CoilModel as Coil
import cube_to_calib as CubeModel

n = 10000

cube_origin = np.array([-50., -50., 50.]) 
cube_side = 100.
coil_model = Coil.CoilModel(module_config={'centers_x': [-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], 
                                      'centers_y': [93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]}) # mm

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp)  

x_points = np.random.uniform(low=cube_origin[0], high=cube_origin[0]+cube_side, size=n)
y_points = np.random.uniform(low=cube_origin[1], high=cube_origin[1]+cube_side, size=n)
z_points = np.random.uniform(low=cube_origin[2], high=cube_origin[2]+cube_side, size=n)
u = np.random.uniform(size=n)
v = np.random.uniform(size=n)
theta = 2*np.pi*u
phi = np.arccos(2*v-1)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

data = np.concatenate((x_points[:, np.newaxis], y_points[:, np.newaxis], z_points[:, np.newaxis], x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1)
fields = np.array([get_theoretical_field(coil_model, point=point[:3], ori=point[3:6]).A1 for point in data])
data = np.concatenate((data, fields), axis=1)

np.savetxt("C:/Users/nicol/Desktop/data/1 random/sampled_points.csv", data)