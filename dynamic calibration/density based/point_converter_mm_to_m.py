from matplotlib import pyplot as plt
import numpy as np
import CoilModel as Coil
import cube_to_calib as CubeModel
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

coil_model = Coil.CoilModel(module_config={'centers_x': [-93.543, 0., 93.543, -68.55, 68.55, -93.543, 0., 93.543], 
                                      'centers_y': [93.543, 68.55, 93.543, 0., 0., -93.543, -68.55, -93.543]}) # mm

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp)

data = np.loadtxt("C:/Users/nicol/Desktop/data/1/sampled_points.csv")
print(data[0])
# for i in range(data.shape[0]):
#     position = data[i][:3]
#     orientation = data[i][3:6]
#     new_orientation = np.matmul(rotation_matrix, orientation)
#     new_position = np.matmul(rotation_matrix, position) + translation_vector
#     data[i][:3] = new_position
#     data[i][3:6] = new_orientation

points = data[:, :6]
points[:, :3] /= 1000
predicted_fields = np.array([get_theoretical_field(coil_model, point[:3], ori=point[3:6]).A1 for point in points])
print(predicted_fields.shape)
new_data = np.concatenate((points[:,:6], predicted_fields), axis=1)
print(new_data.shape)
print(data[0], new_data[0])

np.savetxt("C:/Users/nicol/Desktop/data/1/resampled_points.csv", new_data)