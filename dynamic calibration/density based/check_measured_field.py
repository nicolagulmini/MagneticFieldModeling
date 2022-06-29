from matplotlib import pyplot as plt
import numpy as np
import CoilModel as Coil
import cube_to_calib as CubeModel
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE


coil_model = Coil.CoilModel(module_config={'centers_x': [-93.543, 0., 93.543, -68.55, 68.55, -93.543, 0., 93.543], 
                                      'centers_y': [93.543, 68.55, 93.543, 0., 0., -93.543, -68.55, -93.543]}) # mm

# matrix_for_real_data = np.array([[0.999910386486641, 0.0132637271291318, 0.00177556758556555, 4.60082741045687],
# [-0.0132730439610558, 0.999897566016150, 0.00534218300150869	, 0.386762332074675],
# [-0.00170455528053870, -0.00536526920313416, 0.999984151502086, 28.3150462403445],
# [0,	 0, 0, 1]])

# rotation_matrix = matrix_for_real_data[:3, :3]
# translation_vector = matrix_for_real_data.T[3][:3]

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp)  

data = np.loadtxt("C:/Users/nicol/Desktop/data/1/resampled_points.csv")

# for i in range(data.shape[0]):
#     position = data[i][:3]
#     orientation = data[i][3:6]
#     new_orientation = np.matmul(rotation_matrix, orientation)
#     new_position = np.matmul(rotation_matrix, position) + translation_vector
#     data[i][:3] = new_position
#     data[i][3:6] = new_orientation

points = data[:, :6]
# points[:, :3] /= 1000
measured_magnetic_field = data[:, 6:14]
# print(points.shape, measured_magnetic_field.shape)
predicted_fields = np.array([get_theoretical_field(coil_model, point[:3], ori=point[3:6]).A1 for point in points])
# print(predicted_fields.shape)
den_to_norm = np.max(abs(measured_magnetic_field)) 
# print(MAE(measured_magnetic_field, predicted_fields)/den_to_norm)
# print(MSE(measured_magnetic_field, predicted_fields, squared=False)/den_to_norm)

k = 1.

n_coil = 3

# k = sum(measured_magnetic_field[:,n_coil]*predicted_fields[:,n_coil]) / sum(measured_magnetic_field[:, n_coil]**2) 
# k += 0.01e-8
# print(sum((k*measured_magnetic_field[:,0]-predicted_fields[:,0])**2))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.title("test points")
colors = [((k*measured_magnetic_field[i]-predicted_fields[i])/predicted_fields[i])[n_coil] for i in range(points.shape[0])]
# print(colors)
p = ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', cmap='coolwarm') # change the cmap
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
fig.colorbar(p, ax=ax)
plt.show()

fig = plt.figure()
plt.plot(range(points.shape[0]), k*measured_magnetic_field[:,n_coil])
plt.plot(range(points.shape[0]), predicted_fields[:,n_coil])
plt.show()