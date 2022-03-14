import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

def produce_basis_vectors_to_predict(n):
    to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
    to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
    to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
    return to_pred_x, to_pred_y, to_pred_z

def compute_kernel(v_1, v_2, gamma):
    pos_1 = v_1[:3]
    pos_2 = v_2[:3]
    or_1 = v_1[3:]
    or_2 = v_2[3:]
    return np.dot(or_1, or_2) * np.exp(-gamma*(np.linalg.norm(pos_1-pos_2)**2))

def produce_kernel(X, Y, gamma):
    matrix = np.zeros(shape=(X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            matrix[i][j] = compute_kernel(X[i], Y[j], gamma)
    return matrix

def train_crbfi(x, y, chosen_gamma):
    return np.linalg.solve(produce_kernel(x, x, chosen_gamma), y) # shape = (300, 8), one weight for each training point and for each coil

def grid_evaluation(high_dim_x, y_train, x_val, y_val, chosen_gamma=.0005):
    W = train_crbfi(high_dim_x, y_train, chosen_gamma)
    basis_vectors_x_val, basis_vectors_y_val, basis_vectors_z_val = produce_basis_vectors_to_predict(y_val.shape[0]) # each one has got shape = (3, 125)
    high_dim_x_val_x = np.transpose(np.concatenate((x_val, basis_vectors_x_val))) # shape = (125, 6)
    high_dim_x_val_y = np.transpose(np.concatenate((x_val, basis_vectors_y_val))) # shape = (125, 6)
    high_dim_x_val_z = np.transpose(np.concatenate((x_val, basis_vectors_z_val))) # shape = (125, 6)

    stack_together_val = np.concatenate((high_dim_x_val_x, high_dim_x_val_y, high_dim_x_val_z)) # shape = (375, 6)

    grid_kernel = produce_kernel(stack_together_val, high_dim_x, chosen_gamma)

    grid_predictions = np.matmul(grid_kernel, W) # shape = (375, 8)

    grid_predictions_x = grid_predictions[:125] # shape = (125, 8)
    grid_predictions_y = grid_predictions[125:250] # shape = (125, 8)
    grid_predictions_z = grid_predictions[250:] # shape = (125, 8)

    grid_predictions_x = np.reshape(grid_predictions_x, (125, 1, 8))
    grid_predictions_y = np.reshape(grid_predictions_y, (125, 1, 8))
    grid_predictions_z = np.reshape(grid_predictions_z, (125, 1, 8))

    grid_predictions = np.concatenate((grid_predictions_x, grid_predictions_y, grid_predictions_z), axis=1)

    den = np.mean(np.abs(y_val.reshape(-1)))
    nmae = MAE(grid_predictions.reshape(-1), y_val.reshape(-1)) / den * 100
    nrmse = MSE(grid_predictions.reshape(-1), y_val.reshape(-1), squared=False) / den * 100
    return nmae, nrmse

def diag_evaluation(high_dim_x, y_train, x_diag, y_diag, chosen_gamma=.0005):
    basis_vectors_x_diag, basis_vectors_y_diag, basis_vectors_z_diag = produce_basis_vectors_to_predict(y_diag.shape[0]) # each one has got shape = (3, 100)
    high_dim_x_diag_x = np.transpose(np.concatenate((x_diag, basis_vectors_x_diag))) # shape = (100, 6)
    high_dim_x_diag_y = np.transpose(np.concatenate((x_diag, basis_vectors_y_diag))) # shape = (100, 6)
    high_dim_x_diag_z = np.transpose(np.concatenate((x_diag, basis_vectors_z_diag))) # shape = (100, 6)

    stack_together_diag = np.concatenate((high_dim_x_diag_x, high_dim_x_diag_y, high_dim_x_diag_z)) # shape = (300, 6)

    diag_kernel = produce_kernel(stack_together_diag, high_dim_x, chosen_gamma)

    diag_predictions = np.matmul(diag_kernel, train_crbfi(high_dim_x, y_train, chosen_gamma)) # shape = (300, 8)

    diag_predictions_x = diag_predictions[:100] # shape = (100, 8)
    diag_predictions_y = diag_predictions[100:200] # shape = (100, 8)
    diag_predictions_z = diag_predictions[200:] # shape = (100, 8)

    diag_predictions_x = np.reshape(diag_predictions_x, (100, 1, 8))
    diag_predictions_y = np.reshape(diag_predictions_y, (100, 1, 8))
    diag_predictions_z = np.reshape(diag_predictions_z, (100, 1, 8))

    diag_predictions = np.concatenate((diag_predictions_x, diag_predictions_y, diag_predictions_z), axis=1) # shape = (100, 3, 8)

    den = np.mean(np.abs(y_diag.reshape(-1)))
    nmae = MAE(diag_predictions.reshape(-1), y_diag.reshape(-1)) / den * 100
    nrmse = MSE(diag_predictions.reshape(-1), y_diag.reshape(-1), squared=False) / den * 100
    return diag_predictions, nmae, nrmse

def visualize_diag_pred(diag_pred, stdv=None, title="Magnetic fields components estimation", namefile=None):
    diag_predictions_first_coil = np.swapaxes(diag_pred, 0, 2)[0]
    y_diag_first_coil = np.swapaxes(y_diag, 0, 2)[0]

    fig, axs = plt.subplots(1, 3, figsize=(25,5))

    plt.suptitle(title, fontsize=16)

    axs[0].set_title('H_x')
    axs[0].scatter(range(y_diag.shape[0]), y_diag_first_coil[0], color='green', marker='.')
    axs[0].set_xlabel('points on the diag')
    axs[0].set_ylabel('H_x')
    axs[0].scatter(range(y_diag.shape[0]), diag_predictions_first_coil[0], marker='.', color='red')
    if stdv is not None:
        axs[0].plot(range(y_diag.shape[0]),  diag_predictions_first_coil[0]+3*stdv[:100], linewidth=.8, color='blue')
        axs[0].plot(range(y_diag.shape[0]),  diag_predictions_first_coil[0]-3*stdv[:100], linewidth=.8, color='blue')
        axs[0].fill_between(range(y_diag.shape[0]),  diag_predictions_first_coil[0]-3*stdv[:100], diag_predictions_first_coil[0]+3*stdv[:100], color='blue', alpha=.2)
    axs[0].grid(color='grey', linewidth=.5)

    axs[1].set_title('H_y')
    axs[1].scatter(range(y_diag.shape[0]), y_diag_first_coil[1], color='green', marker='.')
    axs[1].set_xlabel('points on the diag')
    axs[1].set_ylabel('H_y')
    axs[1].scatter(range(y_diag.shape[0]), diag_predictions_first_coil[1], marker='.', color='red')
    if stdv is not None:
        axs[1].plot(range(y_diag.shape[0]), diag_predictions_first_coil[1]+3*stdv[100:200], linewidth=.8, color='blue')
        axs[1].plot(range(y_diag.shape[0]), diag_predictions_first_coil[1]-3*stdv[100:200], linewidth=.8, color='blue')
        axs[1].fill_between(range(y_diag.shape[0]), diag_predictions_first_coil[1]-3*stdv[100:200], diag_predictions_first_coil[1]+3*stdv[100:200], color='blue', alpha=.2)
    axs[1].grid(color='grey', linewidth=.5)

    axs[2].set_title('H_z')
    axs[2].scatter(range(y_diag.shape[0]), y_diag_first_coil[2], color='green', marker='.')
    axs[2].set_xlabel('points on the diag')
    axs[2].set_ylabel('H_z')
    axs[2].scatter(range(y_diag.shape[0]), diag_predictions_first_coil[2], marker='.', color='red')
    if stdv is not None:
        axs[2].plot(range(y_diag.shape[0]), diag_predictions_first_coil[2]+3*stdv[200:], linewidth=.8, color='blue')
        axs[2].plot(range(y_diag.shape[0]), diag_predictions_first_coil[2]-3*stdv[200:], linewidth=.8, color='blue')
        axs[2].fill_between(range(y_diag.shape[0]), diag_predictions_first_coil[2]-3*stdv[200:], diag_predictions_first_coil[2]+3*stdv[200:], color='blue', alpha=.2)
    axs[2].grid(color='grey', linewidth=.5)

    if namefile is not None:
        plt.savefig(namefile+".png")
    plt.show()
