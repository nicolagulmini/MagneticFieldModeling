import os
import numpy as np
import crbfi
import gpr
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
import matplotlib.pyplot as plt
import CoilModel as Coil

import tensorflow as tf
from tensorflow import keras

folder = "1"
filename = "resampled_points.csv"
path = "C:/Users/nicol/Desktop/data/" + folder + "/"

matrix_for_real_data = np.array([[0.999910386486641, 0.0132637271291318, 0.00177556758556555, 4.60082741045687],
                                 [-0.0132730439610558, 0.999897566016150, 0.00534218300150869	, 0.386762332074675],
                                 [-0.00170455528053870, -0.00536526920313416, 0.999984151502086, 28.3150462403445],
                                 [0, 0, 0, 1]])

rotation_matrix = matrix_for_real_data[:3, :3]
translation_vector = matrix_for_real_data.T[3][:3]

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp) 
    
def get_metrics(pred, unc, x_train, y_train, x_val, y_val, n_diag_points, den_to_normalize_val, den_to_normalize_train):
    pred_train, pred_val, pred_x, pred_y, pred_z = pred[:x_train.shape[0]], pred[x_train.shape[0]:x_train.shape[0]+x_val.shape[0]], pred[x_train.shape[0]+x_val.shape[0]:x_train.shape[0]+x_val.shape[0]+n_diag_points], pred[x_train.shape[0]+x_val.shape[0]+n_diag_points:x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points)], pred[x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points):]
    unc_train, unc_val, unc_x, unc_y, unc_z = unc[:x_train.shape[0]], unc[x_train.shape[0]:x_train.shape[0]+x_val.shape[0]], unc[x_train.shape[0]+x_val.shape[0]:x_train.shape[0]+x_val.shape[0]+n_diag_points], unc[x_train.shape[0]+x_val.shape[0]+n_diag_points:x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points)], unc[x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points):]
    
    mae_per_point_train = [MAE(pred_train[i], y_train[i]) / den_to_normalize_train for i in range(y_train.shape[0])]
    mae_train = sum(mae_per_point_train) / len(mae_per_point_train) * 100
    
    mae_per_point = [MAE(pred_val[i], y_val[i]) / den_to_normalize_val for i in range(y_val.shape[0])]
    mae = sum(mae_per_point) / len(mae_per_point) * 100
    
    rmse_per_point = [MSE(pred_val[i], y_val[i], squared=False) / den_to_normalize_val for i in range(y_val.shape[0])]
    rmse = sum(rmse_per_point) / len(rmse_per_point) * 100
    
    return pred_train, pred_val, pred_x, pred_y, pred_z, unc_train, unc_val, unc_x, unc_y, unc_z, mae, mae_per_point, rmse, rmse_per_point, mae_train, mae_per_point_train
    

def main(origin=np.array([-50., -50., 50.]), side_length=100., n_diag_points=25, centers_x=[-93.543, 0., 93.543, -68.55, 68.55, -93.543, 0., 93.543], centers_y=[93.543, 68.55, 93.543, 0., 0., -93.543, -68.55, -93.543], smaller_cube=False, real=False):
    # put the origin of the cube, the side length and the number of points along the diagonal manually
    
    if not os.path.exists(path + filename):
        print('There are no data.')
        return
        
    # the coil model is necessary to get the comparison for the simulated magnetic data
    # it makes sense almost only if we sample with a simulated magnetic field. Otherwise just ignore the related section.
    coil_model = Coil.CoilModel(module_config={'centers_x': centers_x, 'centers_y': centers_y})

    x_diag = np.linspace(origin[0], origin[0]+side_length, n_diag_points)[:, np.newaxis] / 1000
    y_diag = np.linspace(origin[1], origin[1]+side_length, n_diag_points)[:, np.newaxis] / 1000
    z_diag = np.linspace(origin[2], origin[2]+side_length, n_diag_points)[:, np.newaxis] / 1000
    diag = np.concatenate((x_diag, y_diag, z_diag), axis=1)
    
    diag_for_x = np.concatenate((diag, np.array([[1., 0., 0.] for _ in range(n_diag_points)])), axis=1)
    diag_for_y = np.concatenate((diag, np.array([[0., 1., 0.] for _ in range(n_diag_points)])), axis=1)
    diag_for_z = np.concatenate((diag, np.array([[0., 0., 1.] for _ in range(n_diag_points)])), axis=1)

    simulated_x = np.array([get_theoretical_field(coil_model, diag_for_x[i][:3], diag_for_x[i][3:]).A1 for i in range(n_diag_points)])
    simulated_y = np.array([get_theoretical_field(coil_model, diag_for_y[i][:3], diag_for_y[i][3:]).A1 for i in range(n_diag_points)])
    simulated_z = np.array([get_theoretical_field(coil_model, diag_for_z[i][:3], diag_for_z[i][3:]).A1 for i in range(n_diag_points)])
        
    dataset = np.loadtxt(path + filename) # shape should be (n, 14)
    # n is the number of points
    # each point is 14-dimensional: 3 positions, 3 orientations, 8 coils 
    
    if real:
        for i in range(dataset.shape[0]):
            position = dataset[i][:3]
            orientation = dataset[i][3:6]
            new_orientation = np.matmul(rotation_matrix, orientation)
            new_position = np.matmul(rotation_matrix, position) + translation_vector
            dataset[i][:3] = new_position
            dataset[i][3:6] = new_orientation
            
    # convert the data from mm to m
    for i in range(dataset.shape[0]):
        dataset[i][:3] = dataset[i][:3]
    
    # if smaller_cube:
    #     effective_dimensions = origin/np.sqrt(2.) # it is the smaller cube of length L/sqrt(2), where L is the bigger cubes length
    #     effective_dataset = []
    #     for point in dataset:
    #         if point[0] > origin[0]+effective_dimensions[0]/2 and point[0] < origin[0]+side_length-effective_dimensions[0]/2:
    #             if point[1] > origin[1]+effective_dimensions[1]/2 and point[1] < origin[1]+side_length-effective_dimensions[1]/2:
    #                 if point[2] > origin[2]+effective_dimensions[2]/2 and point[2] < origin[2]+side_length-effective_dimensions[2]/2:
    #                     effective_dataset.append(point)
    #     effective_dataset = np.array(effective_dataset)
    #     print("%i discarded points" % (dataset.shape[0]-effective_dataset.shape[0]))
    #     print("now the dataset contains %i points" % effective_dataset.shape[0])
    #     dataset = effective_dataset
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title("gathered points")
    ax.scatter3D(dataset[:, 0], dataset[:, 1], dataset[:, 2], alpha=.1, marker='.')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.savefig(path + 'sampled points.png')
    
    # plot the spherical distribution of the orientations
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title("gathered points' orientations' distribution")
    ax.scatter3D(dataset[:, 3], dataset[:, 4], dataset[:, 5], alpha=.1, marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(path + 'orientations distribution.png')

    np.random.shuffle(dataset)
    training = dataset[:int(.8*dataset.shape[0])]
    x_train, y_train = training[:, :6], training[:, 6:]
    den_to_normalize_train = np.mean(abs(y_train))
    
    # this is actually a test set
    validation = dataset[int(.8*dataset.shape[0]):int(.9*dataset.shape[0])]
    x_val, y_val = validation[:, :6], validation[:, 6:]
    den_to_normalize_val = np.mean(abs(y_val)) 
    
    # this is a validation set for the neural network
    val_per_nn = dataset[int(.9*dataset.shape[0]):]
    x_val_nn, y_val_nn = val_per_nn[:, :6], val_per_nn[:, 6:]
    den_to_normalize_val_nn = np.mean(abs(y_val_nn))
    
    # to get also the diagonal predictions
    to_predict = np.concatenate((x_train, x_val, diag_for_x, diag_for_y, diag_for_z), axis=0)
    dictionary_with_performances = {}
    
    # to model the noise
    alphas = [1e-10, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.]
    for alpha in alphas:
        
        # radial basis function interpolation 
        
        crbf = crbfi.custom_radial_basis_function_interpolator(gamma=.0005, sigma=alpha, points=x_train, measures=y_train, stack_grid=to_predict) 
        pred = crbf.predict()
        unc = crbf.uncertainty()
        pred_train, pred_val, pred_x, pred_y, pred_z, unc_train, unc_val, unc_x, unc_y, unc_z, mae, mae_per_point, rmse, rmse_per_point, mae_train, mae_per_point_train = get_metrics(pred, unc, x_train, y_train, x_val, y_val, n_diag_points, den_to_normalize_val, den_to_normalize_train)
        dictionary_with_performances["custom radial basis function interpolator " + str(alpha)] = {"alpha": alpha, 
                                                            "nmae": mae,
                                                            "nrmse": rmse,
                                                            "uncertainty": unc_val,
                                                            "nmae per point": mae_per_point,
                                                            "nrmse per point": rmse_per_point,
                                                            "diag x preds": pred_x,
                                                            "diag y preds": pred_y,
                                                            "diag z preds": pred_z,
                                                            "unc diag x": unc_x,
                                                            "unc diag y": unc_y,
                                                            "unc diag z": unc_z,
                                                            "train nmae": mae_train
                                                            }
            
    # neural network 
    input = tf.keras.layers.Input((6))
    x = tf.keras.layers.Dense(10, activation='sigmoid')(input)
    x = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    output = tf.keras.layers.Dense(8, activation='linear')(x)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
                  loss=tf.keras.losses.MeanAbsoluteError(), 
                  metrics=[tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanSquaredError()])
    #model.summary()
    # normalized_y_train = (y_train-np.min(y_train)) / (np.max(y_train)-np.min(y_train))
    # normalized_y_val_nn = (y_val_nn-np.min(y_val_nn)) / (np.max(y_val_nn)-np.min(y_val_nn))

    history = model.fit(x_train, 
                        y_train, 
                        validation_data=(x_val_nn, y_val_nn), 
                        epochs=100, 
                        batch_size=32, 
                        verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
                        )
    
    # loss, nn_mae, nn_rmse = model.evaluate(x_val, y_val)
    # print(nn_mae/den_to_normalize_val)
    
    # nn history
    
    plt.figure()
    plt.title("Neural Network training history")
    y = history.history['mean_absolute_error']
    plt.plot(range(len(y)), y, ls='--', label='training MAE')
    
    y = history.history['val_mean_absolute_error']
    plt.plot(range(len(y)), y, ls='--', label='validation MAE')
        
    y = history.history['mean_squared_error']
    plt.plot(range(len(y)), y, ls='--', label='training MSE')
    
    y = history.history['val_mean_squared_error']
    plt.plot(range(len(y)), y, ls='--', label='validation MSE')
    
    plt.xlabel('epochs')
    plt.ylabel('training error')
    plt.legend(loc='upper right')
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.savefig(path + "nn training performance")   
    
    # crbfi error vs alpha
        
    plt.figure()
    plt.title(r'error on test set vs $\alpha$ Radial Basis Function Interpolation')
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'] for alpha in alphas]
    plt.scatter(alphas, y, marker='^', label='nMAE')
    plt.plot(alphas, y, ls='--')
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'] for alpha in alphas]
    plt.scatter(alphas, y, marker='o', label='nRMSE')
    plt.plot(alphas, y, ls='--')
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['train nmae'] for alpha in alphas]
    plt.scatter(alphas, y, marker='.', label='train nMAE')
    plt.plot(alphas, y, ls='--')
    
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('error')
    plt.legend(loc='upper right')
    plt.savefig(path + "crbfi alpha error")
    
    # crbfi vs alpha (logarithmic scale)
    
    plt.figure()
    plt.title(r'error on test set vs $\alpha$ Radial Basis Function Interpolation')
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'] for alpha in alphas]
    plt.scatter(alphas, y, marker='^', label='nMAE')
    plt.plot(alphas, y, ls='--')
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'] for alpha in alphas]
    plt.scatter(alphas, y, marker='o', label='nRMSE')
    plt.plot(alphas, y, ls='--')
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['train nmae'] for alpha in alphas]
    plt.scatter(alphas, y, marker='.', label='train nMAE')
    plt.plot(alphas, y, ls='--')
    
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('error')
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.savefig(path + "crbfi log alpha error")
    
    # pick the best alpha, in terms of nMAE (need to define what does it mean 'best alpha')
    
    alpha_star = alphas[0] 
    for alpha in alphas:
        if alpha_star < dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['train nmae']:
            alpha_star = alpha
        
    # plot which points have an higher error
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title("error on test positions (train points in green)")
    colors = dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae per point']
    # print(colors)
    ax.scatter3D(x_val[:, 0], x_val[:, 1], x_val[:, 2], c=colors, marker='o', cmap='coolwarm') # change the cmap
    ax.scatter3D(x_train[:, 0], x_train[:, 1], x_train[:, 2], c='green', marker='.', alpha=.2) # change the cmap
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.show()
    
    # plot which orientations have an higher error
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title("error on test orientations (train points in green)")
    colors = dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae per point']
    # print(colors)
    ax.scatter3D(x_val[:, 3], x_val[:, 4], x_val[:, 5], c=colors, marker='o', cmap='coolwarm') # change the cmap
    ax.scatter3D(x_train[:, 3], x_train[:, 4], x_train[:, 5], c='green', marker='.', alpha=.2) 
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.show()
    
    # correlation error and uncertainty crbfi
    
    plt.figure()
    plt.title("RBFI nMAE and uncertainty. nMAE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nmae per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nMAE')
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.savefig(path + "rbfi nmae unc")
    
    plt.figure()
    plt.title("RBFI nRMSE and uncertainty. nRMSE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nrmse per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nRMSE')
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.savefig(path + "rbfi nrmse unc")
    
    plt.figure()
    plt.title("RBFI nMAE and uncertainty. nMAE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nmae per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nMAE')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.savefig(path + "rbfi nmae unc log")
    
    plt.figure()
    plt.title("RBFI nRMSE and uncertainty. nRMSE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nrmse per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nRMSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.savefig(path + "rbfi nrmse unc log")
    
    # figure out why some points have less uncertainty than others
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title("uncertainty on test points (train points in green)")
    colors = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    colors /= max(colors)
    ax.scatter3D(x_val[:, 0], x_val[:, 1], x_val[:, 2], c=colors, marker='o', cmap='coolwarm') # change the cmap
    ax.scatter3D(x_train[:, 0], x_train[:, 1], x_train[:, 2], c='green', marker='.', alpha=.2) # change the cmap
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.show()

    # test on a diagonal
    
    plt.figure()
    plt.title("Magnetic field prediction and comparison of RBFI - x component, first coil\n(only for simulated magnetic data)")
    
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['diag x preds']
    k_x = 1.
    
    if real: 
        # rescale the predictions (has to be done also for the other components but for now I plot only the x one)
        # idk if ignoring the other components is wrong...
        k_x = sum(simulated_x[:,0]*y[:,0]) / sum(y[:,0]**2) 
    
    plt.scatter(range(n_diag_points), k_x*y[:,0], marker='o', label='predicted')
    plt.plot(range(n_diag_points), k_x*y[:,0], ls='--')
    
    plt.scatter(range(n_diag_points), simulated_x[:,0], marker='^', label='simulated')
    plt.plot(range(n_diag_points), simulated_x[:,0], ls='--')
    
    # maybe uncertainty as confidence intervals?
    plt.xlabel('diag point')
    plt.ylabel('magnetic field')
    plt.legend(loc='upper right')
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.savefig(path + "rbfi pred 1st coil x comp")

    '''
    plt.figure()
    plt.title("Magnetic field prediction and comparison of RBFI - y component, first coil\n(only for simulated magnetic data)")
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['diag y preds']
    nice_plot(range(n_diag_points), y[:,0], 'diag point', 'magnetic field (y component)', 'predicted', grid=True)
    nice_plot(range(n_diag_points), simulated_y[:,0], name='simulated', marker='^', legend_pos='upper right')
    # maybe uncertainty as confidence intervals?
    plt.savefig(path + "rbfi pred 1st coil y comp")
    
    plt.figure()
    plt.title("Magnetic field prediction and comparison of RBFI - z component, first coil\n(only for simulated magnetic data)")
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['diag z preds']
    nice_plot(range(n_diag_points), y[:,0], 'diag point', 'magnetic field (z component)', 'predicted', grid=True)
    nice_plot(range(n_diag_points), simulated_z[:,0], name='simulated', marker='^', legend_pos='upper right')
    # maybe uncertainty as confidence intervals?
    plt.savefig(path + "rbfi pred 1st coil z comp")
    '''
    
    # nn on a diagonal    
    
    plt.figure()
    plt.title("Magnetic field prediction and comparison of NN - x component, first coil\n(only for simulated magnetic data)")
    
    y = model.predict(diag_for_x)    
    plt.scatter(range(n_diag_points), y[:,0], marker='o', label='predicted')
    plt.plot(range(n_diag_points), y[:,0], ls='--')
    
    plt.scatter(range(n_diag_points), simulated_x[:,0], marker='^', label='simulated')
    plt.plot(range(n_diag_points), simulated_x[:,0], ls='--')
    
    # maybe uncertainty as confidence intervals?
    plt.xlabel('diag point')
    plt.ylabel('magnetic field')
    plt.legend(loc='upper right')
    plt.grid(color='grey', linewidth=.5, alpha=.5)
    plt.savefig(path + "nn pred 1st coil x comp")
    
    
main(smaller_cube=False, real=False)