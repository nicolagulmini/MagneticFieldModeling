import os
import numpy as np
import crbfi
import gpr
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
import matplotlib.pyplot as plt
import CoilModel as Coil
# import tensorflow as tf
# from tensorflow import keras

filename = "sampled_points"

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp) 
    
def get_metrics(pred, unc, x_train, y_train, x_val, y_val, n_diag_points, den_to_normalize_val, den_to_normalize_train):
    pred_train, pred_val, pred_x, pred_y, pred_z = pred[:x_train.shape[0]], pred[x_train.shape[0]:x_train.shape[0]+x_val.shape[0]], pred[x_train.shape[0]+x_val.shape[0]:x_train.shape[0]+x_val.shape[0]+n_diag_points], pred[x_train.shape[0]+x_val.shape[0]+n_diag_points:x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points)], pred[x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points):]
    unc_train, unc_val, unc_x, unc_y, unc_z = unc[:x_train.shape[0]], unc[x_train.shape[0]:x_train.shape[0]+x_val.shape[0]], unc[x_train.shape[0]+x_val.shape[0]:x_train.shape[0]+x_val.shape[0]+n_diag_points], unc[x_train.shape[0]+x_val.shape[0]+n_diag_points:x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points)], unc[x_train.shape[0]+x_val.shape[0]+int(2*n_diag_points):]
    
    mae_per_point_train = [MAE(pred_train[i], y_train[i])/den_to_normalize_train for i in range(y_train.shape[0])]
    mae_train = sum(mae_per_point_train)/len(mae_per_point_train) * 100
    
    mae_per_point = [MAE(pred_val[i], y_val[i])/den_to_normalize_val for i in range(y_val.shape[0])]
    mae = sum(mae_per_point)/len(mae_per_point) * 100
    
    rmse_per_point = [MSE(pred_val[i], y_val[i], squared=False)/den_to_normalize_val for i in range(y_val.shape[0])]
    rmse = sum(rmse_per_point)/len(rmse_per_point) * 100
    
    return pred_train, pred_val, pred_x, pred_y, pred_z, unc_train, unc_val, unc_x, unc_y, unc_z, mae, mae_per_point, rmse, rmse_per_point, mae_train, mae_per_point_train
    

def main(origin=np.array([-25., -25., 100.]), side_length=50., n_diag_points=50, centers_x=[-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], centers_y=[93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]):
    # put the origin of the cube, the side length and the number of points along the diagonal manually
    
    if not os.path.exists("./" + filename + ".csv"):
        print('There are no data.')
        return
    
    # the coil model is necessary to get the comparison for the simulated magnetic data
    # it makes sense almost only if we sample with a simulated magnetic field. Otherwise just ignore the related section.
    coil_model = Coil.CoilModel(module_config={'centers_x': centers_x, 'centers_y': centers_y})

    x_diag = np.linspace(origin[0], origin[0]+side_length, n_diag_points)[:, np.newaxis]
    y_diag = np.linspace(origin[1], origin[1]+side_length, n_diag_points)[:, np.newaxis]
    z_diag = np.linspace(origin[2], origin[2]+side_length, n_diag_points)[:, np.newaxis]
    diag = np.concatenate((x_diag, y_diag, z_diag), axis=1)

    diag_for_x = np.concatenate((diag, np.array([[1., 0., 0.] for _ in range(n_diag_points)])), axis=1)
    diag_for_y = np.concatenate((diag, np.array([[0., 1., 0.] for _ in range(n_diag_points)])), axis=1)
    diag_for_z = np.concatenate((diag, np.array([[0., 0., 1.] for _ in range(n_diag_points)])), axis=1)

    simulated_x = np.array([get_theoretical_field(coil_model, diag_for_x[i][:3], diag_for_x[i][3:]).A1 for i in range(n_diag_points)])
    simulated_y = np.array([get_theoretical_field(coil_model, diag_for_y[i][:3], diag_for_y[i][3:]).A1 for i in range(n_diag_points)])
    simulated_z = np.array([get_theoretical_field(coil_model, diag_for_z[i][:3], diag_for_z[i][3:]).A1 for i in range(n_diag_points)])
        
    dataset = np.loadtxt("./" + filename + ".csv") # shape should be (n, 14)
    # n is the number of points
    # each point is 14-dimensional: 3 positions, 3 orientations, 8 coils 
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title("gathered points")
    ax.scatter3D(dataset[:, 0], dataset[:, 1], dataset[:, 2], alpha=.1, marker='.')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    #plt.savefig('comparison.png')
    plt.show()
    
    np.random.shuffle(dataset)
    training = dataset[:int(.8*dataset.shape[0])]
    x_train, y_train = training[:, :6], training[:, 6:]
    den_to_normalize_train = np.mean(abs(y_train))
    
    validation = dataset[int(.8*dataset.shape[0]):]#int(.9*dataset.shape[0])]
    x_val, y_val = validation[:, :6], validation[:, 6:]
    den_to_normalize_val = np.mean(abs(y_val)) # this is a scalar
    
    # test = dataset[int(.9*dataset.shape[0]):]
    # x_test, y_test = test[:, :6], test[:, 6:]
    # den_to_normalize_test = np.mean(abs(y_test))
    
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
    # None for now
    
    # test also on a diagonal (need to know the dimension of the cube)

 
    # crbfi error vs alpha
    
    def nice_plot(x, y, x_name=None, y_name=None, name='', marker='o', grid=False, legend_pos=None):
        plt.scatter(x, y, marker=marker, label=name)
        plt.plot(x, y, ls='--')
        if grid: plt.grid(color='grey', linewidth=.5, alpha=.5)
        if x_name is not None: plt.xlabel(x_name)
        if y_name is not None: plt.ylabel(y_name)
        if legend_pos is not None: plt.legend(loc=legend_pos)
    
    plt.figure()
    plt.title(r'error on test set vs $\alpha$ Radial Basis Function Interpolation')
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'] for alpha in alphas]
    nice_plot(alphas, y, r'$\alpha$', 'error', 'nMAE', marker='^', grid=True)
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'] for alpha in alphas]
    nice_plot(alphas, y, None, None, 'nRMSE', marker='o')
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['train nmae'] for alpha in alphas]
    nice_plot(alphas, y, None, None, 'nMAE on train set', marker='.', legend_pos='upper right')
    plt.savefig('crbfi alpha error')
    
    # crbfi vs alpha (logarithmic scale)
    plt.figure()
    plt.title(r'error on test set vs $\alpha$ Radial Basis Function Interpolation')
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'] for alpha in alphas]
    nice_plot(alphas, y, r'$\alpha$', 'error', 'nMAE', marker='^', grid=True)
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'] for alpha in alphas]
    nice_plot(alphas, y, None, None, 'nRMSE', marker='o')
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['train nmae'] for alpha in alphas]
    nice_plot(alphas, y, None, None, 'nMAE on train set', marker='.', legend_pos='upper right')
    plt.xscale('log')
    plt.savefig('crbfi alpha error log')
    
    alpha_star = alphas[0] # pick the best alpha on training set 
    for alpha in alphas:
        if dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['train nmae'] < dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['train nmae']:
            alpha_star = alpha
    alpha_star = 1e-7
    
    # correlation error and uncertainty crbfi
    
    plt.figure()
    plt.title("RBFI nMAE and uncertainty. nMAE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nmae per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nMAE')
    plt.savefig('rbfi nmae unc')
    
    plt.figure()
    plt.title("RBFI nRMSE and uncertainty. nRMSE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nrmse per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nRMSE')
    plt.savefig('rbfi nrmse unc')
    
    plt.figure()
    plt.title("RBFI nMAE and uncertainty. nMAE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nmae per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nMAE')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('rbfi nmae unc log')
    
    plt.figure()
    plt.title("RBFI nRMSE and uncertainty. nRMSE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['nrmse per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nRMSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('rbfi nrmse unc log')

    
    # only for simulated magnetic data
    
    plt.figure()
    plt.title("Magnetic field prediction and comparison of RBFI - x component, first coil\n(only for simulated magnetic data)")
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['diag x preds']
    nice_plot(range(n_diag_points), y[:,0], 'diag point', 'magnetic field (x component)', 'predicted', grid=True)
    nice_plot(range(n_diag_points), simulated_x[:,0], name='simulated', marker='^', legend_pos='upper right')
    # maybe uncertainty as confidence intervals?
    plt.savefig('rbfi pred 1st coil x comp')
    # we can choose to plot different coils and also different components
    
    plt.figure()
    plt.title("Magnetic field prediction and comparison of RBFI - y component, first coil\n(only for simulated magnetic data)")
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['diag y preds']
    nice_plot(range(n_diag_points), y[:,0], 'diag point', 'magnetic field (y component)', 'predicted', grid=True)
    nice_plot(range(n_diag_points), simulated_y[:,0], name='simulated', marker='^', legend_pos='upper right')
    # maybe uncertainty as confidence intervals?
    plt.savefig('rbfi pred 1st coil y comp')
    
    plt.figure()
    plt.title("Magnetic field prediction and comparison of RBFI - z component, first coil\n(only for simulated magnetic data)")
    y = dictionary_with_performances["custom radial basis function interpolator " + str(alpha_star)]['diag z preds']
    #nice_plot(range(n_diag_points), y[:,0], 'diag point', 'magnetic field (z component)', 'predicted', grid=True)
    nice_plot(range(n_diag_points), simulated_z[:,0], name='simulated', marker='^', legend_pos='upper right')
    # maybe uncertainty as confidence intervals?
    plt.savefig('rbfi pred 1st coil z comp')

main()