import os
import numpy as np
import crbfi
import gpr
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

# import tensorflow as tf
# from tensorflow import keras

import matplotlib.pyplot as plt

import CoilModel as Coil

coil_model = Coil.CoilModel(module_config={'centers_x': [-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], 
                                      'centers_y': [93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]}) # mm

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp) 

origin = np.array([-50., -50., 50.])
side_length = 100.
n_diag_points = 100
x_diag = np.linspace(origin[0], origin[0]+side_length, n_diag_points)[:, np.newaxis]
y_diag = np.linspace(origin[1], origin[1]+side_length, n_diag_points)[:, np.newaxis]
z_diag = np.linspace(origin[2], origin[2]+side_length, n_diag_points)[:, np.newaxis]
diag = np.concatenate((x_diag, y_diag, z_diag), axis=1)

diag_for_x = np.concatenate((diag, np.array([[1., 0., 0.] for _ in range(n_diag_points)])), axis=1)
diag_for_y = np.concatenate((diag, np.array([[0., 1., 0.] for _ in range(n_diag_points)])), axis=1)
diag_for_z = np.concatenate((diag, np.array([[0., 0., 1.] for _ in range(n_diag_points)])), axis=1)

simulated_x = np.array([get_theoretical_field(coil_model, diag_for_x[i][:3], diag_for_x[i][3:]) for i in range(n_diag_points)])
simulated_y = np.array([get_theoretical_field(coil_model, diag_for_y[i][:3], diag_for_y[i][3:]) for i in range(n_diag_points)])
simulated_z = np.array([get_theoretical_field(coil_model, diag_for_z[i][:3], diag_for_z[i][3:]) for i in range(n_diag_points)])


def nice_plot(x, y, x_name, y_name, name, marker, grid=False, legend_pos=None):
    
    plt.scatter(x, y, marker=marker, label=name)
    plt.plot(x, y, ls='--')
    if grid: plt.grid(color='grey', linewidth=.5, alpha=.5)
    if x_name is not None: plt.xlabel(x_name)
    if y_name is not None: plt.ylabel(y_name)
    if legend_pos is not None: plt.legend(loc=legend_pos)

def main():
    if not os.path.exists('./sampled_points.csv'):
        print('There are no data.')
        return
        
    dataset = np.loadtxt('sampled_points.csv') # shape should be (n, 14)
    # n is the number of points
    # each point is 14-dimensional: 3 positions, 3 orientations, 8 coils 
    
    np.random.shuffle(dataset)
    training = dataset[:int(.8*dataset.shape[0])]
    x_train, y_train = training[:, :6], training[:, 6:]
    
    validation = dataset[int(.8*dataset.shape[0]):]#int(.9*dataset.shape[0])]
    x_val, y_val = validation[:, :6], validation[:, 6:]
    den_to_normalize_val = np.mean(abs(y_val)) # this is a scalar
    
    # test = dataset[int(.9*dataset.shape[0]):]
    # x_test, y_test = test[:, :6], test[:, 6:]
    # den_to_normalize_test = np.mean(abs(y_test))
    
    # prepare stack grid for crbfi
    
    dictionary_with_performances = {}
    
    # to model the noise
    alphas = [1e-10]#, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.]
    for alpha in alphas:
        
        # gaussian process regression
        
        gp = gpr.gaussian_process_regressor(alpha=alpha, points=x_train, measures=y_train)
        gp.fit()
        pred, unc = gp.predict(x_val)
        mae = MAE(pred, y_val) / den_to_normalize_val * 100
        rmse = MSE(pred, y_val, squared=False) / den_to_normalize_val * 100
        r2 = gp.score(x_val, y_val)
        dictionary_with_performances["gaussian process " + str(alpha)] = {"alpha": alpha, 
                                                            "nmae": mae,
                                                            "nrmse": rmse,
                                                            "r2": r2,
                                                            "uncertainty": unc}
        
        # radial basis function interpolation 
        
        crbf = crbfi.custom_radial_basis_function_interpolator(gamma=.0005, sigma=alpha, points=x_train, measures=y_train, stack_grid=x_val) 
        pred = crbf.predict()
        unc = crbf.uncertainty()
        
        mae_per_point = [MAE(pred[i], y_val[i]) for i in range(y_val.shape[0])]
        mae = sum(mae_per_point)/len(mae_per_point)
        mae = mae / den_to_normalize_val * 100
        
        rmse_per_point = [MSE(pred[i], y_val[i], squared=False) for i in range(y_val.shape[0])]
        rmse = sum(rmse_per_point)/len(rmse_per_point)
        rmse = rmse / den_to_normalize_val * 100
        
        crbf = crbfi.custom_radial_basis_function_interpolator(gamma=.0005, sigma=alpha, points=x_train, measures=y_train, stack_grid=diag_for_x) 
        pred_x = crbf.predict()
        crbf = crbfi.custom_radial_basis_function_interpolator(gamma=.0005, sigma=alpha, points=x_train, measures=y_train, stack_grid=diag_for_y) 
        pred_y = crbf.predict()
        crbf = crbfi.custom_radial_basis_function_interpolator(gamma=.0005, sigma=alpha, points=x_train, measures=y_train, stack_grid=diag_for_z) 
        pred_z = crbf.predict()
        
        dictionary_with_performances["custom radial basis function interpolator " + str(alpha)] = {"alpha": alpha, 
                                                            "nmae": mae,
                                                            "nrmse": rmse,
                                                            "uncertainty": unc,
                                                            "nmae per point": mae_per_point,
                                                            "nrmse per point": rmse_per_point,
                                                            "diag x preds": pred_x,
                                                            "diag y preds": pred_y,
                                                            "diag z preds": pred_z,
                                                            }
    
        # neural network 
        # None for now
        
        # test also on a diagonal (need to know the dimension of the cube)

    '''
    # gaussian process error vs alpha
    
    plt.figure()
    plt.title(r'error vs $\alpha$ Gaussian Process Regression')
    y = [dictionary_with_performances["gaussian process " + str(alpha)]['nmae'] for alpha in alphas]
    nice_plot(alphas, y, r'$\alpha$', 'error', 'nMAE', marker='^', grid=True)
    
    y = [dictionary_with_performances["gaussian process " + str(alpha)]['nrmse'] for alpha in alphas]
    nice_plot(alphas, y, None, None, 'nRMSE', marker='o', legend_pos='lower right')
    plt.xscale('log')
    plt.savefig('gpr alpha error')
    
    # gassian process r2 vs alpha
    
    plt.figure()
    plt.title(r'$R^2$ vs $\alpha$ Gaussian Process Regression')
    y = [dictionary_with_performances["gaussian process " + str(alpha)]['r2'] for alpha in alphas]
    nice_plot(alphas, y, r'$\alpha$', 'error', 'nMAE', marker='^', grid=True, legend_pos='lower right')
    plt.xscale('log')
    plt.savefig('gpr alpha r2')
    
    # crbfi error vs alpha
    
    plt.figure()
    plt.title(r'error vs $\alpha$ Radial Basis Function Interpolation')
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'] for alpha in alphas]
    nice_plot(alphas, y, r'$\alpha$', 'error', 'nMAE', marker='^', grid=True)
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'] for alpha in alphas]
    nice_plot(alphas, y, None, None, 'nRMSE', marker='o', legend_pos='lower right')
    plt.xscale('log')
    plt.savefig('crbfi alpha error')
    
    # correlation error and uncertainty crbfi
    
    plt.figure()
    plt.title("RBFI nMAE and uncertainty. nMAE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['nmae per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nMAE')
    plt.savefig('rbfi nmae unc')
    
    plt.figure()
    plt.title("RBFI nRMSE and uncertainty. nRMSE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['nrmse per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nRMSE')
    plt.savefig('rbfi nrmse unc')
    
    plt.figure()
    plt.title("RBFI nMAE and uncertainty. nMAE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nmae'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['nmae per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nMAE')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('rbfi nmae unc log')
    
    plt.figure()
    plt.title("RBFI nRMSE and uncertainty. nRMSE = " + str(round(dictionary_with_performances["custom radial basis function interpolator " + str(alpha)]['nrmse'], 4)) + " %")
    x = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['uncertainty']
    y = dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['nrmse per point']
    plt.scatter(x, y, marker='.', alpha=.5, color='black')
    plt.xlabel('uncertainty')
    plt.ylabel('nRMSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('rbfi nrmse unc log')
    '''

    plt.figure()
    plt.plot(range(n_diag_points), dictionary_with_performances["custom radial basis function interpolator " + str(1e-10)]['diag x preds'][:, 0], color='red')
    plt.plot(range(n_diag_points), np.array(simulated_x)[:, 0, 0], color='green')
    plt.show()

main()

# time (?)









