import os
import numpy as np
import crbfi
import gpr
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
import matplotlib.pyplot as plt
import CoilModel as Coil

number_of_points = [200, 800, 2440, 6660, 16250]
average_coverage = [4.94, 15.78, 36.8, 65.4, 76.04]

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp) 

def nice_plot(x, y, x_name=None, y_name=None, name='', marker='o', grid=False, legend_pos=None):
    plt.scatter(x, y, marker=marker, label=name)
    plt.plot(x, y, ls='--')
    if grid: plt.grid(color='grey', linewidth=.5, alpha=.5)
    if x_name is not None: plt.xlabel(x_name)
    if y_name is not None: plt.ylabel(y_name)
    if legend_pos is not None: plt.legend(loc=legend_pos)
    
def get_metrics(pred, unc, x_val, y_val, n_diag_points, den_to_normalize_val):
    pred_val, pred_x, pred_y, pred_z = pred[:x_val.shape[0]], pred[x_val.shape[0]:x_val.shape[0]+n_diag_points], pred[x_val.shape[0]+n_diag_points:x_val.shape[0]+int(2*n_diag_points)], pred[x_val.shape[0]+int(2*n_diag_points):]
    unc_val, unc_x, unc_y, unc_z = unc[:x_val.shape[0]], unc[x_val.shape[0]:x_val.shape[0]+n_diag_points], unc[x_val.shape[0]+n_diag_points:x_val.shape[0]+int(2*n_diag_points)], unc[x_val.shape[0]+int(2*n_diag_points):]
    
    mae_per_point = [MAE(pred_val[i], y_val[i])/den_to_normalize_val for i in range(y_val.shape[0])]
    mae = sum(mae_per_point)/len(mae_per_point) * 100
    
    rmse_per_point = [MSE(pred_val[i], y_val[i], squared=False)/den_to_normalize_val for i in range(y_val.shape[0])]
    rmse = sum(rmse_per_point)/len(rmse_per_point) * 100
    
    return pred_val, pred_x, pred_y, pred_z, unc_val, unc_x, unc_y, unc_z, mae, mae_per_point, rmse, rmse_per_point
    

def main(origin=np.array([-100., -100., 50.]), side_length=200., n_diag_points=50, centers_x=[-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], centers_y=[93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]):

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
        
    dictionary_with_performances = {}
    
    dataset = np.loadtxt("./5.csv")

    np.random.shuffle(dataset)
    training_5 = dataset[:int(.8*dataset.shape[0])]
    x_train_5, y_train_5 = training_5[:, :6], training_5[:, 6:]
    
    validation = dataset[int(.8*dataset.shape[0]):]
    x_val, y_val = validation[:, :6], validation[:, 6:]
    den_to_normalize_val = np.mean(abs(y_val))
    
    to_predict = np.concatenate((x_val, diag_for_x, diag_for_y, diag_for_z), axis=0)
    
    for i in [1, 2, 3, 4, 5]:
        dataset = np.loadtxt("./" + str(i) + ".csv")
    
        np.random.shuffle(dataset)
        training = dataset[:int(.8*dataset.shape[0])]
        x_train, y_train = training[:, :6], training[:, 6:]
        
        if i == 5:
            x_train, y_train = x_train_5, y_train_5

        crbf = crbfi.custom_radial_basis_function_interpolator(gamma=.0005, sigma=1e-10, points=x_train, measures=y_train, stack_grid=to_predict) 
        pred = crbf.predict()
        unc = crbf.uncertainty()
        pred_val, pred_x, pred_y, pred_z, unc_val, unc_x, unc_y, unc_z, mae, mae_per_point, rmse, rmse_per_point = get_metrics(pred, unc, x_val, y_val, n_diag_points, den_to_normalize_val)
        dictionary_with_performances["custom radial basis function interpolator " + str(i)] = {"experiment": i, 
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
                                                            "unc diag z": unc_z
                                                            }
    
    # crbfi error vs alpha
    '''
    plt.figure()
    plt.title('error vs average % coverage\nRadial Basis Function Interpolation')
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(i)]['nmae'] for i in [1, 2, 3, 4, 5]]
    nice_plot(average_coverage, y, "average % coverage", 'error', 'nMAE', marker='^', grid=True)
    
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(i)]['nrmse'] for i in [1, 2, 3, 4, 5]]
    nice_plot(average_coverage, y, None, None, 'nRMSE', marker='o', legend_pos='upper right')
    #plt.xscale('log')
    plt.savefig('crbfi alpha error')'''
    
    plt.figure()
    plt.title("Magnetic field prediction and comparison of RBFI - x component, first coil\n(only for simulated magnetic data)")
    y = [dictionary_with_performances["custom radial basis function interpolator " + str(i)]['diag x preds'] for i in [1, 2, 3, 4, 5]]
    nice_plot(range(n_diag_points), y[0][:, 0], 'diag point', 'magnetic field (x component)', '1', marker='', grid=True)
    nice_plot(range(n_diag_points), y[1][:, 0], name='2', marker='')
    nice_plot(range(n_diag_points), y[2][:, 0], name='3', marker='')
    nice_plot(range(n_diag_points), y[3][:, 0], name='4', marker='')
    nice_plot(range(n_diag_points), y[4][:, 0], name='5', marker='')
    nice_plot(range(n_diag_points), simulated_x[:,0], name='simulated', marker='^', legend_pos='lower left')
    # maybe uncertainty as confidence intervals?
    plt.savefig('rbfi pred 1st coil x comp')
    # we can choose to plot different coils and also different components
    
'''
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
    nice_plot(range(n_diag_points), y[:,0], 'diag point', 'magnetic field (z component)', 'predicted', grid=True)
    nice_plot(range(n_diag_points), simulated_z[:,0], name='simulated', marker='^', legend_pos='upper right')
    # maybe uncertainty as confidence intervals?
    plt.savefig('rbfi pred 1st coil z comp') '''
    
main()