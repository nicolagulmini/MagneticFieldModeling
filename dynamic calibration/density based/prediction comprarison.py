import os
import numpy as np
import crbfi
import gpr
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

import tensorflow as tf
from tensorflow import keras


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
    
    validation = dataset[int(.8*dataset.shape[0]):int(.9*dataset.shape[0])]
    x_val, y_val = validation[:, :6], validation[:, 6:]
    den_to_normalize_val = np.mean(abs(y_val)) # this is a scalar
    
    test = dataset[int(.9*dataset.shape[0]):]
    x_test, y_test = test[:, :6], test[:, 6:]
    den_to_normalize_test = np.mean(abs(y_test))
    
    # prepare stack grid for crbfi
    
    dictionary_with_performances = {}
    
    # to model the noise
    for alpha in [1e-10, 1e-8, 1e-6, 1e-4]:
        
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
        
        # crbf = crbfi.custom_radial_basis_function_interpolator(gamma=.0005, sigma=alpha, points=x_train, measures=y_train, stack_grid=) 
        pred = crbf.predict()
        unc = crbf.uncertainty()
        mae = MAE(pred, y_val) / den_to_normalize_val * 100
        rmse = MSE(pred, y_val, squared=False) / den_to_normalize_val * 100
        dictionary_with_performances["custom radial basis function interpolator " + str(alpha)] = {"alpha": alpha, 
                                                            "nmae": mae,
                                                            "nrmse": rmse,
                                                            "uncertainty": unc}
        

    
    # neural network 
    # None for now
    
    # test on a diagonal (need to know the dimension of the cube)
    # otherwise produce a test from the same dataset of the training and the validation

    
    # once we have all of them, compare them 
    # nMAE and nRMSE on validation and test
    # time (?)
    # uncertainty
    # correlation between uncertainty and error (only for RBFI and GP)

main()