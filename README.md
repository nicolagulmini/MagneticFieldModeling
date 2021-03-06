Python library to model and interpolate the magnetic field. The objective is to apply it on the EMT system developed by [Biomedical Design Laboratory](https://biodesignucc.ie/build/html/index.html) ([University College Cork](https://www.ucc.ie/en/), Ireland) to mitigate the distortions during a surgical operation. The challenge is to develop a software with real-time performances, which has to assist the operator during the calibration of the instrument in a fast and simple way.

For each described algorithm, the training set is a cloud of randomly sampled points inside a fixed volume. The validation set is a regular grid of points in which the three magnetic field components have to be predicted. The cuboids plots shows only the magnitude, while the prediction of the three magnetic field components are shown along the diagonal that starts from a corner of the cuboid and ends up to the opposite one. Here only a brief description of the main results with plots, the numeric results and the theory behind them are left for the paper. 

![points](https://user-images.githubusercontent.com/62892813/154316415-648a3016-045c-4fa8-8f53-236be1a13eec.png)

## [Gaussian Process Regression](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
The following plots take into considerations different metrics vs the number of random points in the training set. The evaluations are performed over the same validation set. 

![comparison metrics training points](https://user-images.githubusercontent.com/62892813/154314397-1cc4b624-e564-4948-8562-c452c58b062f.png)

Since in a realistic scenario the magnetic field is not known, to know if the model is doing well, the standard deviation of each prediction has to be considered. So, for the following plots each point marker size corresponds to its level of 'uncertainty'.

![comparison](https://user-images.githubusercontent.com/62892813/154316745-767108df-9d21-463b-a3ab-5c57708b1935.png)

The next two .gif show the evolution of the confidence intervals during the training, for each one of the three magnetic field components measured on the diagonal of the cube. The first one is a simulation with measurements not affected by noise, the second one yes, with an SNR between the standard deviation and the RMS of -60 dB.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/62892813/154541454-aff8b2c4-6218-4c3a-9e4a-2fe8c0c7fc66.gif)
![ezgif com-gif-maker](https://user-images.githubusercontent.com/62892813/154707767-c44c99e3-351c-4730-8800-e2a3f8aec33a.gif)

Using 24-dim points (8 coils x 3 magnetic fields components) instead of using only the first coil, the results are better, and the required computational time still remains acceptable. For instance, the correlation between the error and the uncertainty of the prediction starts to seem linear, as shown in the following plot.

![corr mae and std dev](https://user-images.githubusercontent.com/62892813/156613484-59d23443-7b91-4161-ac6c-1c4b553e494d.png)

## [Radial Basis Function Interpolation](https://en.wikipedia.org/wiki/Radial_basis_function_interpolation)

Fixing the hyperparameter of the kernel, i.e. the length scale, we can obtain a vector of weights simply solving a linear system of equations for each component. Computing the kernel matrix of the training points, only one line of code is enough to obtain so
```Python
W = np.linalg.solve(sklearn.metrics.pairwise.rbf_kernel(training_positions, gamma=gamma), magnetic_field_measurements)
```
and then, computing the kernel between the training points vs the validation ones, after a matrix multiplication we are able to predict the magnetic field, as shown in the following .gif

![movie](https://user-images.githubusercontent.com/62892813/156361455-8c8ca59d-6acb-43d6-ae93-d0b00065e34f.gif)

and the following plot shows the correlation between the nMAE and the standard deviation (computed through the [Cholensky decomposition](https://stats.stackexchange.com/questions/330185/how-to-calculate-the-standard-deviation-for-a-gaussian-process)) for each one of the 24-dim points

![corr mae and std dev RBF](https://user-images.githubusercontent.com/62892813/156600120-dcf971cb-ce06-461f-b1f1-899830ac44f8.png)

### Custom RBF Interpolation

Taking into account also the orientation information, and with the uniaxial measures (one-dimensional measures of the magnetic field instead of the three components, allowing a smaller sensor), it is possible to interpolate the magnetic field changing the definition of the radial basis function. 

![diag results](https://user-images.githubusercontent.com/62892813/157286164-9d806e2a-c2cf-4332-afda-797c6aa9cbaa.png)

## Calibration
These algorithms are useful to calibrate the instrument. Collecting sensor's data it is possible to obtain a map of the static magnetic field, and the real time feedback from the program is useful to understand which area has to be covered during the sampling.
This is the first interface of the program:

![interface](https://user-images.githubusercontent.com/62892813/166935564-95d033ca-14cd-4cbd-9d26-7ae9cad18cb2.JPG)

in which:
- each cube shows a fixed grid of points from red to green, that means not-covered and well-covered, respectively
- the position and orientation of the sensor are shown as a cone in each cube
- the first cube shows the uncertainty relative to the x-component of the field, the second cube is relative to the y-component, and the third is for z
- each cube is independent in terms of visualization, so it is possible to zoom, move and rotate them as you want, even during the sampling.
