# Magnetic Field Interpolation through Gaussian Process Regression

Python library to model and interpolate the magnetic field. The objective is to apply it on the EMT system developed by [Biomedical Design Laboratory](https://biodesignucc.ie/build/html/index.html) ([University College Cork](https://www.ucc.ie/en/), Ireland) to mitigate the distortions during a surgical operation. The challenge is to develop a software with real-time performances, which has to assist the operator during the calibration of the instrument in a fast and simple way.

For each described algorithm, the training set is a cloud of randomly sampled points inside a fixed volume. The validation set is a regular grid of points in which the three magnetic field components have to be predicted. The cuboids plots shows only the magnitude, while the prediction of the three magnetic field components are shown along the diagonal that starts from a corner of the cuboid and ends up to the opposite one. Here only a brief description of the main results with plots, the numeric results and the theory behind them are left for the paper. 

![points](https://user-images.githubusercontent.com/62892813/154316415-648a3016-045c-4fa8-8f53-236be1a13eec.png)

## Gaussian Process Regression
The following plots take into considerations different metrics vs the number of random points in the training set. The evaluations are performed over the same validation set. 

![comparison metrics training points](https://user-images.githubusercontent.com/62892813/154314397-1cc4b624-e564-4948-8562-c452c58b062f.png)

Since in a realistic scenario the magnetic field is not known, to know if the model is doing well, the standard deviation of each prediction has to be considered. So, for the following plots each point marker size corresponds to its level of 'uncertainty'.

![comparison](https://user-images.githubusercontent.com/62892813/154316745-767108df-9d21-463b-a3ab-5c57708b1935.png)

The following .gif shows the evolution of the error and the uncertainty over the validation points changing the cardinality of the training set. The reason why we have chosen a cube as a portion of space is that the system is designed as shown in the other plot. 8 coils generate the field inside a cubic portion of space, and for now we are focusing only on a fraction of it.

![comparison](https://user-images.githubusercontent.com/62892813/154314386-9d2d61eb-031f-4a78-9292-9b6344e7d532.gif)

The next two .gif show the evolution of the confidence intervals during the training, for each one of the three magnetic field components measured on the diagonal of the cube. The first one is a simulation with measurements not affected by noise, the second one yes, with an SNR between the standard deviation and the RMS of -60 dB.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/62892813/154541454-aff8b2c4-6218-4c3a-9e4a-2fe8c0c7fc66.gif)
![ezgif com-gif-maker](https://user-images.githubusercontent.com/62892813/154707767-c44c99e3-351c-4730-8800-e2a3f8aec33a.gif)

Using 24-dim points (8 coils x 3 magnetic fields components) instead of using only the first coil, the results are better, and the required computational time still remains acceptable. For instance, the correlation between the error and the uncertainty of the prediction starts to seem linear, as shown in the following plot.

![corr mae and std dev](https://user-images.githubusercontent.com/62892813/154728116-7a33c79b-0b48-4137-bd9a-2719674395b2.png)
