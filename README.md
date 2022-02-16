# Magnetic Field Interpolation through Gaussian Process Regression

Python implementation of an algorithm to model and interpolate the magnetic field. The objective is to apply it on the EMT system developed by [Biomedical Design Laboratory](https://biodesignucc.ie/build/html/index.html) ([University College Cork](https://www.ucc.ie/en/), Ireland) to mitigate the distortions. The challenge is to develop a software with good real time performances (about 20 kHz), which has also to assist the operator during the calibration of the instrument in a fast and simple way.

The GP regressor is trained using a cloud of randomly sampled points inside a fixed volume. The validation set is a regular grid of points in which the three magnetic field components have to be predicted (the plot shows only the magnitude). 
![points](https://user-images.githubusercontent.com/62892813/154316415-648a3016-045c-4fa8-8f53-236be1a13eec.png)

The following plots take into considerations different metrics vs the number of random points in the training set. The evaluations are performed over the same validation set. 
![comparison metrics training points](https://user-images.githubusercontent.com/62892813/154314397-1cc4b624-e564-4948-8562-c452c58b062f.png)

Since in a realistic scenario the magnetic field is not known, to know if the model is doing well, the standard deviation of each prediction has to be considered. So, for the following plots each point marker size corresponds to its level of 'uncertainty'.
![comparison](https://user-images.githubusercontent.com/62892813/154316745-767108df-9d21-463b-a3ab-5c57708b1935.png)

The following .gif shows the evolution of the error and the uncertainty over the validation points changing the cardinality of the training set.
![movie](https://user-images.githubusercontent.com/62892813/154314386-9d2d61eb-031f-4a78-9292-9b6344e7d532.gif)
