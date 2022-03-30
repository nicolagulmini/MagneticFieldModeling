import csv
import time
from scipy.io import loadmat
import numpy as np

namefile = 'random_cloud.csv'
headers = ["x", "y", "z", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]

mat = loadmat('./MagneticFieldModeling/simulated_data/fluxes_biot_5_cube5cm.mat')

random_cloud = np.transpose(mat['PP_test_rnd'])

# points (batch)
fluxes_biot_rnd = np.swapaxes(mat['fluxes_biot_rnd'], 1, 2)
fluxes_biot_rnd = fluxes_biot_rnd.reshape((fluxes_biot_rnd.shape[0], 24))

with open(namefile, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers, lineterminator = '\n')
    csv_writer.writeheader()

with open(namefile, 'a') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers, lineterminator = '\n')
    i = 0
    while True:
        towrite = np.concatenate((random_cloud[i].flatten(), fluxes_biot_rnd[i].flatten()), axis=0)
        csv_writer.writerow({headers[j]: towrite[j] for j in range(towrite.shape[0])})
        i += 1
        if i == random_cloud.shape[0]-1:
            csv_file.close()
            break
        time.sleep(.5)