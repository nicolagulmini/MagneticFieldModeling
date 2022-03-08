import numpy as np

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
