import numpy as np
import matplotlib.pyplot as plt

class GaussianProcessRegressionUtils():
    
    def __init__(self, n, m, X, Y, Omega, sens=1.e-10): # change sens if you have troubles with m
        if (m >= n) or (abs(m**(1./3.)-round(m**(1./3.))) > 1.e-10) or (X.shape[0] != n):
            print("Error: m must be a perfect cube < n. Return.")
            return
        self.n = n 
        self.m = m 
        self.X = X
        self.Y = Y
        self.vecY = np.concatenate(self.Y, axis=None)
        self.Omega = Omega
        matrix_n = np.ones((m, 3))
        count = 0
        m_hat = round(m**(1/3))
        for i in range(1, m_hat+1):
            for j in range(1, m_hat+1):
                for k in range(1, m_hat+1):
                    matrix_n[count] = np.array([i, j, k])
                    count += 1
        self.matrix_n = matrix_n
            
    def phi_j(self, i, j, range_d=[0, 1, 2]):
        to_ret = 1.
        for d in range_d:
            to_ret *= (1/np.sqrt(self.Omega[d])) * np.sin( (np.pi * self.matrix_n[j,d] * (self.X[i,d]+self.Omega[d])) / (2 * self.Omega[d]) )
        return to_ret
        
    def build_Phi_matrix(self):
        Phi_matrix = np.ones(shape=(self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                Phi_matrix[i,j] = self.phi_j(i, j)
        self.Phi_matrix = Phi_matrix
        
    def partial_derivative_phi(self, i, j, dim):
        dim_list = [0,1,2]
        if dim not in dim_list:
            print("Error: derivative is only for x, y or z dimension. Return.")
            return
        ''' # unoptimised method
        cos_term = (1/np.sqrt(self.Omega[dim])) * np.cos( np.pi * self.matrix_n[j,dim] * (self.X[i,dim]+self.Omega[dim]) / (2 * self.Omega[dim]) ) * ( (np.pi * self.matrix_n[j,dim]) / (2 * self.Omega[dim]) )
        dim_list.remove(dim)
        other_term = self.phi_j(i, j, range_d=dim_list)
        return cos_term*other_term    
        '''
        return np.pi*.5*self.phi_j(i, j)/np.tan( np.pi * self.matrix_n[j,dim] * (self.X[i,dim]+self.Omega[dim]) / (2 * self.Omega[dim]) )
        
    '''
    def build_NablaPhi_matrix(self):
        #  I think that the tensor will become a 3n x m matrix
        NablaPhi_matrix = np.ones(shape=(self.n, self.m, 3))
        for i in range(self.n):
            for j in range(self.m):
                for d in range(3):
                    NablaPhi_matrix[i,j,d] = self.partial_derivative_phi(i,j,d)
        self.NablaPhi_matrix = NablaPhi_matrix
        return self.NablaPhi_matrix
    '''
    
    def build_NablaPhi_matrix(self):
        NablaPhi_matrix = np.ones(shape=(3*self.n, self.m))
        for i in range(3*self.n):
            d = i % 3
            for j in range(self.m):
                NablaPhi_matrix[i,j] = self.partial_derivative_phi(int(i/3),j,d)
        self.NablaPhi_matrix = NablaPhi_matrix
        
    def kernel_function(self, x1, x2, sigma, l):
        return ( sigma**2 )*np.exp( - (np.linalg.norm(x1-x2)**2)/(2*l**2) )
    
    def build_covariance(self, sigma, l):
        covariance = np.ones(shape=(self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                covariance[i,j] = self.kernel_function(self.X[i], self.X[j], sigma, l)
        self.covariance = covariance
    
    def eigenvalue(self, j):
        to_ret = 0
        for d in range(3):
            to_ret += (np.pi * self.matrix_n[j,d]) / (2*self.Omega[d])
        return np.sqrt(to_ret)

    def S_SE(self, sigma, l, eigenval):
        return sigma**2 * (2*np.pi*(l**2))**(3./2.) * np.exp(-.5*(eigenval*l)**2)
        
    def build_Lambda_matrix(self, sigma, l):
        self.Lambda = np.diag([self.S_SE(sigma, l, self.eigenvalue(j)) for j in range(self.m)])
        
    def build_approximateCovariance_matrix(self):
        phi_T = np.transpose(self.Phi_matrix)
        first_term = np.matmul(self.Phi_matrix, self.Lambda)
        self.approx_K = np.matmul(first_term, phi_T)
        
    def eval_phi_j(self, x, j, range_d=[0, 1, 2]):
        to_ret = 1.
        for d in range_d:
            to_ret *= (1/np.sqrt(self.Omega[d])) * np.sin( (np.pi * self.matrix_n[j,d] * (x[d]+self.Omega[d])) / (2 * self.Omega[d]) )
        return to_ret
    
    def eval_partial_derivative_phi(self, x, j, dim):
        return np.pi*.5*self.eval_phi_j(x, j)/np.tan( np.pi * self.matrix_n[j,dim] * (x+self.Omega[dim]) / (2 * self.Omega[dim]) )

    def compute_approx_expectation(self, x, sigma, l, sigma_noise):
        first_term = np.ones(shape=(3, self.m))
        for i in range(3):
            for j in range(m):
                first_term[i][j] = self.eval_partial_derivative_phi(x, j, i)

        second_term = np.linalg.inv(np.matmul(np.transpose(self.NablaPhi_matrix), self.NablaPhi_matrix) + sigma_noise**2*np.linalg.inv(self.Lambda))
        mul = np.matmul(first_term, second_term)
        mul = np.matmul(mul, np.transpose(self.NablaPhi_matrix))
        mul = np.matmul(mul, self.vecY)
        return mul
    
    def compute_approx_likelihood(self, sigma, l, sigma_noise):
        NablaPhi_matrix_T = np.transpose(self.NablaPhi_matrix)
        sn2 = sigma_noise**2
        tr_Lambda = np.sum(self.Lambda)
        Lambda_inv = np.linalg.inv(self.Lambda)
        vecY_T = np.transpose(self.vecY)
        vecYmul = np.matmul(vecY_T, self.vecY)
        NTN_Phi = np.matmul(NablaPhi_matrix_T, self.NablaPhi_matrix)

        first_term = (3*self.n-self.m) * np.log(sn2) + tr_Lambda + np.linalg.slogdet(sn2*Lambda_inv+NTN_Phi)[1]
        sec_term = vecYmul - np.matmul(np.matmul(np.matmul(np.matmul(vecY_T, self.NablaPhi_matrix), np.linalg.inv(sn2*Lambda_inv+NTN_Phi)), NablaPhi_matrix_T), self.vecY)
        sec_term /= sn2
        third_term = 3*self.n*np.log(2*np.pi)
        loss = first_term + sec_term + third_term
        return .5*loss
