import numpy as np
import matplotlib.pyplot as plt

class GaussianProcessRegressionUtils():
    
    def __init__(self, n, m, D, Omega, sens=1.e-10): # change sens if you have troubles with m
        if (m >= n) or (abs(m**(1./3.)-round(m**(1./3.))) > 1.e-10):
            print("Error: m must be a perfect cube < n. Return.")
            return
        self.n = n 
        self.m = m 
        self.X = np.array([el[0] for el in D])
        self.Y = np.array([el[1] for el in D])
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
        return self.Phi_matrix
        
    def partial_derivative_phi(self, i, j, dim):
        #dim_list = [0,1,2]
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
            for j in range(self.m):
                d = i%3
                NablaPhi_matrix[i,j] = self.partial_derivative_phi(i,j,d)
        self.NablaPhi_matrix = NablaPhi_matrix
        return self.NablaPhi_matrix
        
    def kernel_function(self, x1, x2, sigma, l):
        return ( sigma**2 )*np.exp( - (np.linalg.norm(x1-x2)**2)/(2*l**2) )
    
    def build_covariance(self, sigma, l):
        covariance = np.ones(shape=(self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                covariance[i,j] = self.kernel_function(self.X[i], self.X[j], sigma, l)
        self.covariance = covariance
        return self.covariance
    
    def eigenvalue(self, j):
        to_ret = 0
        for d in range(3):
            to_ret += (np.pi * self.matrix_n[j,d]) / (2*self.Omega[d])
        return np.sqrt(to_ret)

    def S_SE(self, sigma, l, eigenval):
        return sigma**2 * (2*np.pi*(l**2))**(3./2.) * np.exp(-.5*(eigenval*l)**2)
        
    def build_Lambda_matrix(self, sigma, l):
        self.Lambda = np.diag(self.S_SE(sigma, l, self.eigenvalue(j)) for j in range(self.m) 
        return self.Lambda
        
    def build_approximateCovariance_matrix(self):
        self.approx_K = np.matmul(np.matmul(self.Phi_matrix, self.Lambda), np.transpose(self.Phi_matrix))
        return self.approx_K
