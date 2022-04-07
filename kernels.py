class Linear:
    def __init__(self):
        self.name= 'linear'
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return np.einsum('nd,md->nm',X,Y)
    
class Poly:
    def __init__(self, gamma, degree, coef0):
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.name= 'poly'
    def kernel(self,X,Y):
        return (self.gamma* np.einsum('nd,md->nm',X,Y) + self.coef0)**self.degree
    
class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
        self.name = 'RBF'
    def kernel(self,X,Y):
        squared_norm = np.expand_dims(np.sum(X**2,axis=1),axis=1) + np.expand_dims(np.sum(Y**2,axis=1),axis=0)-2*np.einsum('ni,mi->nm',X,Y)
        return np.exp(-0.5*squared_norm/self.sigma**2)
    
class chi2:
    def __init__(self, gamma = 1.):
        self.gamma = gamma
        self.name = 'chi2'
    def kernel(self,X,Y):
        out = np.zeros((X.shape[0], Y.shape[0]))
        n_X = X.shape[0]
        n_Y = Y.shape[0]
        n_features = X.shape[1]

        for i in range(n_X):
            for j in range(n_Y):
                p = 0
                for k in range(n_features):
                    denominateur = (X[i, k] - Y[j, k])
                    nominateur = (X[i, k] + Y[j, k])
                    if nominateur != 0:
                        p += denominateur * denominateur / nominateur
                out[i, j] = -p
        tmp = self.gamma * out
        return  np.exp(tmp, tmp)