from tqdm import tqdm
import numpy as np
from scipy import optimize
import cvxopt


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
        
# Kernel SVC classifier for binary classification as in the correction of hwk 3 :

class KernelSVC_Binary:
    def __init__(self, kernel, num_classes = 10, C=1.0):
        self.C = C
        self.kernel = kernel
        
    def fit(self, X, y):
        self._K = self.kernel(X, X)
        mu_support, idx_support = self.svm_solver(self._K, y, self.C)
        w = self.get_w(mu_support, idx_support, X, y)
        b = self.compute_b(self._K, y, mu_support, idx_support)
        return w, b, mu_support, idx_support
        
    def svm_solver(self, K, y, C):
        n = y.shape[0]
        y = y.reshape((n, 1))
        H = np.dot(y, y.T)*K
        e = np.ones(n)
        A = y
        b = np.zeros(n)
        mu = self.quadratic_programming_solver(H, e, A, b, C, l=1e-8, verbose=False)
        idx_support = np.where(np.abs(mu) > 1e-5)[0]
        mu_support = mu[idx_support]
        return mu_support, idx_support
        
    def compute_b(self, K, y, mu_support, idx_support):
        num_support_vector = idx_support.size
        y_support = y[idx_support]
        K_support = K[idx_support][:, idx_support]
        b = [y_support[j] - sum([mu_support[i]*y_support[i]*K_support[i][j] for i in range(num_support_vector)]) for j in range(num_support_vector)]
        return np.mean(b)
        
    def get_w(self, mu_support, idx_support, X, y):
        return np.sum((mu_support * y[idx_support])[: , None] * X[idx_support], axis=0)

    def quadratic_programming_solver(self, H, e, A, b, C=np.inf, l=1e-8, verbose=True):
        
        n = H.shape[0]
        H = cvxopt.matrix(H)
        A = cvxopt.matrix(A, (1, n))
        e = cvxopt.matrix(-e)
        b = cvxopt.matrix(0.0)
        if C == np.inf:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.concatenate([np.diag(np.ones(n) * -1),
                                             np.diag(np.ones(n))], axis=0))
            h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

        cvxopt.solvers.options['show_progress'] = verbose
        solution = cvxopt.solvers.qp(H, e, G, h, A, b)

        mu = np.ravel(solution['x'])
        return mu
        

class KernelSVC_OneVsRest:
    def __init__(self, kernel, num_classes = 10, C=1.0):
        self.C = C
        self.kernel = kernel
    
    def fit(self, X, y):
        self._X, self._y = X, y
        
        self.labels = np.unique(y)
        self.n_labels = len(self.labels)
        self._K = self.kernel(X, X)
        # OneVsAll
        models = {}
        for idx in tqdm(range(len(self.labels))):
            label = self.labels[idx]
            models[label] = {}
            y_label = np.array([1. if e == label else -1. for e in y])
            model = KernelSVC_Binary(self.kernel)
            w, b, mu_support, idx_support = model.fit(X, y_label)
            
            models[label]['y'] = y_label
            models[label]['w'] = w
            models[label]['b'] = b  
            models[label]['mu_support'] = mu_support
            models[label]['idx_support'] = idx_support

        self.models = models
    
    def predict(self, X):
        predictions = []
        for idx in range(len(self.labels)):
            label = self.labels[idx]
            predictions.append(self._predict(X, self.models[label]['y'], self.models[label]['idx_support'], self.models[label]['mu_support'], self.models[label]['b']))
        return self.labels[np.argmax(predictions, axis=0)]
    
    def _predict(self, X, y_model, idx_support, mu_support, b):
        X_support = self._X[idx_support]
        G = self.kernel(X, X_support)
        return G.dot(mu_support * y_model[idx_support]) + b