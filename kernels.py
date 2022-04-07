from tqdm import tqdm
import numpy as np
from scipy import optimize


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
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        hXX = self.kernel(X, X)
        G = np.einsum('ij,i,j->ij',hXX,y,y)
        A = np.vstack((-np.eye(N), np.eye(N)))             
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))  

        # Lagrange dual problem
        def loss(alpha):
            return -alpha.sum() + 0.5 * alpha.dot(alpha.dot(G))  #'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return -np.ones_like(alpha) + alpha.dot(G) # '''----------------partial derivative of the dual loss wrt alpha-----------------'''


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha:  np.dot(alpha, y) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:   y  #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha:  b - np.dot(A, alpha) # '''---------------function defining the ineequality constraint-------------------'''     
        jac_ineq = lambda alpha:  -A # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        
        margin_pointsIndices = (self.alpha > self.epsilon)
        boundaryIndices = (self.alpha > self.epsilon) * (self.C- self.alpha > self.epsilon )
        
        self.support = X[boundaryIndices] #'''------------------- A matrix with each row corresponding to a support vector ------------------'''
        
        self.margin_points = X[margin_pointsIndices]
        self.margin_points_AlphaY = y[margin_pointsIndices] * self.alpha[margin_pointsIndices]
        
        self.b = y[boundaryIndices][0] - self.separating_function(np.expand_dims(X[boundaryIndices][0],axis=0)) #''' -----------------offset of the linear classifier------------------ '''
        K_margin_points = self.kernel(self.margin_points, self.margin_points)
        self.norm_f = np.einsum('i,ij,j->', self.margin_points_AlphaY , K_margin_points, self.margin_points_AlphaY)


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        x1 = self.kernel(self.margin_points, x)
        return np.einsum('ij,i->j',x1,self.margin_points_AlphaY)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1
        
class KernelSVC_OneVsOne :
    
    def __init__(self, num_classes, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.num_classes = num_classes
        self.pool = {} # contain the classifiers for all classes
        for i in range(self.num_classes) : 
            for j in range(i+1, self.num_classes) : 
                self.pool[(i,j)] = KernelSVC_Binary(C=C, kernel=kernel, epsilon=epsilon)
       
    def fit(self, X, y):
        m = self.num_classes
        indices_c = {}
        for i in range(m) : 
            indices_c[i] = np.argwhere(y==i).reshape(-1)

        progress_bar = tqdm_auto(range(m * (m-1) // 2))
        for i in range(m) : 
            for j in range(i+1, m) : 
                indices_ij = np.concatenate((indices_c[i],indices_c[j]))
                self.pool[(i,j)].fit(X[indices_ij], 2*(y[indices_ij]==i)-1)   
                progress_bar.update(1)   
    
    def predict(self, X):
        """ Predictions consist in choosing the category that has the higher number of votes """
        m = self.num_classes
        y = np.zeros((len(X), m))

        for i in range(m) : 
            for j in range(i+1, m) :
                classif_ij = self.pool[(i,j)].separating_function(X) + self.pool[(i,j)].b 
                classif_ij = 1*(classif_ij > 0) # 0 if class j and 1 if class i
                y[:,i] += classif_ij
                y[:,j] += 1 - classif_ij

        return y.argmax(axis=-1)

class KernelSVC_OneVsRest :
    
    def __init__(self, num_classes, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.num_classes = num_classes
        self.pool = {} # contain the classifiers for all classes
        for i in range(self.num_classes) :  
            self.pool[i] = KernelSVC_Binary(C=C, kernel=kernel, epsilon=epsilon)
       
    def fit(self, X, y):
        for i in tqdm(range(self.num_classes)) : 
            self.pool[i].fit(X, 2*(y==i)-1)
    
    def predict(self, X):
        """ Predictions consist in choosing the category for which the classifier gives the highest value of the decision function"""
        y = np.zeros((len(X), self.num_classes))
        for i in range(self.num_classes) :
            y[:,i] = self.pool[i].separating_function(X) + self.pool[i].b  # value of the decision function
  
        return y.argmax(axis=-1)