import numpy as np
from scipy.optimize import minimize


# NOTE: follow the docstrings. In-line comments can be followed, or replaced.
#       Hence, those are the steps, but if it does not match your approach feel
#       free to remove.

def linear_kernel(X1, X2):
    """    Matrix multiplication.

    Given two matrices, A (m X n) and B (n X p), multiply: AB = C (m X p).

    Recall from hw 1. Is there a more optimal way to implement using numpy?
    :param X1:  Matrix A
    type       np.array()
    :param X2:  Matrix B
    type       np.array()

    :return:    C Matrix.
    type       np.array()
    """
    # TODO: implement
    # Use numpy's dot product function to compute the matrix multiplication
    return np.dot(X1, X2.T) # or X1 @ X2.T


def nonlinear_kernel(X1, X2, sigma=0.5):
    """
     Compute the value of a nonlinear kernel function for a pair of input vectors.

     Args:
         X1 (numpy.ndarray): A vector of shape (n_features,) representing the first input vector.
         X2 (numpy.ndarray): A vector of shape (n_features,) representing the second input vector.
         sigma (float): The bandwidth parameter of the Gaussian kernel.

     Returns:
         The value of the nonlinear kernel function for the pair of input vectors.

     """
    # (Bonus) TODO: implement 

    # Compute the Euclidean distance between the input vectors
    distance = np.linalg.norm(X1 - X2)
    # Compute the pairwise Euclidean distances between the input vectors
    # distances = np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)

    # Compute the value of the Gaussian kernel function
    kernel_value = np.exp(-distance ** 2 / (2 * sigma ** 2))
    # Return the kernel value
    return kernel_value


def objective_function(X, y, a, kernel):
    """
    Compute the value of the objective function for a given set of inputs.

    Args:
        X (numpy.ndarray): An array of shape (n_samples, n_features) representing the input data.
        y (numpy.ndarray): An array of shape (n_samples,) representing the labels for the input data.
        a (numpy.ndarray): An array of shape (n_samples,) representing the values of the Lagrange multipliers.
        kernel (callable): A function that takes two inputs X and Y and returns the kernel matrix of shape (n_samples, n_samples).

    Returns:
        The value of the objective function for the given inputs.
    """
    # TODO: implement
    
    # Reshape a and y to be column vectors
    a = a.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Compute the value of the objective function
    # The first term is the sum of all Lagrange multipliers
    term_1 = np.sum(a)

    # The second term involves the kernel matrix, the labels and the Lagrange multipliers
    kernel_matrix = kernel(X, X)
    term_2 = np.sum(a * y * kernel_matrix * y.T * a.T)

    return term_1 - 0.5 * term_2


class SVM(object):
    """
         Linear Support Vector Machine (SVM) classifier.

         Parameters
         ----------
         C : float, optional (default=1.0)
             Penalty parameter C of the error term.
         max_iter : int, optional (default=1000)
             Maximum number of iterations for the solver.

         Attributes
         ----------
         w : ndarray of shape (n_features,)
             Coefficient vector.
         b : float
             Intercept term.

         Methods
         -------
         fit(X, y)
             Fit the SVM model according to the given training data.

         predict(X)
             Perform classification on samples in X.

         outputs(X)
             Return the SVM outputs for samples in X.

         score(X, y)
             Return the mean accuracy on the given test data and labels.
         """

    def __init__(self, kernel=nonlinear_kernel, C=1.0, max_iter=1e3):
        """
        Initialize SVM

        Parameters
        ----------
        kernel : callable
          Specifies the kernel type to be used in the algorithm. If none is given,
          ‘rbf’ will be used. If a callable is given it is used to pre-compute 
          the kernel matrix from data matrices; that matrix should be an array 
          of shape (n_samples, n_samples).
        C : float, default=1.0
          Regularization parameter. The strength of the regularization is inversely
          proportional to C. Must be strictly positive. The penalty is a squared l2
          penalty.
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.a = None
        self.w = None
        self.b = None
        # self.X_train = None
        # self.y_train = None

    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
          Training vectors, where n_samples is the number of samples and n_features 
          is the number of features. For kernel=”precomputed”, the expected shape 
          of X is (n_samples, n_samples).

        y : array-like of shape (n_samples,)
          Target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
          Fitted estimator.
        """
        self.X = X
        self.y = y
        # save alpha parameters, weights, and bias weight

        # Save the input data
        # self.X_train = X
        # self.y_train = y
        
        # TODO: Define the constraints for the optimization problem
        
        # constraints = ({'type': 'ineq', 'fun': ...},
        #                {'type': 'eq', 'fun': ...})

        # Set up the optimization problem
        n_samples, n_features = X.shape
        constraints = ({'type': 'ineq', 'fun': lambda a: a},
                       {'type': 'eq', 'fun': lambda a: np.dot(a, y)})
        bounds = [(0, self.C) for _ in range(n_samples)]
        # Initialize the Lagrange multipliers with zeros
        initial_a = np.zeros(n_samples)
        
        # TODO: Use minimize from scipy.optimize to find the optimal Lagrange multipliers
        
        # res = minimize(...)
        # self.a = ...

        # Define the objective function for the optimization problem
        def obj_func(a):
            return -objective_function(X, y, a, self.kernel)
        # Use minimize to find the optimal Lagrange multipliers and store the optimal Lagrange multipliers
        res = minimize(obj_func, x0=initial_a, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': self.max_iter})
        self.a = np.array(res.x)
        
        # TODO: Substitute into dual problem to find weights
        
        # self.w = ...
        
        # TODO: Substitute into a support vector to find bias
        
        # self.b = ...

        # # Compute the weight vector w
        # if self.kernel == linear_kernel:
        #     self.w = np.dot(X.T, self.a * y)
        # else:
        #     self.w = None

        # # Compute the bias term b
        # support_vectors = self.a > 1e-5
        # if np.sum(support_vectors) == 0:
        #     self.b = 0
        # else:
        #     if self.kernel == linear_kernel:
        #         self.b = y[support_vectors[0]] - np.dot(X[support_vectors[0]], self.w)
        #     else:
        #         self.b = y[support_vectors[0]] - np.sum(self.a * y * self.kernel(X[support_vectors[0]], X))


        # Find the support vectors
        sv = self.a > 1e-8

        # Compute the weights and bias term
        if np.sum(sv) > 0:
            self.w = np.sum(self.a[sv, None] * y[sv, None] * X[sv], axis=0)
            self.b = np.mean(y[sv] - np.dot(self.a * y, self.kernel(X, X[sv])))
            # self.b = y[sv[0]] - np.sum(self.a[sv, None] * y[sv, None] * self.kernel(X[sv], X[sv]), axis=0)[0]
        else:
            self.w = None
            self.b = None


        # # Substitute into dual problem to find weights
        # if self.kernel == linear_kernel:
        #     self.w = np.sum(self.a * y[:, np.newaxis] * X, axis=0)
        # else:
        #     self.w = None

        # # Substitute into a support vector to find bias
        # support_indices = np.where(self.a > 1e-5)[0]
        # if self.w is not None:
        #     self.b = np.mean(y[support_indices] - np.dot(X[support_indices], self.w))
        # else:
        #     K = np.array([self.kernel(X[i], X[support_indices]) for i in support_indices])
        #     self.b = np.mean(y[support_indices] - np.sum(K * (self.a[support_indices] * y[support_indices]), axis=1))

        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        # TODO: implement
        # Calculate kernel matrix between training and test data
        K = self.kernel(X, self.X)

        # Calculate output for each test sample using the learned support vectors and their alpha values
        outputs = np.dot(K, (self.a * self.y).reshape(-1, 1)) + self.b
        # outputs = np.dot(self.a * self.y, K) + self.b
        y_pred = np.where(outputs >= 0, 1, -1) # y_pred = np.sign(outputs)
        return y_pred.flatten()

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels. 

        In multi-label classification, this is the subset accuracy which is a harsh 
        metric since you require for each sample that each label set be correctly 
        predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          True labels for X.

        Return
        ------
        score : float
          Mean accuracy of self.predict(X)
        """
        # TODO: implement
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
