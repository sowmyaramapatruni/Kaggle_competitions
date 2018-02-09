import numpy as np
from scipy.special import expit
import math
class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class Majority(Model):

    def __init__(self):
        super().__init__()
        self.max_label = None
        
    def fit(self, X, y):
    
        #labels are already stripped and copied to array y
        #count the number of 1 and zeros in y
        self.num_input_features = X.shape[1]
        num_zeros = 0
        num_ones = 0
        
        #Count the number of zeros and ones in the train data
        for i in range(self.num_input_features):
            if y[i] == 0:
                num_zeros += 1
            else:
                num_ones += 1
                
        #set the max_label 
        self.max_label = 0 if (num_zeros > num_ones) else 1
        #if there is a tie, pick a number randomly between [1,0]
        if num_zeros == num_ones:
            self.max_label = np.random.choice(y)
            
            
    def predict(self, X):
        #if num_input_features < 0:
        #raise Exception("fit must be called before")
        num_examples, num_input_features = X.shape
        # predict the test data as the max_label
        y_hat = np.full([num_examples], self.max_label, dtype=np.int)
        return (y_hat)

# Perceptron Classification Model
class Perceptron(Model):

    def __init__(self, online_learning_rate, online_training_iterations):
        super().__init__()
        self.online_learning_rate = online_learning_rate
        self.online_training_iterations = online_training_iterations
        self.W = []
        
    def fit(self, X, y):
        #get the number of inputs and features
        num_examples, self.num_input_features = X.shape
        #create a matrix for holding the weights.Initialize the weights to zero
        self.W = np.zeros((1,X.shape[1]), dtype=np.float)
        #empty n-d array to hold the caluculated y_hat value.
        y_hat = np.empty(num_examples, dtype=float)
        
        #Online learning: Iterate over each input example and adjust the value
        #of W for each input based on error. 
        #Update the values of W which are available for prediction function.
        while (self.online_training_iterations >= 0):
            for i in range(num_examples):
                y_i = y[i]
                if(y_i == 0):
                    y_i = -1
                x_i = X[i, :]
                W_t = np.transpose(self.W)
                y_hat[i] = np.sign(x_i.dot(W_t))
                if (y_i != y_hat[i]):
                    #loss = (y_i - y_hat[i])
                    self.W = self.W + (self.online_learning_rate * y_i * x_i)
            self.online_training_iterations -= 1
            
    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        y_hat = np.empty([num_examples], dtype=np.int)
        #For each inout in test data, caluclate the y value based on weights.
        for i in range(num_examples):
            x_i = X[i, :]
            W_t = np.transpose(self.W)
            y_hat[i] = np.sign(x_i.dot(W_t))
            if y_hat[i] == -1:
                y_hat[i] = 0
        return (y_hat)

class Logistic(Model):

    #def __init__(self, optimizer, online_learning_rate, online_training_iterations):
    def __init__(self, optimizer, online_learning_rate, online_training_iterations):
        super().__init__()
        self.optimizer = optimizer
        self.online_learning_rate = online_learning_rate
        self.online_training_iterations = online_training_iterations
        self.W = []
    def sigmoid(self, w, x):
        #return logistic.cdf(x.dot(w))
        return expit(x.dot(w))

    def fit(self, X, y):
        #create a matrix for holding the weights.Initialize the weights to zero
        self.W = np.zeros((1, X.shape[1]), dtype=np.float)
        
        if self.optimizer == "sgd":
            self.sgd_fit(X, y)
        else:
            self.sgd_adam_fit(X, y)
        
        #Fitting: Maximum likliehood or minimize loss.
        # Use Stochastic gradient
        #caluculate change in w for each input
        #Update the values of W which are available for prediction function.
    def sgd_fit(self, X, y):
        #get the number of inputs and features
        num_examples, self.num_input_features = X.shape
        while (self.online_training_iterations >= 0):
            for i in range(num_examples):
                y_i = y[i]
                x_i = X[i, :]
                #x_i = X[i]
                w_t = np.transpose(self.W)
                #w_delta = ((1-y_i) * self.sigmoid(w_t,x_i) * x_i) - (y_i * self.sigmoid(-w_t, x_i) * x_i)
                w_delta = ((1-y_i) * expit(x_i.dot(w_t)) * x_i) - (y_i * expit(-x_i.dot(w_t)) * x_i)
                self.W = self.W - (self.online_learning_rate * w_delta)
            self.online_training_iterations -= 1
     
    def sgd_adam_fit(self, X, y):
        #get the number of inputs and features
        num_examples, self.num_input_features = X.shape
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10**(-8)
        m = np.zeros(self.W.shape, dtype=np.float64)
        v = np.zeros(self.W.shape, dtype=np.float64)
        
        #set step count to zero
        t = 0
        eta = self.online_learning_rate
        while (self.online_training_iterations >= 0):
            for i in range(num_examples):
                y_i = y[i]
                x_i = X[i, :]
                w_t = np.transpose(self.W)
                t = t + 1
                g = ((1-y_i) * self.sigmoid(w_t,x_i) * x_i) - (y_i * self.sigmoid(-w_t, x_i) * x_i)
                m = beta_1 * m + (1 - beta_1) * g
                m_hat = m / (1 - (beta_1)**t)
                
                v = beta_2 * v + (1 - beta_2) * (g)**2
                v_hat = v / (1 - (beta_2)**t)
                self.W = self.W - eta * (m_hat / ((v_hat)**(0.5) + epsilon))
            self.online_training_iterations -= 1
    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        num_examples, num_input_features = X.shape
        y_hat = np.empty([num_examples], dtype=np.int)
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        #For each input in test data, caluclate the y value based on weights.
        w_t = np.transpose(self.W)
        for i in range(num_examples):
            x_i = X[i, :]
            y_temp = self.sigmoid(w_t, x_i)
        
            if y_temp >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat
