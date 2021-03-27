##############################################################################
#                                                                            #
#           3 Activations: Sigmoid, ReLu, Linear                             #
#                                                                            #
##############################################################################

import numpy as np

class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass
def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)

class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        #store the input in the cache to later use in backward pass
        self._cache_current = x
        #return x passed through the sigmoid function
        return 1/(1+np.exp(-x))


    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """

        #retrieve input from the cache
        x = self._cache_current
        #compute sigmoid derivative of output of previous linear layer (input of activation function)
        sigmoid_derivative = np.exp(-x)/(1+np.exp(-x))**2
        
        #check whether same shape
        assert grad_z.shape == sigmoid_derivative.shape
        
        #do element wise multiplication of gradient wrt to output of activation function and sigmoid derivative 
        return np.multiply(grad_z, sigmoid_derivative)


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        #store the input in the cache to later use in backward pass
        self._cache_current = x
        #return x passed through the relu function
        return np.maximum(0,x)



    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """

        #retrieve input from the cache
        x = self._cache_current
        #compute relu derivative of output of previous linear layer (input of activation function)
        relu_derivative = (x > 0) * 1
        
        #check whether same shape
        assert grad_z.shape == relu_derivative.shape
        
        #do element wise multiplication of gradient wrt to output of activation function and relu derivative 
        return np.multiply(grad_z, relu_derivative)

class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out


        # initialize weights of linear layer with Xavier Glorot initialization (gain of 1)
        self._W = xavier_init((self.n_in, self.n_out), 1.0)
        # initialize biases with zeros, because we don’t want the neurons to start out with a bias
        self._b = np.zeros((1,self.n_out))

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None



    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        #retrieve input from the cache
        self._cache_current = x
        
        #check whether same shape
        assert (x.shape[1] == self._W.shape[0])
        
        # calculate output of linear layer and return
        z = x @ self._W + np.repeat(self._b,x.shape[0],axis=0)
        
        return z


    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """

        # check whether same shape and multiplications can be performed
        
        assert (np.transpose(self._cache_current).shape[1] == grad_z.shape[0]) & (np.ones((1,self._cache_current.shape[0])).shape[1] == grad_z.shape[0]) & (grad_z.shape[1] == np.transpose(self._W).shape[0])
        
        # calculate loss gradient wrt to weights: input_transpose * grad_z
        self._grad_W_current = np.transpose(self._cache_current) @ grad_z
        # calculate loss gradient wrt to bias: column vector of ones * grad_z
        self._grad_b_current = np.ones((1,self._cache_current.shape[0])) @ grad_z
        
        #return gradient with respect to the inputs of the layer: grad_z * w_transpose
        return grad_z @ np.transpose(self._W)


    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        #update weights and biases 
        self._W += - learning_rate * self._grad_W_current
        self._b += - learning_rate * self._grad_b_current













