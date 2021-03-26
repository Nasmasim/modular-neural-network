from layers.layer import Layer
import numpy as np

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