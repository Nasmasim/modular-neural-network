from layers.layer import Layer 
import numpy as np

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