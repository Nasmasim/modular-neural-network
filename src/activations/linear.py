from layers.layer import Layer
from layers.xavier_initialisation import xavier_init
import numpy as np

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
