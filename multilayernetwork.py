##############################################################################
#                                                                            #
#   MultiLayerNetwork: Network consisting of stacked linear layers and       #
#   activation functions.                                                    #
#                                                                            #
##############################################################################
import numpy as np
from layers.activations import LinearLayer, SigmoidLayer, ReluLayer

class MultiLayerNetwork(object):

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #stack all layers in one ndarray 
        layers = np.ndarray((len(self.neurons)*2),dtype=np.object)
        n_in = self.input_dim
        for i in range(len(self.neurons)):
            layers[2*i] = LinearLayer(n_in,self.neurons[i])
            n_in = self.neurons[i]
            if (self.activations[i] == 'relu'):
                layers[2*i+1] = ReluLayer()
            elif (self.activations[i] == 'sigmoid'):
                layers[2*i+1] = SigmoidLayer()
            else:
                layers[2*i+1] = 'identity'
            
        self._layers = layers              

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """

        #pass through all layers
        z_temp = x
        for i in range(len(self._layers)):
            #check whether identity layer and no activation function should be called
            if not self._layers[i] == 'identity':
                z_temp = self._layers[i].forward(z_temp)
        return z_temp 

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """

        #backwards pass through all layers
        grad_z_temp = grad_z
        for i in range(len(self._layers)-1,-1,-1):
            #check whether identity layer and no activation function should be called
            if not self._layers[i] == 'identity':
                grad_z_temp = self._layers[i].backward(grad_z_temp)
        return grad_z_temp


    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        #update all weights and biases in the individual layers
        for i in range(len(self._layers)):
            #check that only linear layers are called
            if isinstance(self._layers[i], LinearLayer):
                self._layers[i].update_params(learning_rate)