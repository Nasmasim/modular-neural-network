##############################################################################
#                                                                            #
#   Trainer: Object that manages the training of a neural network.           #
#                                                                            #
##############################################################################
from layers.losses import CrossEntropyLossLayer, MSELossLayer
import numpy as np

class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        if loss_fun == 'cross_entropy':
            self._loss_layer = CrossEntropyLossLayer()
        elif loss_fun == 'mse':
            self._loss_layer = MSELossLayer()
        #self._loss_layer = CrossEntropyLossLayer() if loss_fun == 'cross_entropy' else self._loss_layer = MSELossLayer()

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """

        #randomly shuffle indices and use shuffled indices to get shuffled dataset (both input and target)
        idx_list = np.arange(len(input_dataset))
        np.random.shuffle(idx_list)
        shuffled_inputs  = input_dataset[idx_list]
        shuffled_targets  = target_dataset[idx_list]
        return shuffled_inputs, shuffled_targets

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """

        #loop over number of epochs
        for i in range(self.nb_epoch):
            #check whether data should be shuffled
            if self.shuffle_flag == True:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
            #Splits the dataset into batches of size batch_size
            input_batches = [input_dataset[i:i + self.batch_size] for i in range(0, len(input_dataset), self.batch_size)]  
            target_batches = [target_dataset[i:i + self.batch_size] for i in range(0, len(target_dataset), self.batch_size)] 
            
            #loop over batches
            for i in range(len(input_batches)):
                y_pred = self.network(input_batches[i])
                self._loss_layer.forward(y_pred, target_batches[i])
                grad_loss = self._loss_layer.backward()
                self.network.backward(grad_loss)
                self.network.update_params(self.learning_rate)

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        """

        #get current target prediction and return calculated loss (with target data)
        y_pred = self.network(input_dataset)
        return self._loss_layer.forward(y_pred, target_dataset)
