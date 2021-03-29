# Modular Neural Network Library from Scratch
## Description
This project presents a low-level implementation of a multi-layered neural network, including a basic implementation of the backpropagation algorithm using only ```NumPy```. Included are also funtions for data preprocessing, training and evaluation. Performance comparable to ```PyTorch``` implementations. 

## Project structure
The mini-library implements:
| Custom Layers          | Function          | 
| -------------          |:-------------:| 
| [activations.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/layers/activations.py)| Batch Normalization over a mini-batch of 2D inputs |
| [losses.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/layers/losses.py)      | Max-pooling layer      |
| [multilayernetwork.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/multilayernetwork.py) | Convolutional layer      |
| [linear.py](https://github.com/Nasmasim/modular-CNNs/blob/main/custom_cnn_layers/linear.py) | Linear layer |
| Training          | Function          | 
| -------------          |:-------------:| 
|[trainer.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/trainer.py) | |


* [activations.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/layers/activations.py): 
  * Sigmoid Layer
    * applies sigmoid function elementwise
  * ReLu Layer
    * applies ReLU elementwise
  * Linear Layer
    * performs affine transformation **XW + B** on a batch on inputs **X**
    * Xavier Glorot weight initialization
* [losses.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/layers/losses.py): 
  * MSE Loss Layer
    * computes mean-squared error between y_pred and y_target
  * Cross Entropy Loss Layer
    * computes the softmax followed by the negative log-likelihood loss
* [multilayernetwork.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/multilayernetwork.py):
    * modular stacked linear layers with activation function
* [trainer.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/trainer.py):
    * Trainer class which handles data shuffling and training given a network
    * using minibatch gradient descent as well as computing the loss on a validation dataset
* [preprocessor.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/preprocessor.py):
    * Preprocessor class which performs data normalization (min-max scaling)
* [main.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/main.py):
    * example of an implementation of the mini library on the Iris Dataset
