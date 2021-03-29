# Modular Neural Network Library from Scratch
## Description
This project presents a low-level implementation of a multi-layered neural network, including a basic implementation of the backpropagation algorithm using only ```NumPy```. Included are also funtions for data preprocessing, training and evaluation. Performance comparable to ```PyTorch``` implementations. 

## Project structure
The mini-library implements:
| Custom Layers          | Function      | Comment     |
| -------------          |-------------  |-------------| 
| [activations.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/layers/activations.py)| SigmoidLayer | applies sigmoid function elementwise |
| | ReluLayer | applies ReLU elementwise |
| | LinearLayer | performs affine transformation on a batch of inputs with Xavier Glorot weight initialization |
| [losses.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/layers/losses.py)      | MSELossLayer | computes mean-squared error between y_pred and y_target |
| | CrossEntropyLossLayer | computes the softmax followed by the negative log-likelihood loss |
| [multilayernetwork.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/multilayernetwork.py) | MultiLayerNetwork | modular stacked linear layers with activation function |


For training:
| Function          | Comment       | 
| -------------     |-------------  |
|[trainer.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/trainer.py) | Trainer class handles data shuffling and training given a network, using minibatch gradient descent |
| [preprocessor.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/preprocessor.py) | Preprocessor class performs data normalization (min-max scaling) |
| [main.py](https://github.com/Nasmasim/modular-neural-network-mini-Library/blob/main/main.py) | Example of an implementation of the mini library on the Iris Dataset|
