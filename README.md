# Modular Neural Network Library from Scratch
## Description:
This project presents a low-level implementation of a multi-layered neural network, including a basic implementation of the backpropagation algorithm using only ```NumPy```. This includes funtions for data preprocessing, training and evaluation. 

## Project structure
The mini-library implements in the **layers** folder:
* under **activations.py**: 
  * a Sigmoid Layer
    * applies sigmoid function elementwise
  * a ReLu Layer
    * applies sigmoid function elementwise
  * a Linear Layer
    * performs affine transformation $XW + B$ on a batch on inputs $X$, 
    * Xavier Glorot weight initialization
* under **losses.py**: 
  * an MSE Loss Layer
    * computes mean-squared error between y_pred and y_target
  * a Cross Entropy Loss Layer
    * computes the softmax followed by the negative log-likelihood loss
Under **multilayernetwork.py**:
* modular stacked linear layers with activation function

Under **trainer.py**:
* Trainer class which handles data shuffling and training given a network, using minibatch gradient descent as well as computing the loss on a validation dataset

Under **preprocessor.py**:
* Preprocessor class which performs data normalization (min-max scaling)

Under **main.py**:
* Training using the mini library on the Iris Dataset. Results are comparable to ```PyTorch``` implementaions. 
