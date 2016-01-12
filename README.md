# Neural Network OCR (MNIST Classifier)

## Overview

A multi-layer neural network framework which utilises the standard Stochastic Gradient Descent algorithm via Backpropagation to optimise the learning of the network.

The network allows for hyperparameter customisation via a JSON file. This is limited to keeping only 1 hidden layer however and is optimised for the MNIST handwriting dataset, so has 784 input neurons and 10 output neurons.

## Hidden Layers N-Comparisons

To compare the effect of the number of hidden layers on the accuracy curve of the network, I ran several iterations of the network, with the hyperparameters set at: "learning_rate": 3.0, "num_epochs": 50, "mini_batch_size": 10 and on each iteration changing the hidden neurons number by 10, starting at 10 neurons.

The graph below details the final epoch accuracy vs. number of hidden neurons. And the subsequent graphs below show the accuracy rates per epoch for a given N of hidden neurons.

###### Final Epoch Accuracy vs. Number of Hidden Neurons
![Final epoch accuracy vs. Number of hidden neurons](https://raw.githubusercontent.com/jaungiers/Neural-Network-OCR/master/outputs/epochs-vs-n-hidden.png)

###### Accuracy Performance over 50-epochs with 10 Hidden Neurons
![Accuracy Performance over 50-epochs with 10 Hidden Neurons](https://raw.githubusercontent.com/jaungiers/Neural-Network-OCR/master/outputs/10-hidden.png)

###### Accuracy Performance over 50-epochs with 20 Hidden Neurons
![Accuracy Performance over 50-epochs with 20 Hidden Neurons](https://raw.githubusercontent.com/jaungiers/Neural-Network-OCR/master/outputs/20-hidden.png)

###### Accuracy Performance over 50-epochs with 30 Hidden Neurons
![Accuracy Performance over 50-epochs with 30 Hidden Neurons](https://raw.githubusercontent.com/jaungiers/Neural-Network-OCR/master/outputs/30-hidden.png)

###### Accuracy Performance over 50-epochs with 40 Hidden Neurons
![Accuracy Performance over 50-epochs with 40 Hidden Neurons](https://raw.githubusercontent.com/jaungiers/Neural-Network-OCR/master/outputs/40-hidden.png)

###### Accuracy Performance over 50-epochs with 50 Hidden Neurons
![Accuracy Performance over 50-epochs with 50 Hidden Neurons](https://raw.githubusercontent.com/jaungiers/Neural-Network-OCR/master/outputs/50-hidden.png)

###### Accuracy Performance over 50-epochs with 60 Hidden Neurons
![Accuracy Performance over 50-epochs with 60 Hidden Neurons](https://raw.githubusercontent.com/jaungiers/Neural-Network-OCR/master/outputs/60-hidden.png)

Note: As can be seen from the varying degrees of final epoch accuracy, some networks with higher hidden neurons do not perform as accurately as others with a lower amount of hidden neurons. This can be explained by the random initialisations of weights and biases. If a network gets unlucky with a random initialisation then it will take far longer to train. Unfortunately due to processor and time limitations I did not have the capacity to run the network over a large amount of epochs, however had I done so, I would expect to see the final accuracy of higher numbers of hidden neurons improve and plateau in a logarithmic fashion.