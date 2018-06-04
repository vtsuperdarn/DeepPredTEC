# ST-ResNet in Tensorflow

A TensorFlow implementation of a deep learning based model, called Spatio-Temporal Residual Netwotk (ST-ResNet). It is an efficient predictive model that is exclusively built upon convolutions and residual links which are based on unique properties of spatio-temporal GPS TEC Maps data. More specifically, the residual neural network framework is used model the temporal closeness, period, and trend properties
of the TEC Maps. These properties along with exogeneous variables like AU, AL, SymH and ASymH indices are used to predict the future TEC Maps.

## Prerequisites

* Python 2.7
* Tensorflow 1.8
* NumPy 1.14.2

## Usage

To create the TensorFlow computation graph of the ST-ResNet architecture run:

    $ python main.py

## Code Organization

The model is coded by following OOP paradigm. The complex model architecture parts are abstracted through extensive use of functions which brings in more flexibility and helps in coding Tensorflow functionality like sharing of tensors. 

File structure:
* `main.py`: This file contains the main program. The computation graph for ST-ResNet is built and launched in a session.
* `params.py`: This file contains class Params for hyperparameter declarations.
* `modules.py`: This file contain helper functions and custom neural layers. The functions help in abstracting the complexity of the architecture and Tensorflow features. These functions are being called in the st_resnet.py for defining the computational graph.
* `qanet.py`: This file defines the Tensorflow computation graph for the ST-ResNet (Deep Spatio-temporal Residual Networks) architecture. The skeleton of the architecture from inputs to outputs in defined here using calls to functions defined in modules.py. Modularity ensures that the functioning of a component can be easily modified in modules.py without changing the skeleton of the ST-ResNet architecture defined in this file.

## References

- [Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/pdf/1610.00081.pdf)
