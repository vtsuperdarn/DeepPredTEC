# ST-ResNet in Tensorflow

A TensorFlow implementation of a deep learning based model, called Spatio-Temporal Residual Netwotk (ST-ResNet). It is an efficient predictive model that is exclusively built upon convolutions and residual links which are based on unique properties of spatio-temporal GPS TEC Maps data. More specifically, the residual neural network framework is used model the temporal closeness, period, and trend properties
of the TEC Maps. These properties along with exogeneous variables like AU, AL, Sym and ASym indices are used to predict the future TEC Maps.

## Prerequisites

* Python 2.7
* Tensorflow 1.8
* NumPy 1.14.2

## Usage

To create the TensorFlow computation graph of the ST-ResNet architecture run:

    $ python main.py

## References

- [Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/pdf/1610.00081.pdf)
