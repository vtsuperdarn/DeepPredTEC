# ST-ResNet in Tensorflow

A TensorFlow implementation of a deep learning based model, called Spatio-Temporal Residual Netwotk (ST-ResNet). It is an efficient predictive model that is exclusively built upon convolutions and residual links which are based on unique properties of spatio-temporal GPS TEC Maps data. More specifically, the residual neural network framework is used model the temporal closeness, period, and trend properties
of the TEC Maps. These properties along with exogeneous variables including the OMNI IMF data (By, Bz, Vx, Np) are used to predict the future TEC Maps.

## Model architecture

<p align="center"> 
<img src="assets/st-resnet.png">
</p>

## Requirements

* Python 3.6
* Tensorflow 1.8
* NumPy 1.14.3
* Dask 0.18.2
* Pandas 0.22.0
* Tqdm 4.19.9
* Matplotlib 2.2.2
* Seaborn 0.8.1

## Usage

To create the TensorFlow computation graph of the ST-ResNet architecture and train the model run:

    $ python main.py

To get the predicted TEC maps on the saved trained model run:
    
    $ python get_prediction.py
    
## Code Organization

The model is coded by following OOP paradigm. The complex model architecture parts are abstracted through extensive use of functions which brings in more flexibility and helps in coding Tensorflow functionality like sharing of tensors. 

File structure:
* `params.py`: This file contains class Params for hyperparameter declarations.
* `modules.py`: This file contain helper functions and custom neural layers. The functions help in abstracting the complexity of the architecture and Tensorflow features. These functions are being called in the st_resnet.py for defining the computational graph.
* `st_resnet.py`: This file defines the Tensorflow computation graph for the ST-ResNet (Deep Spatio-temporal Residual Networks) architecture. The skeleton of the architecture from inputs to outputs in defined here using calls to functions defined in modules.py. Modularity ensures that the functioning of a component can be easily modified in modules.py without changing the skeleton of the ST-ResNet architecture defined in this file.
* `batch_utils.py`: This file contains two major classes called `BatchDateUtils` and `TECUtils`. The `BatchDateUtils` object is initialized in the `main.py` of ST-ResNet model for creating a dictionary of datetime variables. The key in the dictionary is the current date & time from which the past TEC maps have to be used for data point creation and corresponding value in the dictionary is a numpy array of input datetime variables for the three channels (closeness, period and trend) of the model and the output TEC maps used as the true label during model training. When the Tensorflow session gets initialized for training and testing the model in `main.py` the above mentioned dictionary is used for loading the tensor of actual 2D numpy TEC maps with the help of `TECUtils` class.     
* `omn_utils.py`: This file contains the `OmnData` class used for loading the OMNI IMF data originally stored in a sqlite3 database and creating the sequential IMF data array for (By, Bz, Vx, Np) based on a look back window. This look back window is a hyperparameter of the model and can be tuned later. The missing IMF values are filled using forward filling algorithm of pandas library. 
* `main.py`: This file contains the code for training and validating the model. The computation graph for ST-ResNet is built and launched in a session. 
* `get_prediction.py`: This file contains the code for getting the predicted TEC maps using the saved trained model. We need to set the appropriate values for 'saved_model_path' and 'saved_model' variables in params.py for this code to run correctly.
* `plotting_results.py`: This file contains the code for getting the predicted and channel-wise plots.
* `plotting_loss_curves.py`: This file contains the code for getting the loss vs epoch curve to understand the mode's performance.

## References

- [Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/pdf/1610.00081.pdf)
