# Deep Learning on GPS TEC Maps
Global Positioning System obtained Total Electron Content (GPS TEC) Map is an important descriptive quantity of the ionosphere for analysis of space weather. Building an accurate predictive model for TEC maps can help in anticipating adverse ionospheric effects (ex: due to a solar storm), thereby safeguarding critical communication, energy and navigation infrastructure. In this work, we employ a deep learning approach to predict TEC maps using deep Spatio-Temporal Residual Networks (ST-ResNets), the first attempted work of its kind, by focusing on the North American region. To obtain a contextual awareness of other space weather indicators during prediction, we also use exogenous data including OMNI IMF data (By, Bz, Vx, Np). Our model predicts TEC maps for the next hour and beyond, at a time resolution of five minutes, providing state-of-the-art prediction accuracy. Towards explainable Artificial Intelligence (AI), especially in deep learning models, we provide extensive visualizations of the outputs from our convolutional networks in all three branches (closeness, period and trend) for better analysis and insight. In the future, we aim to demonstrate the effectiveness and robustness of our model in predicting extremely rare solar storms, a challenging task given the insufficient training data for such events.

# Google Summer of Code 2018
* [Project Link](https://summerofcode.withgoogle.com/projects/#6628265427468288)
* [Project Website](https://vtsuperdarn.github.io/DeepPredTEC/)

# License 
See the [LICENSE](LICENSE.md) file for license rights and limitations. 
 
