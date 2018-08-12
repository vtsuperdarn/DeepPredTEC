---
layout: post
title: GSOC 2018 Project Part 2
date: 2018-07-10
---

## Title
Deep Predictive Models for Predicting GPS TEC Maps

## Outline

The major focus in the second phase of the project is on getting a better understanding of the model through plotting of results and output. The output is also analysed before and after integration of IMF data through exogenous module implemented using a Long-Short Term Memory (LSTM) network. The models scalability is analysed by predicting and plotting the TEC maps for the next one hour at a time resolution of five minutes.     

## Results and plots of output tec maps
The model was completely implemented in Tensorflow. Without considersing the exogenous module, the model is trained on a fraction of the dataset and the accuracy are reported in a qualitative manner. The predicted maps are plotted and the results are analysed by looking at the absolute and relative difference at each location between the true TEC and predicted TEC maps. The sample predicted plot along with the difference plots is given below.

![Sample predicted plot]({{site.url}}/assets/difference.png)   
 
## Channel wise TEC map plots

Not only the final output, but the output from each of the ResNet channels, closeness, period and trend, are also plotted for better understanding of the patterns that are being learnt by the model. This also helps in evaluating the performance of each of the channels and will further help in model parameter tuning. The sample channel wise output plots for the predicted TEC map is given below.  

![Sample channel wise plot]({{site.url}}/assets/channel.png)

## IMF data preprocessing
From the list of exogenous variables, only the IMF data including Bz (OMNI IMF z component), By (OMNI IMF y component), Vx which is the plasma bulk velocity in km/s along x-axis and Np which is the proton number density is taken as in input to the exogenous module. The IMF data is available in a sqlite3 database. Using the forward filling mechanism the missing rows corresponding to the datetime variable are filled and updated in the original database.

## Model: Encoder LSTM
The IMF data with preprocessing is given as an input to the exogenous module which is based on an LSTM architecture. LSTM is a variant of recurrent neural network (RNN) which is very useful for processing sequential data. IMF data in a loaded with a look back window which is initialized as one of the paramters of the model. LSTM takes the (Bz, By, Vx, Np) vectors in a sequential manner based on the datetime associated with it and outputs an abstract complex representation by considering the temporal dependencies present in the data. 

## Improved Results with IMF data
The model is trained by considering the ResNet modules and the exogenous module. The output of the predicted TEC map is compared with the integration of the IMF data. Results show that IMF data helps in more accurate and smooth prediction at majority of the locations. The sample predicted and difference plot after integration with IMF data is given below.

![Sample predicted plot]({{site.url}}/assets/exodifference.png) 

The channel wise output plots with the IMF data integration helps us to understand the change in behaviour of the module. It also provides information on the relationship between IMF data and TEC maps. The sample channel wise output plots with IMF data integration is given below.

![Sample channel wise plot]({{site.url}}/assets/exochannel.png) 

## Block prediction (next 12)
 
Instead of predicting the next TEC map as the only output, the model tries solve the harder task of predicting the next one hour TEC maps at a time resolution of five minutes given the past TEC maps as the input. This helps in testing and tuning the model's scability. On a minimally trained model the accuracy results are reported. The sample block (next 12) prediction TEC maps are given below. ADD THE IMAGEEEEEEEE .........
 

## Presentation Slides
ADD ......................

## Future Steps
ADD FUTURE STEPS ................

## People 
<ul>
<li>Sneha Singhania (sneha3295[at]gmail[dot]com)</li>
<li>Bharat Kunduri (bharatr[at]vt[dot]edu)</li>
<li>Muhammad Rafiq (rafiq[at]vt[dot]edu)</li>
</ul> 

## References
* [https://github.com/vtsuperdarn/DeepPredTEC](https://github.com/vtsuperdarn/DeepPredTEC)
* [https://www.haystack.mit.edu/atm/open/radar/index.html](https://www.haystack.mit.edu/atm/open/radar/index.html)
* [Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/pdf/1610.00081.pdf)
