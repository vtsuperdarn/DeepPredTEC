'''
Author: Sneha Singhania
Date: June 3, 2018
Comment: This file defines the Tensorflow computation graph for the ST-ResNet (Deep Spatio-temporal Residual Networks) architecture. The skeleton of the architecture from inputs to outputs in defined here using calls to functions defined in modules.py. Modularity ensures that the functioning of a component can be easily modified in modules.py without changing the skeleton of the ST-ResNet architecture defined in this file.
'''

from params import Params as param
import modules as my
import tensorflow as tf
import numpy as np

class Graph(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            B, H, W, C, P, T, O, F, U, V, L, S, N  = param.batch_size, param.map_height, param.map_width, param.closeness_sequence_length, param.period_sequence_length, param.trend_sequence_length, param.num_of_output_tec_maps ,param.num_of_filters, param.num_of_residual_units, param.exo_values, param.look_back, param.lstm_size, param.num_layers 
            
            #get inputs and outputs            
            #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
            self.c_tec = tf.placeholder(tf.float32, shape=[B, H, W, C], name="closeness_tec_maps")
            self.p_tec = tf.placeholder(tf.float32, shape=[B, H, W, P], name="period_tec_maps")
            self.t_tec = tf.placeholder(tf.float32, shape=[B, H, W, T], name="trend_tec_maps")
            self.output_tec = tf.placeholder(tf.float32, shape=[B, H, W, O], name="output_tec_map") 
            
            self.exogenous = tf.placeholder(tf.float32, shape=[B, L, V], name="exogenous")
            
            
            #processing with exogenous variables
            #this will be of shape (batch_size, lstm_size)
            self.external = my.exogenous_module(self.exogenous, S, N)
            #shape (batch_size, 1, lstm_size)
            self.external = tf.expand_dims(self.external, 1)
            print self.external.shape
            
            #ResNet architecture for the three modules
            #module 1: Capturing the closeness(recent)
            self.closeness_output = my.ResInput(inputs=self.c_tec, filters=F, kernel_size=(7, 7), scope="closeness_input", reuse=None)
            self.closeness_output = my.ResNet(inputs=self.closeness_output, filters=F, kernel_size=(7, 7), repeats=U, scope="resnet", reuse=None)
            self.closeness_output = my.ResOutput(inputs=self.closeness_output, filters=1, kernel_size=(7, 7), scope="resnet_output", reuse=None)
            
            #module 2: Capturing the period(near)
            self.period_output = my.ResInput(inputs=self.p_tec, filters=F, kernel_size=(7, 7), scope="period_input", reuse=None)
            self.period_output = my.ResNet(inputs=self.period_output, filters=F, kernel_size=(7, 7), repeats=U, scope="resnet", reuse=True)
            self.period_output = my.ResOutput(inputs=self.period_output, filters=1, kernel_size=(7, 7), scope="resnet_output", reuse=True)
            
            #module 3: Capturing the trend(distant) 
            self.trend_output = my.ResInput(inputs=self.t_tec, filters=F, kernel_size=(7, 7), scope="trend_input", reuse=None)
            self.trend_output = my.ResNet(inputs=self.trend_output, filters=F, kernel_size=(7, 7), repeats=U, scope="resnet", reuse=True)
            self.trend_output = my.ResOutput(inputs=self.trend_output, filters=1, kernel_size=(7, 7), scope="resnet_output", reuse=True)
            
            #TODO: combining the exogenous and each module output
            #populating the exogenous variable
            self.val = tf.tile(self.external, [1, H*W, 1])
            self.exo = tf.reshape(self.val, [B, H, W, S])
            
            #concatenate the modules output with the exogenous module output
            self.close_concat = tf.concat([self.exo, self.closeness_output], 3, name="close_concat")
            self.period_concat = tf.concat([self.exo, self.period_output], 3, name="period_concat")
            self.trend_concat = tf.concat([self.exo, self.trend_output], 3, name="trend_concat")
            
            #last convolutional layer for getting information from exo and each of the modules
            self.exo_close = tf.layers.conv2d(self.close_concat, 1, kernel_size=(7, 7), strides=(1,1), padding="SAME", name="exo_close") 
            self.exo_period = tf.layers.conv2d(self.period_concat, 1, kernel_size=(7, 7), strides=(1,1), padding="SAME", name="exo_period") 
            self.exo_trend = tf.layers.conv2d(self.trend_concat, 1, kernel_size=(7, 7), strides=(1,1), padding="SAME", name="exo_trend") 
            
            
            # parameter-matrix-based fusion of the outputs after combining with exo
            self.x_res = my.Fusion(self.exo_close, self.exo_period, self.exo_trend, scope="fusion", shape=[W, W])
            
            
            #here we calculate the total sum and then divide - the inbuilt function will handle overflow
            self.loss = tf.reduce_sum(tf.pow(self.x_res - self.output_tec, 2)) / tf.cast((self.x_res.shape[0]), tf.float32)
            
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr, beta1=param.beta1, beta2=param.beta2, epsilon=param.epsilon).minimize(self.loss)
            
            #loss summary
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()
            
            self.saver = tf.train.Saver(max_to_keep=None)
