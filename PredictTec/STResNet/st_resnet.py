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
            B, H, W, C, P, T, O, F, U  = param.batch_size, param.map_height, param.map_width, param.closeness_sequence_length, param.period_sequence_length, param.trend_sequence_length, param.num_of_output_tec_maps ,param.num_of_filters, param.num_of_residual_units
            
            #get inputs and outputs
            #self.c_tec, self.p_tec, self.t_tec, self.output_tec = my.get_batch_data()
            
            #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
            self.c_tec = tf.placeholder(tf.float32, shape=[B, H, W, C], name="closeness_tec_maps")
            self.p_tec = tf.placeholder(tf.float32, shape=[B, H, W, P], name="period_tec_maps")
            self.t_tec = tf.placeholder(tf.float32, shape=[B, H, W, T], name="trend_tec_maps")
            self.output_tec = tf.placeholder(tf.float32, shape=[B, H, W, O], name="output_tec_map") 
            
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
            self.trend_output = my.ResInput(inputs=self.t_tec, filters=F, kernel_size=(7, 7), scope="trend_output", reuse=None)
            self.trend_output = my.ResNet(inputs=self.trend_output, filters=F, kernel_size=(7, 7), repeats=U, scope="resnet", reuse=True)
            self.trend_output = my.ResOutput(inputs=self.trend_output, filters=1, kernel_size=(7, 7), scope="resnet_output", reuse=True)
            
            # parameter-matrix-based fusion
            self.x_res = my.Fusion(self.closeness_output, self.period_output, self.trend_output, scope="fusion", shape=[W, W])
            
            #combining with exogenous variables
            #TODO: get the exogenous variables values and add with the proposed algorithm
            
            self.loss = tf.reduce_mean(tf.squared_difference(self.x_res, self.output_tec))
            
            #training scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            
            #using ADAM optimizer with beta1=0.8, beta2=0.999 and epsilon=1e-7
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr, beta1=param.beta1, beta2=param.beta2, epsilon=param.epsilon)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            
            #loss summary
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()
            
