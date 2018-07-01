'''
Author: Sneha Singhania
Date: June 3, 2018
Comment: This file contain helper functions and custom neural layers. The functions help in abstracting the complexity of the architecture and Tensorflow features. These functions are being called in the st_resnet.py for defining the computational graph
'''

import tensorflow as tf
import numpy as np
from params import Params as param


def lstm_cell(lstm_size):
  return tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
  

def exogenous_module(inputs, lstm_size, num_layers=2):
    
    #stacked lstm implementation
    lstm_cells1 = [tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, use_peepholes=True, activation=tf.nn.relu) for _ in range(num_layers)]
    stacked_lstm1 = tf.contrib.rnn.MultiRNNCell(lstm_cells1)
    annotations, state = tf.nn.dynamic_rnn(stacked_lstm1, inputs=inputs, dtype=tf.float32, time_major=False, scope="lstm")

    
    #single layer lstm
    #annotations, state = tf.nn.dynamic_rnn(lstm_cell(lstm_size), inputs=inputs, dtype=tf.float32, scope="lstm")
    
    annotations = tf.transpose(annotations, [1, 0, 2])
    #extracting the last annotation or hidden state
    #(Batch_size, lstm_size)
    output = tf.gather(annotations, int(annotations.get_shape()[0]) -1)      
    return output
        
        
def ResUnit(inputs, filters, kernel_size, strides, scope, reuse=None):   
    '''
    Defines a residual unit: input->[layernorm->relu->conv->layernorm->relu->conv]->reslink-> output
    '''
    with tf.variable_scope(scope, reuse=reuse): 
        #use layernorm before applying convolution
        outputs = tf.contrib.layers.layer_norm(inputs, scope="layernorm1", reuse=reuse)
        #relu activation
        outputs = tf.nn.relu(outputs, name="relu")
        #perform a 2D convolution
        outputs = tf.layers.conv2d(outputs, filters, kernel_size, strides, padding="SAME", name="conv1", reuse=reuse)  
        
        #use layernorm before applying convolution
        outputs = tf.contrib.layers.layer_norm(inputs, scope="layernorm2", reuse=reuse)
        #relu activation
        outputs = tf.nn.relu(outputs, name="relu")
        #perform a 2D convolution
        outputs = tf.layers.conv2d(outputs, filters, kernel_size, strides, padding="SAME", name="conv2", reuse=reuse)                
        
        #adding the res link        
        outputs += inputs
        return outputs    


def ResNet(inputs, filters, kernel_size, repeats, scope, reuse=None):
    '''
    Defines the ResNet architecture
    '''
    with tf.variable_scope(scope, reuse=reuse):
        #apply repeats number of residual layers
        for layer_id in range(repeats):
            inputs = ResUnit(inputs, filters, kernel_size, (1,1), "reslayer_{}".format(layer_id), reuse)
        outputs = tf.nn.relu(inputs, name="relu")
        return outputs

def ResInput(inputs, filters, kernel_size, scope, reuse=None):
    '''
    Defines the first (input) layer of the ResNet architecture
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1), padding="SAME", name="conv_input", reuse=reuse)
        return outputs


def ResOutput(inputs, filters, kernel_size, scope, reuse=None):
    '''
    Defines the last (output) layer of the ResNet architecture
    '''
    with tf.variable_scope(scope, reuse=reuse):
        #applying the final convolution to the tec map with depth 1 (num of filters=1)
        outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1), padding="SAME", name="conv_out", reuse=reuse)       
        return outputs
                  
        
def Fusion(closeness_output, period_output, trend_output, scope, shape):
    '''
    Combining the output from the module into one tec map
    '''            
    with tf.variable_scope(scope):
        closeness_output = tf.squeeze(closeness_output)
        period_output = tf.squeeze(period_output)
        trend_output = tf.squeeze(trend_output)
        
        Wc = tf.get_variable("closeness_matrix", dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        Wp = tf.get_variable("period_matrix", dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        Wt = tf.get_variable("trend_matrix", dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        
        #print closeness_output
        #print period_output
        #print trend_output
        
        output = tf.reshape(closeness_output, [closeness_output.shape[0]*closeness_output.shape[1], closeness_output.shape[2]])
        output = tf.matmul(output, Wc)
        closeness_output = tf.reshape(output, [closeness_output.shape[0], closeness_output.shape[1], closeness_output.shape[2]])
        
        output = tf.reshape(period_output, [period_output.shape[0]*period_output.shape[1], period_output.shape[2]])
        output = tf.matmul(output, Wp)
        period_output = tf.reshape(output, [period_output.shape[0], period_output.shape[1], period_output.shape[2]])
        
        output = tf.reshape(trend_output, [trend_output.shape[0]*trend_output.shape[1], trend_output.shape[2]])
        output = tf.matmul(output, Wt)
        trend_output = tf.reshape(output, [trend_output.shape[0], trend_output.shape[1], trend_output.shape[2]])
       
        outputs = tf.add(tf.add(closeness_output, period_output), trend_output)
        
        #adding non-linearity
        #outputs = tf.tanh(outputs)
        
        #converting the dimension from (B, H, W) -> (B, H, W, 1)
        outputs = tf.expand_dims(outputs, axis=3)
        return outputs               
