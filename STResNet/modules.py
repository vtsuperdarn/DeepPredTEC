'''
This file contain helper functions and custom neural layers. The functions help in abstracting the complexity of the architecture and Tensorflow features. These functions are being called in the st_resnet.py for defining the computational graph
'''

import tensorflow as tf
import numpy as np


def gru_cell(gru_size):
  return tf.contrib.rnn.GRUCell(gru_size, activation=tf.nn.relu) 
  

def exogenous_module(inputs, gru_size, num_layers=2):
    '''
    GRU encoder module for processing the exogenous data (By, Bz, Vx, Np)
    '''
    #stacked gru implementation
    #gru_cells1 = [gru_cell(gru_size) for _ in range(num_layers)]
    #stacked_gru1 = tf.contrib.rnn.MultiRNNCell(gru_cells1)
    #annotations, state = tf.nn.dynamic_rnn(stacked_gru1, inputs=inputs, dtype=tf.float32, time_major=False, scope="gru")
    
    #single layer lstm
    annotations, state = tf.nn.dynamic_rnn(gru_cell(gru_size), inputs=inputs, dtype=tf.float32, scope="gru")
    
    annotations = tf.transpose(annotations, [1, 0, 2])
    
    #extracting the last annotation or hidden state
    #(Batch_size, gru_size)
    output = tf.gather(annotations, int(annotations.get_shape()[0]) -1)      
    return output
        
        
def ResUnit(inputs, filters, kernel_size, strides, scope, reuse=None):   
    '''
    Defines a residual unit(reference paper): input->[layernorm->relu->conv->layernorm->relu->conv]->reslink-> output
    Alternative(original resnet paper): input->[conv->batchnorm->relu->conv->batchnorm]->reslink->relu->output 
    '''
    with tf.variable_scope(scope, reuse=reuse): 
        '''
        #perform a 2D convolution
        conv1 = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding="SAME", name="conv1", reuse=reuse)  
        
        #use layernorm before applying convolution
        layernorm1 = tf.contrib.layers.layer_norm(conv1, scope="layernorm1", reuse=reuse)
        
        #relu activation
        relu_output = tf.nn.relu(layernorm1, name="relu")
        
        #perform a 2D convolution
        conv2 = tf.layers.conv2d(relu_output, filters, kernel_size, strides, padding="SAME", name="conv2", reuse=reuse)                
        
        #use layernorm before applying convolution
        layernorm2 = tf.contrib.layers.layer_norm(conv2, scope="layernorm2", reuse=reuse)
        
        #adding the res link        
        outputs = layernorm2 + inputs

        #relu activation
        outputs = tf.nn.relu(outputs, name="relu")
        '''
        
        #use layernorm first as we have already applied convolution in the resinput layer 
        layernorm1 = tf.contrib.layers.layer_norm(inputs, scope="layernorm1", reuse=reuse)
        #relu activation
        relu_output1 = tf.nn.relu(layernorm1, name="relu")
        #perform a 2D convolution
        conv1 = tf.layers.conv2d(relu_output1, filters, kernel_size, strides, padding="SAME", name="conv1", reuse=reuse)  
        
        #use layernorm before applying convolution
        layernorm2 = tf.contrib.layers.layer_norm(conv1, scope="layernorm2", reuse=reuse)
        #relu activation
        relu_output2 = tf.nn.relu(layernorm2, name="relu")
        #perform a 2D convolution
        conv2 = tf.layers.conv2d(relu_output2, filters, kernel_size, strides, padding="SAME", name="conv2", reuse=reuse)                
        
        #adding the res link        
        outputs = conv2 + inputs
        
        return outputs    

def ResInput(inputs, filters, kernel_size, scope, reuse=None):
    '''
    Defines the first (input) layer of the ResNet architecture
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1), padding="SAME", name="conv_input", reuse=reuse)
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

def ResOutput(inputs, filters, kernel_size, scope, reuse=None):
    '''
    Defines the last (output) layer of the ResNet architecture
    '''
    with tf.variable_scope(scope, reuse=reuse):
        #applying the final convolution to the tec map with depth 1 (num of filters=1)
        outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1), padding="SAME", name="conv_out", reuse=reuse)       
        return outputs
                  
        
def Fusion(scope, shape, num_outputs, closeness_output=None, period_output=None, trend_output=None):
    '''
    Combining the output from the module into one tec map
    '''            
    with tf.variable_scope(scope):
        
        #initializing the weight matrics
        if(closeness_output != None):
            Wc = tf.get_variable("closeness_matrix", dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        if(period_output != None):
            Wp = tf.get_variable("period_matrix", dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        if(trend_output != None):
            Wt = tf.get_variable("trend_matrix", dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        
        if(num_outputs == 1):
            
            with tf.device('/device:GPU:0'):
                if(closeness_output != None):            
                    output = tf.reshape(closeness_output, [-1, closeness_output.shape[2]])
                    output = tf.matmul(output, Wc)
                    closeness_output = tf.reshape(output, [-1, closeness_output.shape[1], closeness_output.shape[2]])
            
            with tf.device('/device:GPU:1'):    
                if(period_output != None):
                    output = tf.reshape(period_output, [-1, period_output.shape[2]])
                    output = tf.matmul(output, Wp)
                    period_output = tf.reshape(output, [-1, period_output.shape[1], period_output.shape[2]])
            
            with tf.device('/device:GPU:2'):    
                if(trend_output != None):
                    output = tf.reshape(trend_output, [-1, trend_output.shape[2]])
                    output = tf.matmul(output, Wt)
                    trend_output = tf.reshape(output, [-1, trend_output.shape[1], trend_output.shape[2]])
               
            if(closeness_output != None and period_output != None and trend_output != None):
                outputs = tf.add(tf.add(closeness_output, period_output), trend_output)
            elif(closeness_output != None and period_output != None and trend_output == None):
                outputs = tf.add(closeness_output, period_output)
            elif(closeness_output != None and period_output == None and trend_output != None):
                outputs = tf.add(closeness_output, trend_output)
            elif(closeness_output != None and period_output == None and trend_output == None):
                outputs = closeness_output
        
            #converting the dimension from (B, H, W) -> (B, H, W, 1)
            outputs = tf.expand_dims(outputs, axis=3)
    
        #if the number of outputs is greater than 1 then the matrix transformation operations are different
        else:
            with tf.device('/device:GPU:0'):
                if(closeness_output != None):
                    closeness_output = tf.transpose(closeness_output, [0, 3, 1, 2])
                    coutput = tf.reshape(closeness_output, [-1, closeness_output.shape[3]])
                    coutput = tf.matmul(coutput, Wc)
                    closeness_output = tf.reshape(coutput, [-1, closeness_output.shape[1], closeness_output.shape[2], closeness_output.shape[3]])
            
            with tf.device('/device:GPU:1'):
                if(period_output != None):
                    period_output = tf.transpose(period_output, [0, 3, 1, 2])
                    poutput = tf.reshape(period_output, [-1, period_output.shape[3]])
                    poutput = tf.matmul(poutput, Wp)
                    period_output = tf.reshape(poutput, [-1, period_output.shape[1], period_output.shape[2], period_output.shape[3]])
            
            with tf.device('/device:GPU:2'):
                if(trend_output != None):
                    trend_output = tf.transpose(trend_output, [0, 3, 1, 2])
                    toutput = tf.reshape(trend_output, [-1, trend_output.shape[3]])
                    toutput = tf.matmul(toutput, Wt)
                    trend_output = tf.reshape(toutput, [-1, trend_output.shape[1], trend_output.shape[2], trend_output.shape[3]])
               
            if(closeness_output != None and period_output != None and trend_output != None):
                outputs = tf.add(tf.add(closeness_output, period_output), trend_output)
            elif(closeness_output != None and period_output != None and trend_output == None):
                outputs = tf.add(closeness_output, period_output)
            elif(closeness_output != None and period_output == None and trend_output != None):
                outputs = tf.add(closeness_output, trend_output)
            elif(closeness_output != None and period_output == None and trend_output == None):
                outputs = closeness_output
            
            #converting the dimension from (B, O, H, W) -> (B, H, W, O)
            outputs = tf.transpose(outputs, [0, 2, 3, 1])
        return outputs 
        
