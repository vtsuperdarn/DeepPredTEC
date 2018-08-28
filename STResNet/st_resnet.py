'''
This file defines the Tensorflow computation graph for the ST-ResNet (Deep Spatio-temporal Residual Networks) architecture. The skeleton of the architecture from inputs to outputs in defined here using calls to functions defined in modules.py. Modularity ensures that the functioning of a component can be easily modified in modules.py without changing the skeleton of the ST-ResNet architecture defined in this file.
'''

from params import Params as param
import modules as my
import tensorflow as tf
import numpy as np

class STResNetShared(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            B, H, W, C, P, T, O, F, U, V, S, N, R  = param.batch_size, param.map_height, param.map_width, param.closeness_sequence_length, param.period_sequence_length, param.trend_sequence_length, param.num_of_output_tec_maps, param.num_of_filters, param.num_of_residual_units, param.exo_values, param.gru_size, param.gru_num_layers, param.resnet_out_filters 
                        
            #ResNet architecture for the three modules
            #running closeness on first gpu, if GPU number is not specified then it runs by default on GPU:0
            with tf.device('/device:GPU:0'):
                #module 1: Capturing the closeness(recent)
                if(param.closeness_channel == True):
                    #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
                    self.c_tec = tf.placeholder(tf.float32, shape=[None, H, W, C], name="closeness_tec_maps")
                    print ("closeness input shape:", self.c_tec.shape)
                    self.closeness_input = my.ResInput(inputs=self.c_tec, filters=F, kernel_size=param.kernel_size, scope="closeness_input", reuse=None)
                    self.closeness_resnet = my.ResNet(inputs=self.closeness_input, filters=F, kernel_size=param.kernel_size, repeats=U, scope="resnet", reuse=None)
                    self.closeness_output = my.ResOutput(inputs=self.closeness_resnet, filters=R, kernel_size=param.kernel_size, scope="resnet_output", reuse=None)
            
            #running closeness on second gpu
            with tf.device('/device:GPU:1'):
                #module 2: Capturing the period(near)
                if(param.period_channel == True):
                    #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
                    self.p_tec = tf.placeholder(tf.float32, shape=[None, H, W, P], name="period_tec_maps")
                    print ("period input shape:", self.p_tec.shape)
                    self.period_input = my.ResInput(inputs=self.p_tec, filters=F, kernel_size=param.kernel_size, scope="period_input", reuse=None)
                    self.period_resnet = my.ResNet(inputs=self.period_input, filters=F, kernel_size=param.kernel_size, repeats=U, scope="resnet", reuse=True)
                    self.period_output = my.ResOutput(inputs=self.period_resnet, filters=R, kernel_size=param.kernel_size, scope="resnet_output", reuse=True)
            
            #running closeness on third gpu
            with tf.device('/device:GPU:0'):    
                #module 3: Capturing the trend(distant) 
                if(param.trend_channel == True):
                    #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
                    self.t_tec = tf.placeholder(tf.float32, shape=[None, H, W, T], name="trend_tec_maps")
                    print ("trend input shape:", self.t_tec.shape) 
                    self.trend_input = my.ResInput(inputs=self.t_tec, filters=F, kernel_size=param.kernel_size, scope="trend_input", reuse=None)
                    self.trend_resnet = my.ResNet(inputs=self.trend_input, filters=F, kernel_size=param.kernel_size, repeats=U, scope="resnet", reuse=True)
                    self.trend_output = my.ResOutput(inputs=self.trend_resnet, filters=R, kernel_size=param.kernel_size, scope="resnet_output", reuse=True)
                
            if (param.add_exogenous == True):
                #TODO: lookback for exogenous is same as trend freq*trend length (have to change this)
                self.exogenous = tf.placeholder(tf.float32, shape=[None, param.trend_freq*T, V], name="exogenous")
                print ("exogenous variable", self.exogenous.shape)
                
                #processing with exogenous variables
                #this will be of shape (batch_size, gru_size)
                self.external = my.exogenous_module(self.exogenous, S, N)
                #shape (batch_size, 1, gru_size)
                self.external = tf.expand_dims(self.external, 1)
                
                #combining the exogenous and each module output
                #populating the exogenous variable
                self.val = tf.tile(self.external, [1, H*W, 1])
                self.exo = tf.reshape(self.val, [-1, H, W, S])
                
                with tf.device('/device:GPU:0'):
                    #concatenate the modules output with the exogenous module output
                    if(param.closeness_channel == True):
                        self.close_concat = tf.concat([self.exo, self.closeness_output], 3, name="close_concat")
                        #last convolutional layer for getting information from exo and closeness module
                        self.exo_close = tf.layers.conv2d(inputs=self.close_concat, filters=O, kernel_size=param.kernel_size, strides=(1,1), padding="SAME", name="exo_close")
                
                with tf.device('/device:GPU:1'):         
                    if(param.period_channel == True):
                        self.period_concat = tf.concat([self.exo, self.period_output], 3, name="period_concat")
                        #last convolutional layer for getting information from exo and period module
                        self.exo_period = tf.layers.conv2d(inputs=self.period_concat, filters=O, kernel_size=param.kernel_size, strides=(1,1), padding="SAME", name="exo_period")
                        
                with tf.device('/device:GPU:0'):         
                    if(param.trend_channel == True):
                        self.trend_concat = tf.concat([self.exo, self.trend_output], 3, name="trend_concat")
                        #last convolutional layer for getting information from exo and trend module
                        self.exo_trend = tf.layers.conv2d(inputs=self.trend_concat, filters=O, kernel_size=param.kernel_size, strides=(1,1), padding="SAME", name="exo_trend") 
                
                # parameter-matrix-based fusion of the outputs after combining with exo
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):                
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close, period_output=self.exo_period, trend_output=self.exo_trend)
                
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close, period_output=self.exo_period)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close, period_output=None, trend_output=self.exo_trend)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close)

            else:
                # parameter-matrix-based fusion of the outputs after combining with exo
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):                
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output, period_output=self.period_output, trend_output=self.trend_output)
                
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output, period_output=self.period_output)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output, period_output=None, trend_output=self.trend_output)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output)
                    
            #shape of output tec map: (Batch_size, map_height, map_width, number of predictions)
            self.output_tec = tf.placeholder(tf.float32, shape=[None, H, W, O], name="output_tec_map") 
            print ("output shape:", self.output_tec)
            
            self.loss_weight_matrix = tf.placeholder(tf.float32, shape=[None, H, W, O], name="loss_weight_matrix") 
            print ("loss_weight_matrix:", self.loss_weight_matrix)
            
            #scaling the error using the loss_weight_tensor - elementwise operation
            self.tec_error = tf.multiply( tf.pow( (self.x_res - self.output_tec) , 2), self.loss_weight_matrix )
            print ("tec_error:", self.tec_error.shape)
            
            #here we calculate the total sum and then divide - the inbuilt function will handle overflow
            #self.loss = tf.reduce_sum(tf.pow(self.x_res - self.output_tec, 2)) / (self.x_res.shape[0]) - this is equivalent of below one - batch size is declared none - so can't use this form
            
            #this is average loss per the number of output TEC maps
            #self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum( self.tec_error, axis=3), axis=1), axis=1))
            
            #we have divide the loss by number of outputs so this is average loss per TEC map
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum( self.tec_error, axis=3), axis=1), axis=1))/(1.0*O)
             
            #we have divide the loss by number of outputs * dim of TEC map so this is average loss per pixel in a TEC map
            #self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum( self.tec_error, axis=3), axis=1), axis=1))/(1.0*O*H*W)
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr, beta1=param.beta1, beta2=param.beta2, epsilon=param.epsilon).minimize(self.loss)
            
            #loss summary
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()
            
            self.saver = tf.train.Saver(max_to_keep=None)


class STResNetIndep(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            B, H, W, C, P, T, O, F, U, V, S, N, R  = param.batch_size, param.map_height, param.map_width, param.closeness_sequence_length, param.period_sequence_length, param.trend_sequence_length, param.num_of_output_tec_maps ,param.num_of_filters, param.num_of_residual_units, param.exo_values, param.gru_size, param.gru_num_layers, param.resnet_out_filters 
            
            #ResNet architecture for the three modules
            with tf.device('/device:GPU:0'):            
                #module 1: Capturing the closeness(recent)
                if(param.closeness_channel == True):
                    #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
                    self.c_tec = tf.placeholder(tf.float32, shape=[None, H, W, C], name="closeness_tec_maps")
                    print ("closeness input shape:", self.c_tec.shape)
                    self.closeness_input = my.ResInput(inputs=self.c_tec, filters=F, kernel_size=param.kernel_size, scope="closeness_input", reuse=None)
                    self.closeness_resnet = my.ResNet(inputs=self.closeness_input, filters=F, kernel_size=param.kernel_size, repeats=U, scope="closeness_resnet", reuse=None)
                    self.closeness_output = my.ResOutput(inputs=self.closeness_resnet, filters=R, kernel_size=param.kernel_size, scope="closeness_output", reuse=None)
                    
            with tf.device('/device:GPU:1'):    
                #module 2: Capturing the period(near)
                if(param.period_channel == True):
                    #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
                    self.p_tec = tf.placeholder(tf.float32, shape=[None, H, W, P], name="period_tec_maps")
                    print ("period input shape:", self.p_tec.shape)
                    self.period_input = my.ResInput(inputs=self.p_tec, filters=F, kernel_size=param.kernel_size, scope="period_input", reuse=None)
                    self.period_resnet = my.ResNet(inputs=self.period_input, filters=F, kernel_size=param.kernel_size, repeats=U, scope="period_resnet", reuse=None)
                    self.period_output = my.ResOutput(inputs=self.period_resnet, filters=R, kernel_size=param.kernel_size, scope="period_output", reuse=None)
                    
            with tf.device('/device:GPU:0'):    
                #module 3: Capturing the trend(distant) 
                if(param.trend_channel == True):
                    #shape of a tec map: (Batch_size, map_height, map_width, depth(num of history tec maps))
                    self.t_tec = tf.placeholder(tf.float32, shape=[None, H, W, T], name="trend_tec_maps")
                    print ("trend input shape:", self.t_tec.shape)
                    self.trend_input = my.ResInput(inputs=self.t_tec, filters=F, kernel_size=param.kernel_size, scope="trend_input", reuse=None)
                    self.trend_resnet = my.ResNet(inputs=self.trend_input, filters=F, kernel_size=param.kernel_size, repeats=U, scope="trend_resnet", reuse=None)
                    self.trend_output = my.ResOutput(inputs=self.trend_resnet, filters=R, kernel_size=param.kernel_size, scope="trend_output", reuse=None)
                    
            
            if (param.add_exogenous == True):
                #lookback for exogenous is same as trend freq*trend length
                self.exogenous = tf.placeholder(tf.float32, shape=[None, param.trend_freq*T, V], name="exogenous")
                print ("exogenous variable", self.exogenous.shape)
                
                #processing with exogenous variables
                #this will be of shape (batch_size, gru_size)
                self.external = my.exogenous_module(self.exogenous, S, N)
                #shape (batch_size, 1, gru_size)
                self.external = tf.expand_dims(self.external, 1)
                
                #combining the exogenous and each module output
                #populating the exogenous variable
                self.val = tf.tile(self.external, [1, H*W, 1])
                self.exo = tf.reshape(self.val, [-1, H, W, S])
                
                #concatenate the modules output with the exogenous module output                
                with tf.device('/device:GPU:0'):
                    if(param.closeness_channel == True):
                        self.close_concat = tf.concat([self.exo, self.closeness_output], 3, name="close_concat")
                        #last convolutional layer for getting information from exo and closeness module
                        self.exo_close = tf.layers.conv2d(inputs=self.close_concat, filters=O, kernel_size=param.kernel_size, strides=(1,1), padding="SAME", name="exo_close") 
                
                with tf.device('/device:GPU:1'):
                    if(param.period_channel == True):
                        self.period_concat = tf.concat([self.exo, self.period_output], 3, name="period_concat")
                        #last convolutional layer for getting information from exo and period module
                        self.exo_period = tf.layers.conv2d(inputs=self.period_concat, filters=O, kernel_size=param.kernel_size, strides=(1,1), padding="SAME", name="exo_period") 
                
                with tf.device('/device:GPU:0'):
                    if(param.trend_channel == True):
                        self.trend_concat = tf.concat([self.exo, self.trend_output], 3, name="trend_concat")
                        #last convolutional layer for getting information from exo and trend module
                        self.exo_trend = tf.layers.conv2d(inputs=self.trend_concat, filters=O, kernel_size=param.kernel_size, strides=(1,1), padding="SAME", name="exo_trend") 
                
                # parameter-matrix-based fusion of the outputs after combining with exo
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):                
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close, period_output=self.exo_period, trend_output=self.exo_trend)
                
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close, period_output=self.exo_period)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close, period_output=None, trend_output=self.exo_trend)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.exo_close)

            else:
                # parameter-matrix-based fusion of the outputs after combining with exo
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):                
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output, period_output=self.period_output, trend_output=self.trend_output)
                
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output, period_output=self.period_output)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output, period_output=None, trend_output=self.trend_output)
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    self.x_res = my.Fusion(scope="fusion", shape=[W, W], num_outputs=O, closeness_output=self.closeness_output)
                    
            #shape of output tec map: (Batch_size, map_height, map_width, number of predictions)
            self.output_tec = tf.placeholder(tf.float32, shape=[None, H, W, O], name="output_tec_map") 
            print ("output shape:", self.output_tec)
            
            self.loss_weight_matrix = tf.placeholder(tf.float32, shape=[None, H, W, O], name="loss_weight_matrix") 
            print ("loss_weight_matrix:", self.loss_weight_matrix)
            
            #scaling the error using the loss_weight_tensor - elementwise operation
            self.tec_error = tf.multiply( tf.pow( (self.x_res - self.output_tec) , 2), self.loss_weight_matrix )
            print ("tec_error:", self.tec_error.shape)
            
            #here we calculate the total sum and then divide - the inbuilt function will handle overflow
            #self.loss = tf.reduce_sum(tf.pow(self.x_res - self.output_tec, 2)) / (self.x_res.shape[0]) - this is equivalent of below one - batch size is declared none - so can't use this form
            
            #this is average loss per the number of output TEC maps
            #self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum( self.tec_error, axis=3), axis=1), axis=1))
            
            #we have divide the loss by number of outputs so this is average loss per TEC map
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum( self.tec_error, axis=3), axis=1), axis=1))/(1.0*O)
             
            #we have divide the loss by number of outputs * dim of TEC map so this is average loss per pixel in a TEC map
            #self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum( self.tec_error, axis=3), axis=1), axis=1))/(1.0*O*H*W)
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr, beta1=param.beta1, beta2=param.beta2, epsilon=param.epsilon).minimize(self.loss)
            
            #loss summary
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()
            
            self.saver = tf.train.Saver(max_to_keep=None)
            

