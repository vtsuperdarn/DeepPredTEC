from st_resnet import STResNetShared, STResNetIndep
import tensorflow as tf
from params import Params as param
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
import datetime as dt
import os
import numpy
import time
from omn_utils import OmnData
from batch_utils import BatchDateUtils, TECUtils


if(param.independent_channels == True): 
    g = STResNetIndep()
    print ("Computation graph for ST-ResNet with independent channels loaded\n")

else:
    g = STResNetShared()
    print ("Computation graph for ST-ResNet with shared channels loaded\n")
    
train_writer = tf.summary.FileWriter('./logdir/train', g.loss.graph)
val_writer = tf.summary.FileWriter('./logdir/val', g.loss.graph)

# Parameters for loading batch data
# initialize parameters
# current_datetime = dt.datetime(2015, 1, 2, 10, 5)
#file_dir="/home/sd-guest/Documents/data/tec_filled/"
file_dir=param.file_dir

#closeness is sampled 12 times every 5 mins, lookback = (12*5min = 1 hour)
#freq 1 is 5mins
#size corresponds to the sample size
closeness_size = param.closeness_sequence_length

#period is sampled 24 times every 1 hour (every 12th index), lookback = (24*12*5min = 1440min = 1day)
period_size = param.period_sequence_length

#trend is sampled 24 times every 3 hours (every 36th index), lookback = (8*36*5min = 1440min = 1day)
trend_size = param.trend_sequence_length

# get date ranges
start_date = param.start_date
end_date = param.end_date

# get all corresponding dates for batches
batchObj = BatchDateUtils(start_date, end_date, param.batch_size, param.tec_resolution,\
                         param.closeness_freq, closeness_size, param.period_freq, period_size,\
                         param.trend_freq, trend_size, param.num_of_output_tec_maps, param.output_freq,\
                         param.closeness_channel, param.period_channel, param.trend_channel)

batch_date_arr = np.array( list(batchObj.batch_dict.keys()) )

# Bulk load TEC data
t01 = time.time()
tecObj = TECUtils(start_date, end_date, file_dir, param.tec_resolution, param.load_window,\
                 param.closeness_channel, param.period_channel, param.trend_channel)
t02 = time.time()
#print type(tecObj.tec_data)
print("TEC bulk load time ---->" + str(t02-t01))

#Making the model path unique
param.model_path = param.model_path + '_' + str(t02-t01)
print ("Model path: " + param.model_path)

#path for saving the loss values and mean_std.npy
path = param.model_path + "_values"
os.makedirs(path)

# setting appropriate vars and reading 
omn_dbdir = param.omn_dbdir
omn_db_name = param.omn_db_name
omn_table_name = param.omn_table_name
omn_train=True

# Getting OMNI data for training
start_date_omni =  start_date - dt.timedelta(days=param.load_window)
end_date_omni =  end_date + dt.timedelta(days=param.load_window)
omnObj = OmnData( start_date_omni, end_date_omni, omn_dbdir, omn_db_name, omn_table_name , omn_train, param.imf_normalize, path)

#shuffle the batch dates. This is an inplace operation. Uncomment below line for shuffling
#numpy.random.shuffle(batch_date_arr)

# divide the dates into train and test
train_test_ratio = param.train_test_ratio
train_ind = int(round((train_test_ratio*batch_date_arr.shape[0]), 0))
date_arr_train = batch_date_arr[:train_ind]
date_arr_test = batch_date_arr[train_ind:]


# Start training the model
train_loss = []
validation_loss = []

with tf.Session(graph=g.graph) as sess:
    sess.run(tf.global_variables_initializer())    
    for epoch in tqdm(range(param.num_epochs)):            
        loss_train = 0
        loss_val = 0
        print(("epoch: {}\t".format(epoch), ))
        
        # For shuffling the data again uncomment below line
        #numpy.random.shuffle(date_arr_train)
        
        # TRAINING
        for tr_ind, current_datetime in tqdm(enumerate(date_arr_train)):
            #print("Training date-->" + current_datetime.strftime("%Y%m%d-%H%M"))
            
            #if we need to use the exogenous module
            if (param.add_exogenous == True):
                #GET IMF batch data
                t1 = time.time()
                #TODO change the trend_freq with look_back
                imf_batch = omnObj.get_omn_batch(current_datetime, param.batch_size, param.trend_freq, trend_size )
                t2 = time.time()
                print("\nTime to load IMF batch for training " + str(t2-t1))
                t1 = time.time()
                
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.p_tec: data_period,
                                                                   g.t_tec: data_trend,
                                                                   g.output_tec: data_out,
                                                                   g.exogenous: imf_batch})
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))
                    
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.p_tec: data_period,
                                                                   g.output_tec: data_out,
                                                                   g.exogenous: imf_batch})
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))
                    
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.t_tec: data_trend,
                                                                   g.output_tec: data_out,
                                                                   g.exogenous: imf_batch})
                                                                                                          
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))
                    
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.output_tec: data_out,
                                                                   g.exogenous: imf_batch})
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))
            
            #if we don't want to use the exogenous module                                                   
            else:
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.p_tec: data_period,
                                                                   g.t_tec: data_trend,
                                                                   g.output_tec: data_out})
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))
                    
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.p_tec: data_period,
                                                                   g.output_tec: data_out})
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))
                    
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.t_tec: data_trend,
                                                                   g.output_tec: data_out})
                                                                                                          
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))
                    
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    # get the batch of data points
                    t1 = time.time()
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_out = tecObj.create_batch(curr_batch_time_dict)
                    t2 = time.time()
                    print("tec batch for training " + str(t2-t1))
                    
                    t1 = time.time()
                    loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                        feed_dict={g.c_tec: data_close,
                                                                   g.output_tec: data_out})
                    t2 = time.time()
                    print("TF for training " + str(t2-t1))

            loss_train = loss_tr * param.delta + loss_train * (1 - param.delta)
            train_writer.add_summary(summary, tr_ind + len(date_arr_train) * epoch)
        print( "total dates trained--->" + str(len(date_arr_train)) )
        print("NOW VALIDATING...")
        
        # For shuffling the data again uncomment below line
        #numpy.random.shuffle(date_arr_test)
        
        # TESTING/VALIDATION
        for te_ind, current_datetime in tqdm(enumerate(date_arr_test)):
            #print("Testing date-->" + current_datetime.strftime("%Y%m%d-%H%M"))
            
            #if we need to use the exogenous module
            if (param.add_exogenous == True):

                #get the IMF data
                #TODO change the trend freq
                imf_batch = omnObj.get_omn_batch(current_datetime, param.batch_size, param.trend_freq,trend_size )                
                
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.p_tec: data_period,
                                                       g.t_tec: data_trend,
                                                       g.output_tec: data_out,
                                                       g.exogenous: imf_batch})
                                                       
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.p_tec: data_period,
                                                       g.output_tec: data_out,
                                                       g.exogenous: imf_batch})
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.p_tec: data_trend,
                                                       g.output_tec: data_out,
                                                       g.exogenous: imf_batch}) 
                                                                                                 
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.output_tec: data_out,
                                                       g.exogenous: imf_batch}) 
                                                       
            #if we don't want to use the exogenous module  
            else:
                if(param.closeness_channel == True and param.period_channel == True and param.trend_channel == True):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.p_tec: data_period,
                                                       g.t_tec: data_trend,
                                                       g.output_tec: data_out})
                
                elif(param.closeness_channel == True and param.period_channel == True and param.trend_channel == False):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_period, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.p_tec: data_period,
                                                       g.output_tec: data_out})
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == True):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_trend, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.t_tec: data_trend,
                                                       g.output_tec: data_out})                                       
                
                elif(param.closeness_channel == True and param.period_channel == False and param.trend_channel == False):
                    # get the batch of data points
                    curr_batch_time_dict = batchObj.batch_dict[current_datetime]
                    data_close, data_out = tecObj.create_batch(curr_batch_time_dict)
                    
                    loss_v, summary = sess.run([g.loss, g.merged],
                                            feed_dict={g.c_tec: data_close,
                                                       g.output_tec: data_out})
            
            loss_val += loss_v
            val_writer.add_summary(summary, te_ind + len(date_arr_test) * epoch)
        
        #accuracy_val /= num_batches
        if(len(date_arr_test) > 0):
            loss_val /= len(date_arr_test)
        print(("loss: {:.3f}, val_loss: {:.3f}".format(loss_train, loss_val)))
        
        #saving the loss values in an array
        train_loss.append(loss_train)
        validation_loss.append(loss_val)
        
        #Saving the model and the loss values after every 3 epoch runs
        if (epoch % 3 == 0):
            g.saver.save(sess, param.model_path+"/epoch_{}".format(epoch))
            np.save(path+'/training_loss.npy', np.array(train_loss))
            np.save(path+'/validation_loss.npy', np.array(validation_loss))
    
    #Saving the model after the final epoch
    g.saver.save(sess, param.model_path+"/current")
    np.save(path+'/training_loss.npy', np.array(train_loss))
    np.save(path+'/validation_loss.npy', np.array(validation_loss))

train_writer.close()
val_writer.close()
#Use tensorboard for visualization of parameters
print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
