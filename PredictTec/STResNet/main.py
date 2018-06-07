'''
Author: Sneha Singhania
Date: June 3, 2018
Comment: This file contains the main program. The computation graph for ST-ResNet is built and launched in a session.
'''

from st_resnet import Graph
import tensorflow as tf
from params import Params as param
from tqdm import tqdm
from utils import batch_generator
import numpy as np
import h5py

if __name__ == '__main__': 
    #build the computation graph
    g = Graph()
    print ("Computation graph for ST-ResNet loaded\n")
    
    
    x_closeness = []
    x_period = []
    x_trend = []
    y = []
    train_writer = tf.summary.FileWriter('./logdir/train', g.loss.graph)
    val_writer = tf.summary.FileWriter('./logdir/train', g.loss.graph)   
    
    for epoch in range(param.num_epochs):     
        for file_no in range(1, 89):
            
                
            
            with h5py.File("output_files/xcloseness_"+str(file_no)+".h5", 'r') as hf:
                x_closeness = hf["xcloseness_"+str(file_no)][:].tolist()
            with h5py.File("output_files/xperiod_"+str(file_no)+".h5", 'r') as hf:
                x_period = hf["xperiod_"+str(file_no)][:].tolist()         
            with h5py.File("output_files/xtrend_"+str(file_no)+".h5", 'r') as hf:
                x_trend = hf["xtrend_"+str(file_no)][:].tolist()  
            with h5py.File("output_files/ydata_"+str(file_no)+".h5", 'r') as hf:
                y = hf["ydata_"+str(file_no)][:].tolist()               
            print "Loaded file {} in epoch {}".format(file_no, epoch)
            x_closeness = np.array(x_closeness)  
            x_period = np.array(x_period)  
            x_trend = np.array(x_trend)  
        
            x_closeness = np.transpose(np.squeeze(x_closeness), (0, 2, 3, 1))
            x_period = np.transpose(np.squeeze(x_period), (0, 2, 3, 1))
            x_trend = np.transpose(np.squeeze(x_trend), (0, 2, 3, 1))
          
            #x_closeness = np.random.rand(20,75,73,12)
            #x_period = np.random.rand(20,75,73,24)
            #x_trend = np.random.rand(20,75,73,8)
            #y = np.random.rand(20, 75, 73, 1)
        
            print x_closeness[0].shape
            
            X = []    
            for i in range(x_closeness.shape[0]):
                X.append([x_closeness[i].tolist(), x_period[i].tolist(), x_trend[i].tolist()])
        
            print len(X)
            train_index = int(round((0.75*len(X)),0))
            xtrain = X[:train_index]
            ytrain = y[:train_index]
            xtest = X[train_index:]
            ytest = y[train_index:]
        
        
            #print (xtrain[0:3])
            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)
            xtest = np.array(xtest)
            ytest = np.array(ytest)
            #print ("hiiiiiiiiiiiiii")
            train_batch_generator = batch_generator(xtrain, ytrain, param.batch_size)
            test_batch_generator = batch_generator(xtest, ytest, param.batch_size)
        
            print("Start learning ..... ")
            
            #sv = tf.train.Supervisor(graph=g.graph, logdir=param.logdir) 
            

        
        
        
            with tf.Session(graph=g.graph) as sess:
                sess.run(tf.global_variables_initializer())
                if (file_no > 1):
                    g.saver.restore(sess, param.model_path+"/current")   
                
                            
                    loss_train = 0
                    loss_val = 0
                    

                    print("epoch: {}\t".format(epoch), )

                    # Training
                    num_batches = xtrain.shape[0] // param.batch_size
                    for b in tqdm(range(num_batches)):
                        #print ("before")
                        x_batch, y_batch = next(train_batch_generator)
                        #print ("after")
                        x_closeness = np.array(x_batch[:, 0].tolist())
                        x_period = np.array(x_batch[:, 1].tolist())
                        x_trend = np.array(x_batch[:, 2].tolist())
                        #print x_closeness.shape
                        #print x_period.shape
                        #print x_trend.shape
                        #print ("startttt")
                        
                        loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                            feed_dict={g.c_tec: x_closeness,
                                                                       g.p_tec: x_period,
                                                                       g.t_tec: x_trend,
                                                                       g.output_tec: y_batch})
                        #print ("endd")                                               
                        #accuracy_train += acc
                        loss_train = loss_tr * param.delta + loss_train * (1 - param.delta)
                        train_writer.add_summary(summary, b + num_batches * epoch)
                    #accuracy_train /= num_batches

                    # testing
                    num_batches = xtest.shape[0] // param.batch_size
                    for b in tqdm(range(num_batches)):
                        x_batch, y_batch = next(test_batch_generator)
                        
                        x_closeness = np.array(x_batch[:, 0].tolist())
                        x_period = np.array(x_batch[:, 1].tolist())
                        x_trend = np.array(x_batch[:, 2].tolist())
                        
                        loss_v, summary = sess.run([g.loss, g.merged],
                                                            feed_dict={g.c_tec: x_closeness,
                                                                       g.p_tec: x_period,
                                                                       g.t_tec: x_trend,
                                                                       g.output_tec: y_batch})
                        #accuracy_val += acc
                        loss_val += loss_v
                        val_writer.add_summary(summary, b + num_batches * epoch)
                    #accuracy_val /= num_batches
                    if(num_batches != 0):
                        loss_val /= num_batches

                    print("loss: {:.3f}, val_loss: {:.3f}".format(loss_train, loss_val))
                
                train_writer.close()
                val_writer.close()
                g.saver.save(sess, param.model_path+"/current")
                if(file_no % 3 == 0):
                    g.saver.save(sess, param.model_path+"/{}_file_no".format(file_no))
    print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
