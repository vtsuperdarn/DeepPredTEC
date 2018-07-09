'''
This file gets the predicted tec maps by first loading the saved model and then running on the test input.
'''

from st_resnet import Graph
import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm
from utils import unshuffle_batch_generator
from params import Params as param


if __name__ == '__main__': 
    g = Graph()
    print ("Computation graph for ST-ResNet loaded\n")
    
        
    x_closeness = []
    x_period = []
    x_trend = []
    y = []
    X = []
    ctr = 3001
    
    #loading exogenous data points
    exo = np.load("exogenous_jan.npy")
    
    for file_no in range(21, 28):
        with h5py.File("output_files/xcloseness_"+str(file_no)+".h5", 'r') as hf:
            x_closeness += hf["xcloseness_"+str(file_no)][:].tolist()
        with h5py.File("output_files/xperiod_"+str(file_no)+".h5", 'r') as hf:
            x_period += hf["xperiod_"+str(file_no)][:].tolist()         
        with h5py.File("output_files/xtrend_"+str(file_no)+".h5", 'r') as hf:
            x_trend += hf["xtrend_"+str(file_no)][:].tolist()  
        with h5py.File("output_files/ydata_"+str(file_no)+".h5", 'r') as hf:
            y += hf["ydata_"+str(file_no)][:].tolist()               
        print "Loaded file {}".format(file_no)
    
        
    x_closeness = np.array(x_closeness)  
    x_period = np.array(x_period)  
    x_trend = np.array(x_trend)  
    y = np.array(y)
    
    
    print x_closeness.shape
    print x_period.shape
    print x_trend.shape
    
    x_closeness = np.transpose(np.squeeze(x_closeness), (0, 2, 3, 1))
    x_period = np.transpose(np.squeeze(x_period), (0, 2, 3, 1))
    x_trend = np.transpose(np.squeeze(x_trend), (0, 2, 3, 1)) 
    y = np.transpose(np.squeeze(y), (0, 2, 3, 1))  
    
    print x_closeness.shape
    print x_period.shape
    print x_trend.shape
    print y.shape
    
    selected_exo = exo[ctr:ctr+x_closeness.shape[0], :]
    
    print selected_exo.shape
    
    X = []    
    for j in range(x_closeness.shape[0]):
        X.append([x_closeness[j].tolist(), x_period[j].tolist(), x_trend[j].tolist(), selected_exo[j].tolist()])
    
    
    
    xtest = np.array(X)
    ytest = np.array(y)
    
    test_batch_generator = unshuffle_batch_generator(xtest, ytest, param.batch_size)
            

    with tf.Session(graph=g.graph) as sess:
        g.saver.restore(sess, param.model_path+"/epoch_{}_file{}".format(10, 16))
        #g.saver.restore(sess, param.model_path+"/current")
        
        num_batches = xtest.shape[0] // param.batch_size
        for b in tqdm(range(num_batches)):
            x_batch, y_batch = next(test_batch_generator)
            
            x_closeness = np.array(x_batch[:, 0].tolist())
            x_period = np.array(x_batch[:, 1].tolist())
            x_trend = np.array(x_batch[:, 2].tolist())
            exogenous = np.array(x_batch[:, 3].tolist())
            
            loss_v, pred, truth, closeness, period, trend = sess.run([g.loss, g.x_res, g.output_tec, g.exo_close, g.exo_period, g.exo_trend],
                                                feed_dict={g.c_tec: x_closeness,
                                                           g.p_tec: x_period,
                                                           g.t_tec: x_trend,
                                                           g.output_tec: y_batch,
                                                           g.exogenous: exogenous})

            print("val_loss: {:.3f}".format(loss_v))        

            np.save('exopredicted_tec_files/{}_pred_{:.1f}'.format(b, loss_v.item())+'.npy', pred)
            np.save('exopredicted_tec_files/{}_y_{:.1f}'.format(b, loss_v.item())+'.npy', truth)
            np.save('exopredicted_tec_files/{}_exoclose_{:.1f}'.format(b, loss_v.item())+'.npy', closeness)
            np.save('exopredicted_tec_files/{}_exoperiod_{:.1f}'.format(b, loss_v.item())+'.npy', period)
            np.save('exopredicted_tec_files/{}_exotrend_{:.1f}'.format(b, loss_v.item())+'.npy', trend)
            print 'Saving {} batch with {:.1f}'.format(b, loss_v.item())                 
