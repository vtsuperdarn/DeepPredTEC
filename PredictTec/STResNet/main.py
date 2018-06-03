'''
Author: Sneha Singhania
Date: June 3, 2018
Comment: This file contains the main program. The computation graph for ST-ResNet is built and launched in a session.
'''

from st_resnet import Graph
import tensorflow as tf
from params import Params as param
from tqdm import tqdm

if __name__ == '__main__': 
    #build the computation graph
    g = Graph()
    print "\n###Computation graph for ST-ResNet loaded###\n"
