# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 02:18:09 2019

@author: yu
"""

import numpy as np
import tensorflow as tf

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import pdb

# --------- parameters -------------------- #
# classification
NUM_HIDDEN1 = 128 # node of 1 hidden
NUM_HIDDEN2 = 128 # node of 2 hidden
NUM_HIDDEN3 = 128 # node of 3 hidden

# regression
nRegHidden = 128 # node of 1 hidden
nRegHidden2 = 128 # node of 2 hidden
nRegHidden3 = 128 # node of 3 hidden
nRegHidden4 = 128 # node of 4 hidden
nRegOutput = 1
# ----------------------------------------- #

# ----------------------------------------------------------------------- #      
def weight_variable(name,shape):
     return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
# ----------------------------------------------------------------------- #      
def bias_variable(name,shape):
     return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
# ----------------------------------------------------------------------- #          
def fc_relu(inputs,w,b,keepProb):
     relu = tf.matmul(inputs,w) + b
     relu = tf.nn.dropout(relu, keepProb)
     relu = tf.nn.relu(relu)
     return relu
# ----------------------------------------------------------------------- #          
def fc(inputs,w,b,keepProb):
     fc = tf.matmul(inputs,w) + b
     fc = tf.nn.dropout(fc, keepProb)
     return fc
# ----------------------------------------------------------------------- #              
def fc_sigmoid(inputs,w,b,keepProb):
    """
    sigmoid function.
    """
    sigmoid = tf.matmul(inputs,w) + b
    sigmoid = tf.nn.dropout(sigmoid,keepProb)
    sigmoid = tf.nn.sigmoid(sigmoid)
    return sigmoid
# ----------------------------------------------------------------------- #
def Classify(x,reuse=False,NUM_CLS=0,name_scope="Classify"):
    with tf.variable_scope(name_scope) as scope:
        keepProb = 1.0
        if reuse:
            keepProb = 1.0
            scope.reuse_variables()
        
        w1_cls = weight_variable("w1_cls",[NUM_HIDDEN1, NUM_HIDDEN2])
        bias1_cls = bias_variable("bias1_cls",[NUM_HIDDEN2])
        
        h1 = fc_relu(x,w1_cls,bias1_cls,keepProb)
        
        w2_cls = weight_variable("w2_cls",[NUM_HIDDEN2, NUM_HIDDEN3])
        bias2_cls = bias_variable("bias2_cls",[NUM_HIDDEN3])
        
        h2 = fc_relu(h1,w2_cls,bias2_cls,keepProb)
        
        # input -> hidden -> output in nankai
        w3_1_cls = weight_variable("w3_1_cls",[NUM_HIDDEN3, NUM_CLS])
        bias3_1_cls = bias_variable("bias3_1_cls",[NUM_CLS])
        
        # input -> hidden -> output in nankai
        w3_2_cls = weight_variable("w3_2_cls",[NUM_HIDDEN3, NUM_CLS])
        bias3_2_cls = bias_variable("bias3_2_cls",[NUM_CLS])
        
        # in tonankai
        w3_3_cls = weight_variable("w3_3_cls",[NUM_HIDDEN3, NUM_CLS])
        bias3_3_cls = bias_variable("bias3_3_cls",[NUM_CLS])
        
        # in tonankai
        w3_4_cls = weight_variable("w3_4_cls",[NUM_HIDDEN3, NUM_CLS])
        bias3_4_cls = bias_variable("bias3_4_cls",[NUM_CLS])
        
        # tokai
        w3_5_cls = weight_variable("w3_5_cls",[NUM_HIDDEN3, NUM_CLS])
        bias3_5_cls = bias_variable("bias3_5_cls",[NUM_CLS])
    
        # shape=[BATCH_SIZE,NUM_CLS]
        y1 = fc_sigmoid(h2,w3_1_cls,bias3_1_cls,keepProb)
        y2 = fc_sigmoid(h2,w3_2_cls,bias3_2_cls,keepProb)
        y3 = fc_sigmoid(h2,w3_3_cls,bias3_3_cls,keepProb)
        y4 = fc_sigmoid(h2,w3_4_cls,bias3_4_cls,keepProb)
        y5 = fc_sigmoid(h2,w3_5_cls,bias3_5_cls,keepProb)
        
        return y1,y2,y3,y4,y5
# --------------------------------------------------------------------------- #
def Regress(x,reuse=False,depth=0,name_scope="Regress"):
    
    """
    Fully-connected regression networks.
    [arguments]
    x: input data (feature vector or residual, shape=[None, LEN_SEQ*NUM_CELL])
    reuse=False: Train, reuse=True: Evaluation & Test (variables sharing)
    depth=3: 3layer, depth=4: 4layer, depth=5: 5layer
    [activation]
    atr-nets: relu -> relu -> sigmoid
    ordinary regression & anchor-based: relu -> relu -> none
    [parameter]
    keepProb: dropout
    """
    with tf.variable_scope(name_scope) as scope:  
        keepProb = 1.0
        if reuse:
            keepProb = 1.0            
            scope.reuse_variables()
        
        # テキトー
        nRegInput = x.get_shape().as_list()[1]
        w1_reg = weight_variable('w1_reg',[nRegInput,nRegHidden])
        bias1_reg = bias_variable('bias1_reg',[nRegHidden])
        
        # input -> hidden1
        h1 = fc_relu(x,w1_reg,bias1_reg,keepProb)
        
        if depth == 3:
            
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegOutput])
            bias2_reg = bias_variable('bias2_reg',[nRegOutput])
            
            # hidden1 -> hidden2
            # shape=[None,number of dimention (y)]
            return fc(h1,w2_reg,bias2_reg,keepProb)
       
        elif depth == 4:
            
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            # hidden1 -> hidden2
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegOutput])
            bias3_reg = bias_variable('bias3_reg',[nRegOutput])
            
            # hidden2 -> hidden3
            return fc(h2,w3_reg,bias3_reg,keepProb)
       
        elif depth == 5:
            
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            # hidden1 -> hidden2
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegHidden3])
            bias3_reg = bias_variable('bias3_reg',[nRegHidden3])
            # hidden2 -> hidden3
            h3 = fc_relu(h2,w3_reg,bias3_reg,keepProb)
        
            w4_reg = weight_variable('w4_reg',[nRegHidden3,nRegOutput])
            bias4_reg = bias_variable('bias4_reg',[nRegOutput])
            
            # hidden3 -> hidden4
            return fc(h3,w4_reg,bias4_reg,keepProb) 
# --------------------------------------------------------------------------- #