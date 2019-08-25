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
NUM_HIDDEN4 = 128 # node of 3 hidden

# regression
nRegHidden = 128 # node of 1 hidden
nRegHidden2 = 128 # node of 2 hidden
nRegHidden3 = 128 # node of 3 hidden
nRegHidden4 = 128 # node of 4 hidden
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
        
        # 1st
        w1_cls = weight_variable("w1_cls",[NUM_HIDDEN1, NUM_HIDDEN2])
        bias1_cls = bias_variable("bias1_cls",[NUM_HIDDEN2])
        h1 = fc_relu(x,w1_cls,bias1_cls,keepProb)
        
        # 2nd
        w2_cls = weight_variable("w2_cls",[NUM_HIDDEN2, NUM_HIDDEN3])
        bias2_cls = bias_variable("bias2_cls",[NUM_HIDDEN3])
        h2 = fc_relu(h1,w2_cls,bias2_cls,keepProb)
        
        w3_cls = weight_variable("w3_cls",[NUM_HIDDEN3, NUM_HIDDEN4])
        bias3_cls = bias_variable("bias3_cls",[NUM_HIDDEN4])
        h3 = fc_relu(h2,w3_cls,bias3_cls,keepProb)
        
        # 3rd n1
        w4_1_cls = weight_variable("w4_1_cls",[NUM_HIDDEN4, NUM_CLS])
        bias4_1_cls = bias_variable("bias4_1_cls",[NUM_CLS])
        
        # 3rd n2
        w4_2_cls = weight_variable("w4_2_cls",[NUM_HIDDEN4, NUM_CLS])
        bias4_2_cls = bias_variable("bias4_2_cls",[NUM_CLS])
        
        # 3rd tonankai1
        w4_3_cls = weight_variable("w4_3_cls",[NUM_HIDDEN4, NUM_CLS])
        bias4_3_cls = bias_variable("bias4_3_cls",[NUM_CLS])
        
        """
        # 3rd in tonankai2
        w4_4_cls = weight_variable("w4_4_cls",[NUM_HIDDEN4, NUM_CLS])
        bias4_4_cls = bias_variable("bias4_4_cls",[NUM_CLS])
        
        # 3rd tokai
        w4_5_cls = weight_variable("w4_5_cls",[NUM_HIDDEN4, NUM_CLS])
        bias4_5_cls = bias_variable("bias4_5_cls",[NUM_CLS])
        """ 
        # shape=[BATCH_SIZE,NUM_CLS]
        y1 = fc_sigmoid(h3,w4_1_cls,bias4_1_cls,keepProb)
        y2 = fc_sigmoid(h3,w4_2_cls,bias4_2_cls,keepProb)
        y3 = fc_sigmoid(h3,w4_3_cls,bias4_3_cls,keepProb)
        #y4 = fc_sigmoid(h3,w4_4_cls,bias4_4_cls,keepProb)
        #y5 = fc_sigmoid(h3,w4_5_cls,bias4_5_cls,keepProb)
        
        return y1,y2,y3
# --------------------------------------------------------------------------- #
def Regress(x,NUM_OUT,reuse=False,name_scope="Regress"):
    
    """
    Fully-connected regression networks.
    [arguments]
    x: input data (feature vector or residual, shape=[None, LEN_SEQ*NUM_CELL])
    reuse=False: Train, reuse=True: Evaluation & Test (variables sharing)
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
        # 1st
        w1_reg = weight_variable('w1_reg',[nRegInput,nRegHidden])
        bias1_reg = bias_variable('bias1_reg',[nRegHidden])
        h1 = fc_relu(x,w1_reg,bias1_reg,keepProb)
        
        # 2nd
        w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
        bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
        h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
        
        # 3rd
        w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegHidden3])
        bias3_reg = bias_variable('bias3_reg',[nRegHidden3])
        h3 = fc_relu(h2,w3_reg,bias3_reg,keepProb)
        
        # 4th
        w4_reg = weight_variable('w4_reg',[nRegHidden3,NUM_OUT])
        bias4_reg = bias_variable('bias4_reg',[NUM_OUT])
        
        return fc(h3,w4_reg,bias4_reg,keepProb) 
# --------------------------------------------------------------------------- #
