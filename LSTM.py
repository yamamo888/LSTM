# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 03:14:34 2019

@author: yu
"""
import os
import sys
import glob
import pickle
import pdb

import numpy as np
import tensorflow as tf

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import makingData as myData
import TrainingNN as NN


# --------------------------- command argment ------------------------------- #
# number of class
NUM_CLS = int(sys.argv[1])
# number of layer
depth = int(sys.argv[2])
# --------------------------------------------------------------------------- #

# ------------------------------- parameters -------------------------------- #
# if you want Evaluation == True
isEval = False
# if you are Windows user == True
isWindows = True

# number of cell
NUM_CELL = 5
# nankai index(=1,2)
NK1ind,NK2ind = 0,1
# tonankai index(=3,4)
TN1ind,TN2ind = 2,3
# tokai index(=5)
Tind = 4

# sequence for Regression evaluation
LEN_SEQ_EV = 8
# sequence for Regression test & train
LEN_SEQ = 8

# node of hidden for LSTM
NUM_HIDDEN = 128
# number of epocks
EPOCHES = 500
# number of training
NUM_STEPS = 200  

# minimum & maximum of paramter b in nankai 
NKMin, NKMax = 0.0125, 0.017
# in tonakai & tokai
TKTMin, TKTMax = 0.012, 0.0165 
# round decimal 
limitdecimal = 3
# Width class
beta = np.round((NKMax - NKMin)/NUM_CLS, limitdecimal)
# Center variable of the first class in nakai
NK_CENT = np.round(NKMin + (beta/2), limitdecimal)
# in tonakai & tokai
TKT_CENT = np.round(TKTMin + (beta/2), limitdecimal)

# size of batch
BATCH_SIZE = 4
# training rate
lr = 1e-3
# --------------------------------------------------------------------------- #

# ---------------------------------- path ----------------------------------- #
features = "features"
images = "images"
results = "results"

trainpklPath = "b2b3b4b5b6_Vb"
testpklPath = "test_Vb"
evalpklPath = "gt_Vb"

picklePath = "*.pkl"

# saved paramB (test & eval)
#predBpklPath = os.path.join(results)
# train pickle data path
trainfullPath = os.path.join(features,trainpklPath,picklePath)
# test pickle data path
testfullPath = os.path.join(features,testpklPath,picklePath)
# saved gt yV path
evalfullPath = os.path.join(features,evalpklPath,picklePath)

# all train data full path
trfiles = glob.glob(trainfullPath)
# all test data full path
tefiles = glob.glob(testfullPath)
# all evaluation data full path
efiles = glob.glob(evalfullPath)
# --------------------------------------------------------------------------- #

# ---------------------------- placeholder ---------------------------------- #
# input placeholder for LSTM
x = tf.placeholder(tf.float32, [None, None, NUM_CELL])
# sequence length for LSTM
seq = tf.placeholder(tf.int32, [None])
# input placeholder for Regress test & train
#x_reg = tf.placeholder(tf.float32, [None, NUM_HIDDEN])
# input placeholder for Regess evaluation
#x_reg_ev = tf.placeholder(tf.float32, [None, NUM_HIDDEN])
# output placeholder
y = tf.placeholder(tf.float32, [None,NUM_CELL])
# output placeholder class label
y_label = tf.placeholder(tf.int32, [None,NUM_CLS,NUM_CELL])
# --------------------------------------------------------------------------- #

# ---------------------------- Get test data -------------------------------- #
xTest, yTest, yTestLabel, yTestSeq = myData.GenerateTest(tefiles,isWindows=isWindows)
# --------------------------------------------------------------------------- #

# ---------------------------- Get eval data -------------------------------- #
# evaluation, xEval.shape=[number of data(=256),intervals(=8),cell(=3)]
xEval, yEvalSeq = myData.GenerateEval(efiles)
# xEval.shape=[256,8,"5"], nankai,tonankai,tokai -> nankai 2cell,tonankai 2cell,tokai
xEval = np.concatenate((xEval[:,:,0][:,:,np.newaxis],xEval[:,:,0][:,:,np.newaxis],xEval[:,:,1][:,:,np.newaxis],xEval[:,:,1][:,:,np.newaxis],xEval[:,:,2][:,:,np.newaxis]),2)
# xEval_REG.shape=[256,40(=4*5)]
xEval_REG = np.reshape(xEval,[xEval.shape[0],-1])
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def LSTM(x,seq,reuse=False):
    """
    LSTM Model.
    Args:
        x:input vector (3D)
    """
    # hidden layer for "LSTM"
    #cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN,forget_bias=1.0)
    #cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN,use_peepholes=True)

    with tf.variable_scope("LSTM") as scope:
        inkeepProb = 1.0
        outkeepProb = 1.0
        if reuse:
            inkeepProb = 1.0
            outkeepProb = 1.0
            scope.reuse_variables()
        
        # multi cell
        cells = []
        # 1st LSTM
        cell1 = tf.contrib.rnn.LSTMCell(NUM_HIDDEN,use_peepholes=True)
        # Dropout
        #cell2 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1,input_keep_prob=inkeepProb,output_keep_prob=outkeepProb)
        # 2nd LSTM
        cell2 = tf.contrib.rnn.LSTMCell(NUM_HIDDEN,use_peepholes=True)
        
        cells.append(cell1)
        cells.append(cell2)
        
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        
        # states tuple (Ct [None,128], Ht [None,128]) * 3 Ct is long memory, Ht is short memory time_major?
        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=seq)
        
        # shape=[BATCH_SIZE,LEN_SEQ,NUM_CELL] -> shape=[LEN_SEQ,BATCH_SIZE,NUM_CELL]
        #outputs = tf.transpose(outputs, perm=[1, 0, 2])
        # last of hidden, shape=[BATCH_SIZE,NUM_HIDDEN]
        
        # outputs [None,None,HIDDEN] 
        # states[-1] tuple (Ct [None,128], Ht [None,128])
        return outputs, states[-1]
# --------------------------------------------------------------------------- #
def CreateRegInputOutput(y,cls_score,scent):
    
    """
    Create input vector(=cls_center_x) & anchor-based method GT output(=r) for Regress with train & test data. 
    [arguments]
    x: feature vector, shape=[BATCH_SIZE,number of dimention]
    y: ground truth, shape=[BATCH_SIZE,NUM_CELL]
    cls_score: output in Classify (labeled y), shape=[BATCH_SIZE, NUM_CLS]
    scent: center variable of first class (nankai ≠ tonankai & tokai) 
    """
    # Max class of predicted class
    pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
    # Center variable of class        
    pred_cls_center = pred_maxcls * beta + scent
    # residual = objective - center variavle of class 
    r = tf.expand_dims(y,1) - pred_cls_center
    
    return pred_cls_center, r
# --------------------------------------------------------------------------- #
def CreateRegInput(cls_score,scent):
    """
    Create input vector(=cls_center_x) for Regress with evaluation data.
    [arguments]
    x: feature vector, shape=[BATCH_SIZE,number of dimention]
    cls_score: output in Classify (labeled y), shape=[BATCH_SIZE, NUM_CLS]
    scent: center variable of first class (nankai ≠ tonankai & tokai) 
    """
    # Max class of predicted class
    pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
    # Center variable of class        
    pred_cls_center = pred_maxcls * beta + scent
    
    return pred_cls_center
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def main():
    # ========================== LSTM  ====================================== #
    # hidden.shape=[BATCH_SIZE,64] for train
    outputs, hidden = LSTM(x,seq)
    # for test
    outputs_te, hidden_te = LSTM(x,seq,reuse=True)
    # for evaluation
    outputs_ev, hidden_ev = LSTM(x,seq,reuse=True)
    
    """
    # input for Classification
    nn_in = hidden 
    # for test
    nn_in_te = hidden_te
    # for evaluation
    nn_in_ev = hidden_ev
    
    
    """
    # input for Classification
    nn_in = hidden[-1] 
    # for test
    nn_in_te = hidden_te[-1]
    # for evaluation
    nn_in_ev = hidden_ev[-1]
    
    #pdb.set_trace()
    # ======================= Classification NN ============================= #
    # Classification NN for train
    pred_y1,pred_y2,pred_y3,pred_y4,pred_y5 = NN.Classify(nn_in,NUM_CLS=NUM_CLS)
    # for test
    pred_y1_te,pred_y2_te,pred_y3_te,pred_y4_te,pred_y5_te = NN.Classify(nn_in_te,NUM_CLS=NUM_CLS,reuse=True)
    # for evaluation
    pred_y1_ev,pred_y2_ev,pred_y3_ev,pred_y4_ev,pred_y5_ev= NN.Classify(nn_in_ev,NUM_CLS=NUM_CLS,reuse=True)
    
    # Loss function (Cross Entropy) train
    loss_cls1 = tf.losses.softmax_cross_entropy(y_label[:,:,NK1ind], pred_y1)
    loss_cls2 = tf.losses.softmax_cross_entropy(y_label[:,:,NK2ind], pred_y2)
    loss_cls3 = tf.losses.softmax_cross_entropy(y_label[:,:,TN1ind], pred_y3)
    loss_cls4 = tf.losses.softmax_cross_entropy(y_label[:,:,TN2ind], pred_y4)
    loss_cls5 = tf.losses.softmax_cross_entropy(y_label[:,:,Tind], pred_y5)
    # Loss function (Cross Entropy) test
    loss_cls1_te = tf.losses.softmax_cross_entropy(y_label[:,:,NK1ind], pred_y1_te)
    loss_cls2_te = tf.losses.softmax_cross_entropy(y_label[:,:,NK2ind], pred_y2_te)
    loss_cls3_te = tf.losses.softmax_cross_entropy(y_label[:,:,TN1ind], pred_y3_te)
    loss_cls4_te = tf.losses.softmax_cross_entropy(y_label[:,:,TN2ind], pred_y4_te)
    loss_cls5_te = tf.losses.softmax_cross_entropy(y_label[:,:,Tind], pred_y5_te)
    
    # all LSTM loss train
    loss_cls = loss_cls1 + loss_cls2 + loss_cls3 + loss_cls4 + loss_cls5
    # all LSTM loss test
    loss_cls_te = loss_cls1_te + loss_cls2_te + loss_cls3_te + loss_cls4_te + loss_cls5_te
    
    # optimizer
    trainer_cls = tf.train.AdamOptimizer(lr).minimize(loss_cls)
    # ======================================================================= #
    # OUT: pred_cls_cent: center variable of output LSTM, shape=
    # OUT: y_res: residual(=GT-predicted by LSTM), shape=
    # train
    pred_cls_cent1, y_r1 = CreateRegInputOutput(y[:,NK1ind],pred_y1,NK_CENT)
    pred_cls_cent2, y_r2 = CreateRegInputOutput(y[:,NK2ind],pred_y2,NK_CENT)
    pred_cls_cent3, y_r3 = CreateRegInputOutput(y[:,TN1ind],pred_y3,TKT_CENT)
    pred_cls_cent4, y_r4 = CreateRegInputOutput(y[:,TN2ind],pred_y4,TKT_CENT)
    pred_cls_cent5, y_r5 = CreateRegInputOutput(y[:,Tind],pred_y5,TKT_CENT)
    # test
    pred_cls_cent1_te, y_r1_te = CreateRegInputOutput(y[:,NK1ind],pred_y1_te,NK_CENT)
    pred_cls_cent2_te, y_r2_te = CreateRegInputOutput(y[:,NK2ind],pred_y2_te,NK_CENT)
    pred_cls_cent3_te, y_r3_te = CreateRegInputOutput(y[:,TN1ind],pred_y3_te,TKT_CENT)
    pred_cls_cent4_te, y_r4_te = CreateRegInputOutput(y[:,TN2ind],pred_y4_te,TKT_CENT)
    pred_cls_cent5_te, y_r5_te = CreateRegInputOutput(y[:,Tind],pred_y5_te,TKT_CENT)
    # evaluation
    pred_cls_cent1_ev = CreateRegInput(pred_y1_ev,NK_CENT)
    pred_cls_cent2_ev = CreateRegInput(pred_y2_ev,NK_CENT)
    pred_cls_cent3_ev = CreateRegInput(pred_y3_ev,TKT_CENT)
    pred_cls_cent4_ev = CreateRegInput(pred_y4_ev,TKT_CENT)
    pred_cls_cent5_ev = CreateRegInput(pred_y5_ev,TKT_CENT)
    
    
    # all center LSTM for train
    pred_cls_cent = tf.concat((pred_cls_cent1,pred_cls_cent2,pred_cls_cent3,pred_cls_cent4,pred_cls_cent5),1)
    # all center LSTM for test
    pred_cls_cent_te = tf.concat((pred_cls_cent1_te,pred_cls_cent2_te,pred_cls_cent3_te,pred_cls_cent4_te,pred_cls_cent5_te),1)
    # all center LSTM for evaluation
    pred_cls_cent_ev = tf.concat((pred_cls_cent1_ev,pred_cls_cent2_ev,pred_cls_cent3_ev,pred_cls_cent4_ev,pred_cls_cent5_ev),1)
    
    
    # all residual train
    y_r = tf.concat((y_r1,y_r2,y_r3,y_r4,y_r5),1)
    # all residual test
    y_r_te = tf.concat((y_r1_te,y_r2_te,y_r3_te,y_r4_te,y_r5_te),1)
    
    # ================== Regression NN ====================================== #
    # Regression networks for train
    pred_r = NN.Regress(nn_in,depth=depth,name_scope="Regress")
    # for test
    pred_r_te = NN.Regress(nn_in_te,reuse=True,depth=depth,name_scope="Regress")
    # fot evaluation
    pred_r_ev = NN.Regress(nn_in_ev,reuse=True,depth=depth,name_scope="Regress")
    
    # Loss function (MAE)
    loss_reg = tf.reduce_mean(tf.abs(y_r - pred_r))
    loss_reg_te = tf.reduce_mean(tf.abs(y_r_te - pred_r_te))
    
    # optimizer
    Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Regress")
    trainer_reg = tf.train.AdamOptimizer(lr).minimize(loss_reg,var_list=Vars)
    # ======================================================================= #
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # ======================================================================= #
    flag = False
    # start training
    for epoch in range(EPOCHES):
        for i in range(NUM_STEPS):
            
            batchX, batchY, batchYLabel, batchSeq = myData.nextBatch(BATCH_SIZE,trfiles,isWindows=isWindows)
            
            # =========================== train ============================= #
            # lstm
            trainLSTMOut, trainLSTMHidden = sess.run([outputs, hidden], feed_dict={x:batchX, seq:batchSeq})
            # classification
            _, trainClsLoss, trainClsCent = sess.run([trainer_cls, loss_cls, pred_cls_cent], feed_dict={x:batchX, y_label:batchYLabel, seq:batchSeq})
            #pdb.set_trace()
            # regression
            _, trainRegLoss, trainRes = sess.run([trainer_reg, loss_reg, pred_r], feed_dict={x:batchX, y:batchY, seq:batchSeq})
        # ================== test =========================================== #
        # lstm
        testLSTMOut, testLSTMHidden = sess.run([outputs_te, hidden_te], feed_dict={x:xTest, seq:yTestSeq})
        # classification
        testClsLoss, testClsCent = sess.run([loss_cls_te, pred_cls_cent_te], feed_dict={x:xTest, y_label:yTestLabel, seq:yTestSeq})
        # regression
        testRegLoss, testRes = sess.run([loss_reg_te, pred_r_te], feed_dict={x:xTest, y:yTest, seq:yTestSeq})
        # ================== evaluation ===================================== # 
        # lstm
        evalSTMOut, evalLSTMHidden = sess.run([outputs_ev, hidden_ev], feed_dict={x:xEval, seq:yEvalSeq})
        # classification
        evalClsCent = sess.run([pred_cls_cent_ev],feed_dict={x:xEval, seq:yEvalSeq})
        # regression
        evalRes = sess.run(pred_r_ev, feed_dict={x:xEval, seq:yEvalSeq})
        
        # predicted y
        trainPred = trainClsCent + trainRes
        testPred = testClsCent + testRes
        evalPred = evalClsCent + evalRes
        
        print("epoch %d, itr: %d" %  (epoch, i))
        print("trainClsLoss: %f, trainRegLoss: %f, testClsLoss: %f, testRegLoss: %f " % (trainClsLoss, trainRegLoss, testClsLoss, testRegLoss))
        print("==============================")
        print("trainMean: %f, trainVar: %f" % (np.mean(np.abs(batchY-trainPred)),np.var(np.abs(batchY-trainPred))))
        print("testMean: %f, testVar: %f" % (np.mean(np.abs(yTest-testPred)),np.var(np.abs(yTest-testPred))))
        print("==============================")
        print("truePredB",yTest[:4])
        print("testPredB",testPred[:4])
        print("evalPredB",evalPred[:4])
        
        # to save loss & predicted
        if not flag:
            trainClsLosses = trainClsLoss
            trainRegLosses = trainRegLoss
            testClsLosses = testClsLoss
            testRegLosses = testRegLoss
            flag = True
        else:
            trainClsLosses = np.hstack([trainClsLosses, trainClsLoss])
            trainRegLosses = np.hstack([trainRegLosses, trainRegLoss])
            testClsLosses = np.hstack([testClsLosses, testClsLoss])
            testRegLosses = np.hstack([testRegLosses, testRegLoss])
        
        # save predicted paramB (test & eval)
        with open(os.path.join(results,"{}_{}_{}.pkl".format(epoch,NUM_CLS,depth)),"wb") as fp:
            pickle.dump(yTest,fp)
            pickle.dump(trainPred,fp)
            pickle.dump(testPred,fp)
            pickle.dump(evalPred,fp)
#------------------------------------------------------------------------------
    # Plot Loss
    plt.title("Loss")
    plt.plot(np.arange(trainClsLosses.shape[0]),trainClsLosses,color="coral",linewidth=5.0,label="TrainCls")
    plt.plot(np.arange(testClsLosses.shape[0]),testClsLosses,color="dodgerblue",linewidth=5.0,label="TestCls")
    plt.plot(np.arange(trainRegLosses.shape[0]),trainRegLosses,color="m",linewidth=5.0,label="TrainReg")
    plt.plot(np.arange(testRegLosses.shape[0]),testRegLosses,color="c",linewidth=5.0,label="TestReg")
    plt.xlabel("epochs")
    plt.legend()
    
    plt.savefig(os.path.join(images,"Loss.png"))
    
if __name__ == "__main__":
    main()
