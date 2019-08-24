# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:28:38 2019

@author: yu
"""

import os
import glob
import pickle
import shutil

import numpy as np
import random

from natsort import natsorted

import pdb
import time
# ----------- path ------------------------ #
# save log files path
logsPath = "logs"
dataPath = "b2b3b4b5b6_V"
# save features for NN
featurePath = "features"
# save yV & paramB
picklePath = "b2b3b4b5b6_Vb_tmp"
# save gt yV 
gtpicklePath = "gt_Vb"
# file relative path
fName = "*.txt"
pName = "*.pkl"
# ground truth Nankai megaquakes pkl data
gtName = "nankairireki.pkl"
# gt pickle full path
gtNankaiFullPath = os.path.join(featurePath,gtName)
logFullPath = os.path.join(logsPath,dataPath)
# saved B & yV path
picklefullPath = os.path.join(featurePath,picklePath)
# saved gt yV path
gtpicklefullPath = os.path.join(featurePath,gtpicklePath)
# pickles for test
testpicklePath = "test_Vb"
# ----------------------------------------- #

# --------- parameters -------------------- #
# number of cell
NUM_CELL = 8
# year index
yInd = 1
vInds = [2,3,4,5,6,7,8,9]
# number of class
NUM_CLS = 10
# nankai index(=1,2)
NK1ind,NK2ind = 0,1
# tonankai index(=3,4)
TN1ind,TN2ind = 2,3
# tokai index(=5)
Tind = 4
# sequence for Regression
LEN_SEQ = 8
#BATCH_SIZE = 4
# ----------------------------------------- #

# --------------------------------------------------------------------------- #
def LoadBU(fileName):
    """
    Load simulated lgos files parameter B(=taraget) and U(=explanatory)
    [argmument]
    fileName: logs file 
    """
    # load log data
    data = open(os.path.join(logFullPath,fileName)).readlines()
    # zero matrix for parameters(only frictional paramB), shape=[8,]
    paramB = np.zeros(NUM_CELL)
    
    for i in np.arange(1, NUM_CELL + 1):
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        paramB[i-1] = tmp[1]
        
    # start U line
    isRTOL = [True if data[i].count('value of RTOL')==1 else False for i in np.arange(len(data))]
    vInd = np.where(isRTOL)[0][0]+1
    
    # get U(=slip velocity)
    flag = False
    for i in np.arange(vInd,len(data)):
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        
        if not flag:
            V = tmp
            flag = True
        else:
            V = np.vstack([V,tmp])
            
    return paramB, V
# --------------------------------------------------------------------------- #
def convU2YearlyData(V,NUM_YEAR=0,sYear=0):
    """
    累積変位速度 U -> slip velosity yV(year)
    [argument]
    V: 累積変位速度
    sYear: first observation year
    """
    # zero matrix for year slip velocity, shape=[10000,8]
    yV = np.zeros([NUM_YEAR,NUM_CELL])
    #pdb.set_trace()
    # 観測データがない年には観測データの１つ前のデータを入れる(累積)
    for year in np.arange(sYear, NUM_YEAR):
        # 観測データがある場合
        if np.sum(np.floor(V[:,yInd]) == year):
            # 観測データがあるときはそのまま代入
            yV[int(year)] = V[np.floor(V[:, yInd]) == year, vInds[0]:]
        
        # 観測データがない場合
        else:
            # その1つ前の観測データを入れる
            yV[int(year)] = yV[int(year)-1,:]
    # change 累積変位速度 to slip velocity (year) 
    deltaV = yV[yInd:] - yV[:-yInd]
    # fitst + slip velocity (year)
    yV = np.concatenate((yV[np.newaxis,0],deltaV),0)
    yV = yV.T
    
    # nankai(=index[1,2]), tonankai(=index[3,4]), tokai(=index[5])
    yV = np.concatenate([yV[1:6,2000:]])
    # shape=[cell(5),obsyear(8000)]
    return yV
# --------------------------------------------------------------------------- #
def GetyVInterval(yV,fCnt=0,isEval=False):
    """
    Get earthquakes interval by slip velocity V (year) 
    Eval, Test & Train diff. (number of cell) 
    Args:
        yV: slip velosity (year)
        fCnt: for pickle file name (evaluatiton) 0 ~ 256
    """
    # more than 1 cm == slip
    SLIP = 1
    
    if isEval:
        
        nk_year = np.where(yV[0,:]>SLIP)[0]
        tn_year = np.where(yV[1,:]>SLIP)[0]
        t_year = np.where(yV[2,:]>SLIP)[0]
        
        nk_len = len(nk_year[1:] - nk_year[:-1])
        tn_len = len(tn_year[1:] - tn_year[:-1])
        t_len = len(t_year[1:] - t_year[:-1])
        
        nk_intervals = nk_year[1:] - nk_year[:-1]
        tn_intervals = tn_year[1:] - tn_year[:-1]
        t_intervals = t_year[1:] - t_year[:-1]
        
        # all sequence == 8 (longest)
        max_len = 8
        
        nk_intervals = np.pad(nk_intervals,[0,max_len-nk_len],"constant")
        tn_intervals = np.pad(tn_intervals,[0,max_len-tn_len],"constant")
        t_intervals = np.pad(t_intervals,[0,max_len-t_len],"constant")
        
        nankai_intervals = np.vstack([nk_intervals,tn_intervals,t_intervals])
        """
        # save intervals(explanatary) & param B & one-hot(target) vectors (one file)
        with open(os.path.join(gtpicklefullPath,"{}.pkl".format(fCnt)),"wb") as fp:
            pickle.dump(nankai_intervals,fp)"""
        
    else:
        # earthquakes year in nakai, shape=[earthquake-occur-year,]
        nk1_year = np.where(yV[NK1ind,:]>SLIP)[0]
        nk2_year = np.where(yV[NK2ind,:]>SLIP)[0]
        # earthquakes year in tonakai
        tn1_year = np.where(yV[TN1ind,:]>SLIP)[0]
        tn2_year = np.where(yV[TN2ind,:]>SLIP)[0]
        # earthquakes year in tokai
        t_year = np.where(yV[Tind,:]>SLIP)[0]
        
        # earthquakes intervals "length" in nankai, shape=[eqrthquakes intervals,]
        nk1_len = len(nk1_year[1:] - nk1_year[:-1])
        nk2_len = len(nk2_year[1:] - nk2_year[:-1])
        # in tonankai
        tn1_len = len(tn1_year[1:] - tn1_year[:-1])
        tn2_len = len(tn2_year[1:] - tn2_year[:-1])
        # in tokai
        t_len = len(t_year[1:] - t_year[:-1])
        
        # earthquakes intervals in nankai, shape=[eqrthquakes intervals,]
        nk1_intervals = nk1_year[1:] - nk1_year[:-1]
        nk2_intervals = nk2_year[1:] - nk2_year[:-1]
        # in tonankai
        tn1_intervals = tn1_year[1:] - tn1_year[:-1]
        tn2_intervals = tn2_year[1:] - tn2_year[:-1]
        # in tokai
        t_intervals = t_year[1:] - t_year[:-1]
        
        # sort intervals & max intervals
        max_len = np.sort([nk1_len,nk2_len,tn1_len,tn2_len,t_len])[::-1][0]
        # zero padding (fit to the longest cell length.) in nankai
        nk1_intervals = np.pad(nk1_intervals,[0,max_len-nk1_len],"constant")
        nk2_intervals = np.pad(nk2_intervals,[0,max_len-nk2_len],"constant")
        # in tonankai
        tn1_intervals = np.pad(tn1_intervals,[0,max_len-tn1_len],"constant")
        tn2_intervals = np.pad(tn2_intervals,[0,max_len-tn2_len],"constant")
        # in tokai
        t_intervals = np.pad(t_intervals,[0,max_len-t_len],"constant")
        
        # concate intervals in all cell
        nankai_intervals = np.vstack([nk1_intervals,nk2_intervals,tn1_intervals,tn2_intervals,t_intervals])
        # shape=[length maximum intervals(=?), number of cell(=5)]
        
    return nankai_intervals
# --------------------------------------------------------------------------- #
def AnotationB(paramB,intervals,fileName):
    """
    Annotate paramB,and Save intervals & annotate paramb(=one-hot vec.) for classify.
    Specify number of class(=NUM_CLS).
    [argument]
    Bs: frictional paramter B in all cell
    intervals: eqrthquakes intervals in all cell
    fileName: logs file 
    """
    # --------- parameters -------------------- #
    # first paramb in nankai
    sBn = 0.0125
    eBn = 0.017
    # first paramb in tonakai & 
    sB = 0.012
    eB = 0.0165
    # class width in nankai
    iBn = round((eBn - sBn)/NUM_CLS,6)
    # class in nankai
    Bsn = np.arange(sBn,eBn,iBn)
    # class width in tonankai & tokai
    iB = round((eB - sB)/NUM_CLS,6)
    # class in tonankai & tokai
    Bs = np.arange(sB,eB,iB)
    # index in all cell
    Bind = [1,2,3,4,5]
    # --------- parameters -------------------- #
    
    flag1,flag2 = False,False
    for bi in Bind:
        # onehot vector (all zero vector)
        oneHot = np.zeros(len(Bsn))
        ind = 0
        # create one-hot vector for classify label in nankai
        if bi < 3:
            for threB in Bsn:
                if (paramB[bi] >= threB) & (paramB[bi] < threB + iBn):
                    oneHot[ind] = 1
                ind += 1
            nks = oneHot
            # concate nankai 2 cell 
            if not flag1:
                nks_vec = nks
                flag1 = True
            else:
                nks_vec = np.vstack([nks_vec,nks])
        # in tonakai & tokai
        else:
            for threB in Bs:
                if (paramB[bi] >= threB) & (paramB[bi] < threB + iB):
                    oneHot[ind] = 1
                ind += 1
            tn_ts = oneHot
            # concate tonankai 2 cell & tokai
            if not flag2:
                tn_ts_vec = tn_ts
                flag2 = True
            else:
                tn_ts_vec = np.vstack([tn_ts_vec,tn_ts])
    
    # one-hot vectors for classify in all cell, shape=[length of maximum intervals(=?),number of cell(=5)]
    oneHot_vecs = np.vstack([nks_vec,tn_ts_vec]).T
    # paramer B in nankai, tonankai, tokai, shape=[cell(=5),]
    paramB = paramB[Bind[0]:6]
    # shape=[length of intervals(=?), number of cell(=5)]
    intervals = intervals.T
    # maximum of intervals
    max_interval = intervals.shape[0]
    
    # save intervals(explanatary) & param B & one-hot(target) vectors (one file)
    with open(os.path.join(picklefullPath,"{}_{}.pkl".format(max_interval,fileName)),"wb") as fp:
        pickle.dump(intervals,fp)
        pickle.dump(paramB,fp)
        pickle.dump(oneHot_vecs,fp)
# --------------------------------------------------------------------------- #
def SplitTrainTest():
    """
    Split train & test data (random).
    Save test data in features/test/*.txt.pkl
    """
    # load x,y pickle files
    files = glob.glob(os.path.join(picklefullPath,pName))
    
    # train rate
    TRAIN_RATIO = 0.8
    # number of all data
    NUM_DATA = len(files)
    # number of train data    
    NUM_TRAIN = np.floor(NUM_DATA * TRAIN_RATIO).astype(int)
    NUM_TEST = NUM_DATA - NUM_TRAIN
    
    # shuffle files list
    random.shuffle(files)
    
    # train data
    xTest = files[NUM_TRAIN:]
    
    # move test pickles to another directory
    for fi in xTest:
        #fName = fi.split("/")[2]
        fName = fi.split("\\")[2]
        shutil.move(fi,os.path.join(featurePath,testpicklePath,fName))
# --------------------------------------------------------------------------- #
def GenerateEval(files):
    """
    Create Evaluation data.
    ground truth Nankai megaquakes U (all).
    """
    
    # nankai index
    nkInd = 0
    
    # gt_Vb pickle list 
    #files = glob.glob(os.path.join(featurePath,gtpicklePath,pName))
    
    # sort nutural order
    efiles = []
    for path in natsorted(files):
        efiles.append(path)
    
    flag = False
    for fID in efiles:
        with open(fID,"rb") as fp:
            # [cell,seq]
            gt = pickle.load(fp)
            gt = gt.T
            
        # last time of non-zero intervals 
        nk = np.where(gt[:,nkInd] > 0)[0].shape[0] # nankai
        # get gt nankai length of intervals
        nankaiIntervals = GetyVInterval(gt,fCnt=0,isEval=True)
        
        if not flag:
            # int
            xEvalSeq = nk
            # shape=[number of intervals(=8),cell(=3)]
            xEval = nankaiIntervals.T[np.newaxis].astype(np.float32)
            flag = True
        else:
            xEvalSeq = np.hstack([xEvalSeq,nk])
            xEval = np.concatenate([xEval,nankaiIntervals.T[np.newaxis].astype(np.float32)],0)
    
    #xEval_REG = np.reshape(xEval,[xEval.shape[0],-1])
    
    # xEval.shape=[number of data(=256),intervals(=8),cell(3)]
    # xEval_REG.shape=[number of data(=256),intervals*cell(=24)]
    # xEvalSeq.shaoe=[number of data(=256), maximum of sequence]
    return xEval, xEvalSeq
# --------------------------------------------------------------------------- #
def GenerateTest(files,isWindows=False):
    """
    Create Test data.
    Args:    
        files: test pickle files path
    """
    # test pickle list 
    #files = glob.glob(os.path.join(featurePath,testpicklePath,pName))
    
    # sort nutural order
    tefiles = []
    for path in natsorted(files):
        tefiles.append(path)
    
    if isWindows:
        # max interval 
        max_interval =  int(tefiles[-1].split("\\")[-1].split("_")[0])
    else:
        max_interval =  int(tefiles[-1].split("/")[-1].split("_")[0])
        
    # get test data
    teX, teY, teY_label = ZeroPaddingX(tefiles,max_interval)
    
    # NG? wの大きさをそろえるために、evaluationの大きさにそろえるいくら長い系列長でもすべて8
    #teX_reg = teX[:,:LEN_SEQ,:]
    # input feature vector for Regression, shape=[BATCH_SIZE,LEN_SEQ*NUM_CELL]
    #teX_reg = np.reshape(teX_reg,[teX.shape[0],-1])
    
    # get length of intervals
    flag = False
    for file in tefiles:    
        if not flag:
            if isWindows:
                testSeq = int(file.split("\\")[-1].split("_")[0])
            else:
                testSeq = int(file.split("/")[-1].split("_")[0])
            flag = True
        else:            
            if isWindows:
                testSeq = np.hstack([testSeq,int(file.split("\\")[-1].split("_")[0])])
            else:
                testSeq = np.hstack([testSeq,int(file.split("/")[-1].split("_")[0])])
    
    return teX, teY, teY_label, testSeq
# --------------------------------------------------------------------------- #
def nextBatch(BATCH_SIZE,BATCH_CNT,files,isWindows=False):
    """
    Extraction minibatch train data.
    [process]
    1. Sort near length of intervals
    """
    # train pickle files (comment out called by LSTM_Cls.py)
    #files = glob.glob(os.path.join(picklefullPath,pName))
    
    # sort nutural order
    trfiles = []
    for path in natsorted(files):
        trfiles.append(path)
    
    # suffle start index & end index 
    sInd = BATCH_CNT * BATCH_SIZE
    eInd = sInd + BATCH_SIZE
    
    # batch files
    bfiles = trfiles[sInd:eInd]
    # length intervals of last batch files
    if isWindows:
        # max interval 
        max_interval =  int(bfiles[-1].split("\\")[-1].split("_")[0])
    else:
        max_interval =  int(bfiles[-1].split("/")[-1].split("_")[0])
    
    # IN: batch files & length of max intervals, OUT: batchX, batchY
    batchX, batchY, batchY_label = ZeroPaddingX(bfiles,max_interval)
    
    
    # get length of intervals
    flag = False
    for file in bfiles:
        if not flag:
            if isWindows:
                batchSeq = int(file.split("\\")[-1].split("_")[0])
            else:
                batchSeq = int(file.split("/")[-1].split("_")[0])
            flag = True
        else:            
            if isWindows:
                batchSeq = np.hstack([batchSeq,int(file.split("\\")[-1].split("_")[0])])
            else:
                batchSeq = np.hstack([batchSeq,int(file.split("/")[-1].split("_")[0])])
    
    return batchX, batchY, batchY_label, batchSeq
# --------------------------------------------------------------------------- #
def ZeroPaddingX(files,max_interval):
    """
    [process]
    1. Fit to maximum intervals of minibatch data.
    2. Fill with zero for missing length of intervals.
    3. OUT: X, Y, labeled Y 
    [argment]
    files: mini-batch data list (fullPath)
    max_intervals: length of intervals for fitting
    """
    
    flag = False
    for fi in files:
        # load pickle files for mini-batch 
        # X: intervals, Y: param B, Y_label: anotation of param B 
        with open(fi,"rb") as fp:
            X = pickle.load(fp)
            Y = pickle.load(fp)
            Y_label = pickle.load(fp)
        
        # zero matrix for zero padding, shape=[max_interval,cell(=5)]
        zeros = np.zeros((max_interval,X.shape[1]))
        
        # 1. zero padding (fit to the longest batch file length
        zeros[:X.shape[0],:] = X
        
        if not flag:
            Xs = zeros[np.newaxis].astype(np.float32)
            Ys = Y[np.newaxis].astype(np.float32)
            Ys_label = Y_label[np.newaxis].astype(np.int32)
            flag = True
            
        else:
            Xs = np.concatenate([Xs,zeros[np.newaxis].astype(np.float32)],0)
            Ys = np.concatenate([Ys,Y[np.newaxis].astype(np.float32)],0)
            Ys_label = np.concatenate([Ys_label,Y_label[np.newaxis].astype(np.int32)],0)
    
    return Xs, Ys, Ys_label
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    
    # if you use Windows -> True
    isWindows = True
    
    # ----------- path ------------------------ #
    # logs (by simulation) files path
    filePath = os.path.join(logsPath,dataPath,fName)
    # all logs files path
    files = glob.glob(filePath)
    # ----------------------------------------- #
    """
    for fID in np.arange(len(files)):
        # load logs file
        if isWindows:
            file = files[fID].split("\\")[2]
        else:
            file = files[fID].split("/")[2]
        print(file)
        
        # Load paramB, V in all cell
        Bs, Vs = LoadBU(file)
        sYear = np.floor(Vs[0, yInd])
        # Change V to yV (obs of 10000 year)
        yV = convU2YearlyData(Vs,NUM_YEAR=10000,sYear=sYear) 
        # Get interval year of slip velocity yV (year)
        intervals_len = GetyVInterval(yV)
        # Anotate B in all cell
        AnotationB(Bs,intervals_len,file)
    """
    # ----------------------------------------- #
    """
    # move pickle files 
    pickles = glob.glob(os.path.join(picklefullPath,pName))
    # moved files path
    newpicklepath = os.path.join(featurePath,"b2b3b4b5b6_Vb")
    
    for file in pickles:
        # need full path ?
        if isWindows:    
            shutil.move(os.path.join(picklefullPath,file.split("\\")[2]), os.path.join(newpicklepath,file.split("\\")[2]))
        else:
            shutil.move(os.path.join(picklefullPath,file.split("/")[2]), os.path.join(newpicklepath,file.split("/")[2]))
    # ----------------------------------------- #
    """     
    # Split train & test data
    #GenerateEval()
    #GenerateTest()
    #SplitTrainTest(isWindows=isWindows)
    
    
    #nextBatch(BATCH_SIZE=3,isWindows=isWindows)