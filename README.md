# LSTM


### 1. データの作成 : `makingData.py` 
**※1 dummy.txt が読み込まれてしまう恐れあり**
**※2 Windowsユーザーは`if __name__ == "__main__":`を`isWindows=True`にしないと、ファイル読み込めません**
ある地震が発生した時の摩擦パラメータの組み合わせにより発生する地震間隔の個数は (例えば南海は3回の地震間隔、東南海は5回)、最長の地震間隔の個数に合わせる (例えば南海は**5**回の地震間隔、東南海は5回)。
真値の南海トラフ巨大地震履歴の地震間隔の個数はすべて**8**に合わせる


- シミュレーションデータの地震間隔データの保存 
  - features/b2b3b4b5b6_Vb
  
  - コード
  ```python: AnotationB
  ```
  
- 真の南海トラフの地震間隔データの保存
   -  features/gt_Vb
   - git clone で取得可能
   
- コード
  - 地震間隔の個数 `xEval、Regressionの入力 `xEval_REG`、ゼロパディングしないシーケンスの個数 `xEvalSeq` (例えば、全体では8に合わせるが、6までの場合は、`xEvalSeq` = 6)
  - makingData.py で試したい場合は引数 `files=0`を指定し、`files = glob...` のコメントと`if __name__ == "__main__":`の`GenerateTest()`を外す
  - データごとの系列長に合わせた地震間隔データは、`gt_Vb`ディレクトリに保存済み

 - 評価データ作成
``` python: GenerateEval
def GenerateEval(files):
    ...
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
            xEval = nankaiIntervals.T[np.newaxis]
            flag = True
        else:
            xEvalSeq = np.hstack([xEvalSeq,nk])
            xEval = np.vstack([xEval,nankaiIntervals.T[np.newaxis]])
    
    xEval_REG = np.reshape(xEval,[xEval.shape[0],-1])
    
    return xEval, xEval_REG, xEvalSeq

```
   - テストデータ作成

``` python: GenerateTest(files):
def GenerateTest(files):
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
        
    # get test data
    teX, teY, teY_label = ZeroPaddingX(tefiles,max_interval)
    
    return teX, teY, teY_label, testSeq
```
  - 学習データ取得 : mini-batch

``` python : nextbatch
def nextBatch(BATCH_SIZE,files,isWindows=False):
    ...
    # train pickle files (comment out called by LSTM_Cls.py)
    files = glob.glob(os.path.join(picklefullPath,pName))
    
    # sort nutural order
    trfiles = []
    for path in natsorted(files):
        trfiles.append(path)
    
    # suffle start index & end index 
    sInd = np.random.permutation(len(trfiles)-BATCH_SIZE)[0]
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
    
    # NG
    #batchX_reg = batchX[:,:LEN_SEQ,:]
    # input feature vector for Regression, shape=[BATCH_SIZE,LEN_SEQ*NUM_CELL]
    #batchX_reg = np.reshape(batchX_reg,[batchX.shape[0],-1])   
    
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
```

- 引数
  - files : 読み込みたいファイルパスのリスト ex) ["test_Vb/38_*.pkl"]
  - isWindows : Windows ユーザはTrue (default はFalse)


- Zero padding
  - セルの最大地震間隔の個数に合わせる (例えば南海は3 回の地震間隔、東南海は5 回の場合は5 回の個数に合わせる。南海は2 回分zeroで埋まる)
  
``` python: ZeroPaddingX
```
  

    
### 2. 特徴量の作成 (LSTM の実行) `LSTM.py`

## コマンド
  - 引数: クラス数 `NUM_CLS` は Classify で使用 (10, 20)、 層数 `depth`　は Regress で使用 (3,4,5)
  - 例： 10クラスと3層を用いる場合 ``` python LSTM.py 10 3 ```

## LSTM 本体

- placeholder
  - x : LSTM の入力 [バッチサイズ,系列長,セル]
  - sq : LSTM のバッチのシーケンスの長さ [シーケンスの長さ] 
  
``` python: LSTM.py
# input placeholder for LSTM
x = tf.placeholder(tf.float32, [None, None, NUM_CELL])
# sequence length for LSTM
sq = tf.placeholder(tf.int32, [None])
# input placeholder for Regress test & train
x_reg = tf.placeholder(tf.float32, [None, LEN_SEQ*NUM_CELL])
# input placeholder for Regess evaluation
x_reg_ev = tf.placeholder(tf.float32, [None, LEN_SEQ_EV*NUM_CELL])
# output placeholder
y = tf.placeholder(tf.float32, [None,NUM_CELL])
# output placeholder class label
y_label = tf.placeholder(tf.int32, [None,NUM_CLS,NUM_CELL])
```

- 多層のLSTM `cell1 = ...LSTMCell(hidden)`と`cell2 = ...LSTMCell(hidden)` のcellをconcatし、`cell = ...MultiRNNCell(cells(cell1+cell2))`に入れて実装
- `tf.nn.dynamic_rnn()`の引数、**`sequence_length`はzero padding を無視できる(最大系列長に合わせるとzeroの方が多くなるデータもある。)** 
- `tf.nn.dynamic_rnn()`の出力`outputs`には、結果、`states`にはタプルで、1つ前の状態(短記憶ht)とずっと前の状態(長記憶Ct)である


``` python: LSTM.py
def LSTM(x,seq,reuse=False):
    """
    LSTM Model.
    Args:
        x:input vector (3D)
    """
    # hidden layer for "LSTM"
    cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN,forget_bias=1.0)
    
    with tf.variable_scope("LSTM") as scope:
        if reuse:
            scope.reuse_variables()
        
        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=seq, time_major=False)
        # shape=[BATCH_SIZE,LEN_SEQ,NUM_CELL] -> shape=[LEN_SEQ,BATCH_SIZE,NUM_CELL]
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        # last of hidden, shape=[BATCH_SIZE,NUM_HIDDEN]
        state = states[-1]
        
        return outputs, state
```
- 引数
  - x: 入力ベクトル [バッチ数、系列長、セル(5)]　※シミュレーションの方に合わせる
  
