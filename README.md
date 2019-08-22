# LSTM


## 1. データの作成 : `makingData.py` 
**※1 dummy.txt が読み込まれてしまう恐れあり**　<br>
**※2 Windowsユーザーは`if __name__ == "__main__":`を`isWindows=True`にしないと、ファイル読み込めません** <br>
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

<br>


### データの受け渡し : `makingData.py`

- コード
  - データを分けて作成したので、1つのディレクトリにまとめる必要がありました。(使わなくてもok)
``` python
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
  ```

<br>


### 学習テストデータに分割
- ひとつにまとめられた `b2b3b4b5b6_Vb`ディレクトリから、テストデータを格納する `test_Vb`に分割する


``` python
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
```

<br>

### Zero padding : `makingData.py`

- `if int(fi.split("\\")[-1].split("_")[0]) < max_interval:` : windowsユーザとlinuxユーザで手動切り替え 
- `max_interval`分のzero 行列を用意して
- `X`の分だけzero行列に数値を埋める、他は0

<br>

- コード
``` python
def ZeroPaddingX(files,max_interval):
    ...
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
```


<br>


***
    
## 2. 特徴量の作成 (LSTM の実行) `LSTM.py`

### コマンド
  - 引数: クラス数 `NUM_CLS` は Classify で使用 (10, 20)、 層数 `depth`　は Regress で使用 (3,4,5)
  - 例： 10クラスと3層を用いる場合 ``` python LSTM.py 11 3 ```
  - **クラス数は、試したいクラス + 1　にする**

### LSTM 本体

- placeholder
  - x : LSTM の入力 [バッチサイズ,系列長,セル] ※tf.float32指定 (`tf.nn.dynamic`が動かなくなるので)
  - sq : LSTM のバッチのシーケンスの長さ [シーケンスの長さ] 
  - y : 真値
  - y_label : クラスラベル[バッチサイズ, クラス数, セル]  0 or 1 (計算グラフで分割)
  
<br>

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
  
- `main()` で呼び出し

``` python : LSTM.py
    # hidden.shape=[BATCH_SIZE,64] for train
    outputs, hidden = LSTM(x,sq)
    # for test
    outputs_te, hidden_te = LSTM(x,sq,reuse=True)
    # for evaluation
    outputs_ev, hidden_ev = LSTM(x,sq,reuse=True)
```


  <br>
  
 - Regression用の真値(残差)と~~入力~~を作成 ※入力は使わない
  - `CreateRegInputOutput`は学習とテスト用
  - `CreateRegInput`は評価用
 
``` python : LSTM.py
def CreateRegInputOutput(y,cls_score,scent):
    ...
    # Max class of predicted class
    pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
    # Center variable of class        
    pred_cls_center = pred_maxcls * beta + scent
    # residual = objective - center variavle of class 
    r = tf.expand_dims(y,1) - pred_cls_center
    
    return pred_cls_center, r
```

``` python : LSTM.py
def CreateRegInput(cls_score,scent):
    ...
    # Max class of predicted class
    pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
    # Center variable of class        
    pred_cls_center = pred_maxcls * beta + scent
    
    return pred_cls_center
 
``` 
 
- Classification用の入力
  -

``` python : LSTM.py
    # input for Classification
    cls_in = hidden[-1] 
    # for test
    cls_in_te = hidden_te[-1]
    # for evaluation
    cls_in_ev = hidden_ev[-1]
```

