---
title: 文本分类之网格搜索+word2vec的词嵌入+xgboost
summary: 关键词： word2vec 词嵌入 xgboost 文本分类
author: foochane
top: false
cover: false
categories: NLP
date: 2019-09-14 18:46
urlname: 2019091401
tags:
  - 文本分类
---



```python
import codecs
import gensim
from sklearn import  preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from tqdm import tqdm
```

##  1 数据准备


```python
# 读取数据
labels = []
text = []
with codecs.open('output/data_clean_split.txt','r',encoding='utf-8') as f:
    document_split = f.readlines()
    for document in document_split:
        temp = document.split('\t')
        labels.append(temp[0])
        text.append(temp[1].strip())  

# 标签转换为数字
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# 将每个句子切分成单个词
text_s2w= [s.split() for s in text]
```

## 2 构建word2vec模型

### 2.1 训练word2vec模型



```python
model = gensim.models.Word2Vec(text_s2w,
                               min_count=5,
                               workers=6,
                               window =8,
                               size=100)
```

参数说明：

- min_count: 对于词频 < min_count 的单词，将舍弃（其实最合适的方法是用 UNK 符号代替，即所谓的『未登录词』，这里我们简化起见，认为此类低频词不重要，直接抛弃）

- workers: 可以并行执行的核心数，需要安装 Cython 才能起作用（安装 Cython 的方法很简单，直接 pip install cython）

size: 词向量的维度，神经网络隐层节点数

- window: 目标词汇的上下文单词距目标词的最长距离，很好理解，比如 CBOW 模型是用一个词的上下文预测这个词，那这个上下文总得有个限制，如果取得太多，距离目标词太远，有些词就没啥意义了，而如果取得太少，又信息不足，所以 window 就是上下文的一个最长距离

### 2.2 word2vec模型的简单使用
#### 2.2.1 构建词建词嵌入字典


```python

embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))
print('Found %s word vectors.' % len(embeddings_index))
```

    Found 87117 word vectors.


### 2.2.2 获取某个词的向量


```python

model['汽车']

```

    /home/ubuntu/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      """Entry point for launching an IPython kernel.

    array([-2.240292  , -1.1615268 , -1.4746077 ,  2.1054246 ,  4.819405  ,
           -3.1492457 , -0.05073776, -2.1645617 , -1.2719896 ,  1.7608824 ,
           -0.2626409 , -0.64887804,  1.3482507 ,  0.34045577,  1.4765079 ,
           -3.445696  ,  1.449008  , -0.09463242,  0.6401563 , -1.6335047 ,
           -0.30473268,  2.6725786 , -0.1342183 ,  0.27526513, -2.4943345 ,
            0.27751288, -1.9030106 , -0.2115223 ,  0.48280153,  2.8040369 ,
            1.4369518 , -1.6659547 ,  0.6498365 ,  3.1322846 , -1.7274039 ,
           -0.4276681 ,  2.0273833 , -1.2563524 , -2.2891238 ,  0.80385494,
           -0.8380016 , -1.1951414 ,  0.21576834, -1.8307697 ,  1.4016038 ,
           -0.07672032,  0.97227174,  1.3520627 ,  0.568014  , -1.914469  ,
           -1.1551676 ,  0.7751831 ,  0.7154037 ,  1.2694645 ,  1.9431589 ,
           -0.06259096,  3.4280195 ,  0.6663932 , -2.665189  ,  0.6598596 ,
           -0.07868402, -0.5291124 ,  1.8237985 , -0.7853107 , -0.16555293,
           -2.074671  , -0.87207425,  0.7680195 ,  0.40575528,  0.29356548,
           -2.8064344 , -2.5557816 , -1.554487  , -2.7589092 , -0.35392886,
           -0.6011241 , -0.31734776, -1.1346784 ,  0.1052264 ,  0.57027906,
            1.1536218 ,  2.066991  , -1.1962171 ,  1.0027347 ,  0.40441233,
            2.2641828 , -2.0621223 ,  2.0815525 ,  3.5621598 , -0.4967822 ,
           -0.717848  ,  3.1545784 ,  1.1730249 ,  1.3114505 , -0.36371502,
           -0.41231316, -2.3199863 , -0.10876293, -0.44529822, -2.18213   ],
          dtype=float32)



### 2.2.3 查看某个词的与其他词的相似度


```python

model.most_similar('人民日报')

```

    /home/ubuntu/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
      """Entry point for launching an IPython kernel.

    [('光明日报', 0.8604782223701477),
     ('海外版', 0.8062193393707275),
     ('年月日', 0.7948733568191528),
     ('经济日报', 0.7898619174957275),
     ('文汇报', 0.7830426692962646),
     ('社论', 0.7795723676681519),
     ('评论员', 0.765376091003418),
     ('中国作协', 0.7639801502227783),
     ('讲话', 0.7555620670318604),
     ('第五次', 0.7492089867591858)]



### 2.2.4 保存模型


```python

model.save('/tmp/w2v_model')

```

### 2.2.5 加载模型


```python

model_load = gensim.models.Word2Vec.load('/tmp/w2v_model')

```

## 3 训练数据处理




```python
#该函数会将语句转化为一个标准化的向量（Normalized Vector）
def sent2vec(s):
    """
    将每个句子转换会一个100的向量
    """
    words = s.split()
    M = []
    for w in words:
        try:
            #M.append(embeddings_index[w])
            M.append(model[w])
        except:
            continue
    M = np.array(M)  # shape=(x,100),x是句子中词的个数，100是每个词向量的维数
    v = M.sum(axis=0) # 维度是100，对M中的x个数求和，得到每一维度的总和
    if type(v) != np.ndarray: 
        return np.zeros(100)
    
    return v / np.sqrt((v ** 2).sum()) # 正则化，最后每个句子都变为一100维的向量
```


```python
# 对训练集和验证集使用上述函数，进行文本向量化处理
text_s2v = [sent2vec(s) for s in tqdm(text)]

# 转换成numpy array数组
text_s2v = np.array(text_s2v)

# 切分数据集
from sklearn.model_selection import train_test_split
x_train_w2v, x_valid_w2v, y_train, y_valid = train_test_split(text_s2v, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
```

      0%|          | 0/9249 [00:00<?, ?it/s]/home/ubuntu/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:11: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      # This is added back by InteractiveShellApp.init_path()
    100%|██████████| 9249/9249 [01:11<00:00, 129.79it/s]


## 4 调用模型进行分类


```python
# 定义损失函数
def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
```


```python
# 基于word2vec特征在一个简单的Xgboost模型上进行拟合
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
clf.fit(x_train_w2v, y_train)
predictions = clf.predict_proba(x_valid_w2v)

print ("logloss: %0.3f " % multiclass_logloss(y_valid, predictions))
```

    logloss: 0.367 

