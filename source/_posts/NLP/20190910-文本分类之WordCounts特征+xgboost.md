---
title: 文本分类之WordCounts特征+xgboost
summary: 关键词： WordCounts特征  xgboost 文本分类
author: foochane
top: false
cover: false
categories: NLP
date: 2019-09-10 21:46
urlname: 2019091001
tags:
  - 文本分类
---




```python
import codecs 
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
```

## 1 导入数据


```python

labels = []
text = []
with codecs.open('output/data_clean_split.txt','r',encoding='utf-8') as f:
    document_split = f.readlines()
    for document in document_split:
        temp = document.split('\t')
        labels.append(temp[0])
        text.append(temp[1].strip())  


```

## 2 标签转换为数字


```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
```

## 3 提取的word counts特征


```python
ctv = CountVectorizer(min_df=3,
                      max_df=0.5,
                      ngram_range=(1,2))

# 使用Count Vectorizer来fit训练集和测试集（半监督学习）
ctv.fit(text)
text_ctv = ctv.transform(text)
```

## 4 切分数据集


```python
from sklearn.model_selection import train_test_split
x_train_ctv, x_valid_ctv, y_train, y_valid = train_test_split(text_ctv, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
```

## 5定义损失函数


```python
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

## 5 使用模型分类


```python
# 基于word counts特征，使用xgboost
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(x_train_ctv, y_train)
predictions = clf.predict_proba(x_valid_ctv)

print ("logloss: %0.3f " % multiclass_logloss(y_valid, predictions))
```

    logloss: 0.184 

