---
title: 文本分类之TF-IDF的SVD特征+xgboost
summary: 关键词： TF-IDF SVD特征 xgboost 文本分类
author: foochane
top: false
cover: false
categories: NLP
date: 2019-09-11 16:46
urlname: 2019091101
tags:
  - 文本分类
---


## 1 数据准备


```python


# 1 导入数据
labels = []
text = []
with codecs.open('output/data_clean_split.txt','r',encoding='utf-8') as f:
    document_split = f.readlines()
    for document in document_split:
        temp = document.split('\t')
        labels.append(temp[0])
        text.append(temp[1].strip())  

# 2 标签转换为数字
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


# 3 TF-IDF提取文本特征
tfv1 = TfidfVectorizer(min_df=4,  
                       max_df=0.6)
tfv1.fit(text)
features = tfv1.transform(text)


# 4 切分数据集
from sklearn.model_selection import train_test_split
x_train_tfv, x_valid_tfv, y_train, y_valid = train_test_split(features, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
```

## 2 数据标准化
由于SVM需要花费大量时间，因此在应用SVM之前，我们将使用奇异值分解（Singular Value Decomposition ）来减少TF-IDF中的特征数量。

同时，在使用SVM之前，我们还需要将数据标准化（Standardize Data ）


```python
#使用SVD进行降维，components设为120，对于SVM来说，SVD的components的合适调整区间一般为120~200 
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(x_train_tfv)
xtrain_svd = svd.transform(x_train_tfv)
xvalid_svd = svd.transform(x_valid_tfv)

#对从SVD获得的数据进行缩放
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)
```

## 3 定义损失函数


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

## 4 使用模型分类

### 4.1 基于tf-idf的svd特征，使用xgboost


```python
clf = xgb.XGBClassifier(nthread=10)
clf.fit(xtrain_svd, y_train)
predictions = clf.predict_proba(xvalid_svd)

print ("logloss: %0.3f " % multiclass_logloss(y_valid, predictions))
```

    logloss: 0.385 


### 4.2 对经过数据标准化(Scaling)的tf-idf-svd特征使用xgboost


```python
clf = xgb.XGBClassifier(nthread=10)
clf.fit(xtrain_svd_scl, y_train)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(y_valid, predictions))
```

    logloss: 0.385 

