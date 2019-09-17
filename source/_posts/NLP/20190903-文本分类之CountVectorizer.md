---
title: 文本分类之CountVectorizer使用
summary: 关键词：CountVectorizer使用 CountVectorizer参数说明 CountVectorizer实例
author: foochane
top: false
cover: false
categories: NLP
date: 2019-09-03 12:53
urlname: 2019090301
tags:
  - 文本分类
---



CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数

## 1 CountVectorizer参数详解

```python
CountVectorizer(input='content', 
                encoding='utf-8',  
                decode_error='strict', 
                strip_accents=None, 
                lowercase=True, 
                preprocessor=None, 
                tokenizer=None, 
                stop_words=None, 
                dtype=<class 'numpy.int64'>)
```



一般要设置的参数是:ngram_range,max_df，min_df，max_features等，具体情况具体分析

>参数表

|参数|作用|
|:--:|:--:|
|input|一般使用默认即可，可以设置为"filename’或’file’|
|encodeing|	使用默认的utf-8即可，分析器将会以utf-8解码raw document|
|decode_error|	默认为strict，遇到不能解码的字符将报UnicodeDecodeError错误，设为ignore将会忽略解码错误，还可以设为replace，作用尚不明确|
|strip_accents|	默认为None，可设为ascii或unicode，将使用ascii或unicode编码在预处理步骤去除raw document中的重音符号|
|analyzer|	一般使用默认，可设置为string类型，如’word’, ‘char’, ‘char_wb’，还可设置为callable类型，比如函数是一个callable类型|
|preprocessor|	设为None或callable类型|
|tokenizer|	设为None或callable类型|
|ngram_range|	词组切分的长度范围，待详解|
|stop_words	|设置停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词，设为None且max_df的float，也可以设置为没有范围限制的int，默认为1.0。这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。如果参数中已经给定了vocabulary，则这个参数无效|
|min_df|	类似于max_df，不同之处在于如果某个词的document frequence小于min_df，则这个词不会被当作关键词|
|max_features	|默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集|
|vocabulary	|默认为None，自动从输入文档中构建关键词集，也可以是一个字典或可迭代对象？|
|binary	|默认为False，一个关键词在一篇文档中可能出现n次，如果binary=True，非零的n将全部置为1，这对需要布尔值输入的离散概率模型的有用的|
|dtype|	使用CountVectorizer类的fit_transform()或transform()将得到一个文档词频矩阵，dtype可以设置这个矩阵的数值类型|

>属性表

|属性|	作用|
|:--:|:--:|
|vocabulary_|	词汇表；字典型|
|get_feature_names()|	所有文本的词汇；列表型|
|stop_words_|	返回停用词表|

>方法表

|方法|	作用|
|--|--|
|fit_transform(X)|	拟合模型，并返回文本矩阵|
|fit(raw_documents[, y])|	Learn a vocabulary dictionary of all tokens in the raw documents.|
|fit_transform(raw_documents[, y])|	Learn the vocabulary dictionary and return term-document matrix.|


>用数据输入形式为列表，列表元素为代表文章的字符串，一个字符串代表一篇文章，字符串是已经分割好的。CountVectorizer同样适用于中文;


>CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在第i个文本下的词频。即各个词语出现的次数，通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。


## 2 简单例子


```python
from sklearn.feature_extraction.text import CountVectorizer

texts=["dog cat fish","dog cat cat","fish bird", 'bird'] # “dog cat fish” 为输入列表元素,即代表一个文章的字符串

#创建词袋数据结构
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)

#上述代码等价于下面两行
#cv.fit(texts)
#cv_fit=cv.transform(texts)
```

## 3 查看相关信息

所有文本的词汇


```python

print(cv.get_feature_names()) 

```

    ['bird', 'cat', 'dog', 'fish']


查看词汇表及索引


```python

print(cv.vocabulary_)   

```

    {'dog': 2, 'cat': 1, 'fish': 3, 'bird': 0}


查看特征矩阵


```python

print(cv_fit)

```

      (0, 3)	1
      (0, 1)	1
      (0, 2)	1
      (1, 1)	2
      (1, 2)	1
      (2, 0)	1
      (2, 3)	1
      (3, 0)	1



```python
print(cv_fit.toarray())
```

    [[0 1 1 1]
     [0 2 1 0]
     [1 0 0 1]
     [1 0 0 0]]


参考链接：https://blog.csdn.net/weixin_38278334/article/details/82320307


