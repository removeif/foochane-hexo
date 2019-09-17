---
title: 文本分类之TfidfVectorizer的使用
summary: 关键词：TfidfVectorizer的使用 TF-IDF TfidfVectorizer构建模型 TfidfVectorizer参数说明
author: foochane
top: false
cover: false
categories: NLP
date: 2019-09-02 09:53
urlname: 2019090201
tags:
  - 文本分类
---


## 1 构造文本


```python

text = ['机器学习是人工智能的一个分支。',
        '机器学习是对能通过经验自动改进的计算机算法的研究。',
        '机器学习是实现人工智能的一个途径，即以机器学习为手段解决人工智能中的问题。']
```

## 2 文本分词


```python
import jieba
# text_split = [''.join(i) for i in [jieba.cut(t) for t in text]]

text_split = []
jieba.enable_parallel(64) #并行分词开启
for t in text:
    tmp = jieba.cut(t) 
    tmp_split = [''.join(i) for i in tmp]
    split = ' '.join(i for i in tmp_split)
    text_split.append(split)
```

    Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Loading model cost 0.718 seconds.
    Prefix dict has been built succesfully.



```python

print(text_split)

```

    ['机器 学习 是 人工智能 的 一个 分支 。', '机器 学习 是 对 能 通过 经验 自动 改进 的 计算机 算法 的 研究 。', '机器 学习 是 实现 人工智能 的 一个 途径 ， 即以 机器 学习 为 手段 解决 人工智能 中 的 问题 。']


## 3 使用TfidfVectorizer构建模型

构建模型

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfvector = TfidfVectorizer()
model = tfvector.fit(text_split)

```
查看提取的特征词

```python

print("提取的特征词:" + str(model.get_feature_names()))

```

    提取的特征词:['一个', '人工智能', '分支', '即以', '学习', '实现', '手段', '改进', '机器', '研究', '算法', '经验', '自动', '解决', '计算机', '途径', '通过', '问题']


查看特征词和索引
```python

print("特征词和索引:" + str(model.vocabulary_))

```

    特征词和索引:{'机器': 8, '学习': 4, '人工智能': 1, '一个': 0, '分支': 2, '通过': 16, '经验': 11, '自动': 12, '改进': 7, '计算机': 14, '算法': 10, '研究': 9, '实现': 5, '途径': 15, '即以': 3, '手段': 6, '解决': 13, '问题': 17}


特征词的个数是18,对应的索引为0到17

## 4 获取tf-idf矩阵

查看tf-idf矩阵

```python

matrix = model.transform(text_split)

```

### 4.1 矩阵的shape

矩阵是3行18列，也就是有3个文档，每个文档有18个特征词


```python

print(matrix.shape)

```



    (3, 18)



### 4.2 查看矩阵的内容

这是个稀疏矩阵，如(0,8)表示第0个文档，第8个特征词(从0开始)的权重值


```python

print(matrix)

```

      (0, 8)	0.3495777539781596
      (0, 4)	0.3495777539781596
      (0, 2)	0.5918865885345992
      (0, 1)	0.45014500672563534
      (0, 0)	0.45014500672563534
      (1, 16)	0.3604298781233275
      (1, 14)	0.3604298781233275
      (1, 12)	0.3604298781233275
      (1, 11)	0.3604298781233275
      (1, 10)	0.3604298781233275
      (1, 9)	0.3604298781233275
      (1, 8)	0.21287569223847908
      (1, 7)	0.3604298781233275
      (1, 4)	0.21287569223847908
      (2, 17)	0.2925701011880934
      (2, 15)	0.2925701011880934
      (2, 13)	0.2925701011880934
      (2, 8)	0.3455932296344571
      (2, 6)	0.2925701011880934
      (2, 5)	0.2925701011880934
      (2, 4)	0.3455932296344571
      (2, 3)	0.2925701011880934
      (2, 1)	0.4450142061610019
      (2, 0)	0.22250710308050095


直接查看具体的矩阵


```python

print(matrix.todense())

```

    [[0.45014501 0.45014501 0.59188659 0.         0.34957775 0.
      0.         0.         0.34957775 0.         0.         0.
      0.         0.         0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.21287569 0.
      0.         0.36042988 0.21287569 0.36042988 0.36042988 0.36042988
      0.36042988 0.         0.36042988 0.         0.36042988 0.        ]
     [0.2225071  0.44501421 0.         0.2925701  0.34559323 0.2925701
      0.2925701  0.         0.34559323 0.         0.         0.
      0.         0.2925701  0.         0.2925701  0.         0.2925701 ]]


## 5 TfidfVectorizer的参数说明
- token_pattern ：可以添加正则表达式，如 `token_pattern=r"(?u)\b\w+\b")`可以匹配到单个子，如果为`r"(?u)\b\w\w+\b"`则会只匹配两个子以上的词。

- max_df：浮点数：[0.0,1.0]，如0.4，若某词语在的样本点中出现的概率超40%，则生成字典时剔除该词语；默认是1.0,即不剔除。
- min_df：整数：n。若某词语样本点中出现的次数小于n，生成字典时剔除该词语。默认是1，表明若词语只在1个以下文档中出现，剔除。
- max_features：整数：n。根据词语的TF-IDF权重降序排列，取前面n个最高值的词语组成词典。默认是None，即取全部词语。
- stop_words：指定停止词
- ngram_range: tuple(min_n, max_n) 要提取的n-gram的n-values的下限和上限范围，在min_n <= n <= max_n区间的n的全部值
- smooth_idf：boolean通过加1到文档频率平滑idf权重，为防止除零，加入一个额外的文档


```python
from sklearn.feature_extraction.text import TfidfVectorizer
stwlist = ['的','一个','是','。']
tfvector = TfidfVectorizer(min_df=1,  
                           max_df=0.5,
                           max_features=None,                 
                           ngram_range=(1, 2), 
                           use_idf=True,
                           smooth_idf=True,
                           stop_words = stwlist)
model = tfvector.fit(text_split)
```

