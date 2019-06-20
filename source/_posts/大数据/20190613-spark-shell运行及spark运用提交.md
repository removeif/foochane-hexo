---
title: spark-shell运行及spark运用提交
summary: 关键词： spark-shell spark-submit 程序运行 运用提交 运用部署
date: 2019-06-14 14:27:28
urlname: 2019061401
categories: 大数据
tags:
  - 大数据
  - spark
img: 
author: foochane
toc: true
mathjax: false
top: true
cover: false
---

>spark-shell spark-submit 程序运行 运用提交 运用部署

<!-- 
文章作者：[foochane](https://foochane.cn/) 
</br>
原文链接：[https://foochane.cn/article/2019061401.html](https://foochane.cn/article/2019061401.html)  
-->




## 1 spark-shell的使用
### 1.1 连接到本地

连接到本地使用命令： `./bin/spark-shell --master local[*]`
这里的`local[*]`指的是cpu核心数（cores），可以在web界面查看与他的核心数，如`spark-shell --master local[2]`.

另外，执行` spark-shell` 其实也就是相当于执行 `spark-shell --master local[*]`

启动spark-shell 
```bash
hadoop@Master:~$ cd $SPARK_HOME
hadoop@Master:/usr/local/bigdata/spark-2.4.3$ ./bin/spark-shell
19/06/14 01:55:17 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://Master:4040
Spark context available as 'sc' (master = local[*], app id = local-1560477340912).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.3
      /_/

Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_211)
Type in expressions to have them evaluated.
Type :help for more information.
```



### 1.2 连接到集群
连接到集群使用如下命令
```
spark-shell --master spark://master:7077 
```

具体如下：

```bash
$ spark-shell --master spark://master:7077
19/06/14 05:39:22 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://Master:4040
Spark context available as 'sc' (master = spark://master:7077, app id = app-20190614053947-0006).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.3
      /_/

Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_211)
Type in expressions to have them evaluated.
Type :help for more information.
```



**注意：**

**spark-shell如果是连接到集群，那么可以在web页面 [http://Master:8080](http://Master:8080)中的`Running Applications`里查看到相关的内容，而连接到本地则查看不了，另外，连接到集群是访问的数据必须是来自hdfs，无法访问到本地数据**

### 1.3 其他参数
除了指定`master`以外，还可以指定内存，核心数等参数

```bash
spark-shell \
--master spark://master:7077 \
--executor-memory 512m \
--total-executor-cores 2 
```




## 2 测试程序
### 2.1 数据操作
执行`spark-shell`的时候会有点慢，因为`spark-shell`在创建`SparkContext`和`SparkSession`这两个对象，这样后面才可以在`shell`中直接使用`spark` 和 `sc` 对象。

下面进行简单统计测试：

先使用`spark.read.textFile`来读取文件，然后进行统计和相关操作。

textFile默认是从hdfs读取文件，格式为：`hdfs://Master:9000/user/hadoop/xxx`
也可以读取本地文件，格式为：`file:///home/user/spark/xxxx`

```shell
scala>val textFile = spark.read.textFile("file:///usr/local/bigdata/spark-2.4.3/README.md")
textFile: org.apache.spark.sql.Dataset[String] = [value: string]
scala> textFile.count() // Number of items in this Dataset
res0: Long = 105

scala> textFile.first() // First item in this Dataset
res1: String = # Apache Spark

scala> val linesWithSpark = textFile.filter(line => line.contains("Spark"))
linesWithSpark: org.apache.spark.sql.Dataset[String] = [value: string]

scala> textFile.filter(line => line.contains("Spark")).count() // How many lines contain "Spark"?
res2: Long = 20

```

### 2.2 缓存
`spark`可以进行缓存，避免重复计算。

先执行如下操作
```
scala> val textFile = spark.read.textFile("file:///usr/local/bigdata/spark-2.4.3/README.md")
textFile: org.apache.spark.sql.Dataset[String] = [value: string]

scala> textFile.cache()
res0: textFile.type = [value: string]

scala> textFile.count() //第一次count
res1: Long = 105

scala> textFile.count() //第二次count
res2: Long = 105
```

可以发现运行两次`count`的时间是不一样的，第一次运行，需要花费一段实际，而第二次运行很快就出结果了。（读取一个大一点的文件效果会更明显一点）

在这里`spark.read.textFile`和`textFile.cache()`都是延迟执行的，也就是只是记住了这个操作，没有正在去读取，到执行第一次`textFile.count()`的时候才真正去执行，到了第二次执行`textFile.count()`的时候`spark`并没有再次从磁盘读取数据进行计算，所以运行很快，这也是`spark`可以避免重复计算的原因。



## 3 提交运用程序

运用程序写好后可以使用`spark-submit `来提交运行。`spark-submit `脚本负责设置`Spark`及其依赖项的类路径,并可以支持不同的集群管理器,并部署Spark支持的部署模式。

命令提交，具体如下
```bash
spark-submit \
--class  要运行的程序（类） \
--master 指定spar master，集群：spark://Master:7077 ，本地：local[*]\
--driver-memory 指定驱动程序的内存导向 \
--executor-memory 进程内存，指的是每个executor用的内存 \
--total-excutor-cores 指定程序用几个cpu核来运行 \
指定jar包（包含运行程序的类）\
[运行程序所需的参数]
```

下面运行测试用例`SparkPi`,`SparkPi`可以直接使用`run-example SparkPi 100`来运行，
也可以使用 `spark-submit`提交给`spark `运行。

提交到集群运行：

```bash
spark-submit \
--class  org.apache.spark.examples.SparkPi \
--master spark://Master:7077 \
--driver-memory 512m \
--executor-memory 512m \
--total-executor-cores 2 \
/usr/local/bigdata/spark-2.4.3/examples/jars/spark-examples_2.11-2.4.3.jar  \
100
```

提交到本地运行：
```bash

spark-submit \
--class  org.apache.spark.examples.SparkPi \
--master local[*] \
/usr/local/bigdata/spark-2.4.3/examples/jars/spark-examples_2.11-2.4.3.jar  \
100
```

参考连接：
- http://spark.apache.org/docs/latest/submitting-applications.html
- http://spark.apache.org/docs/latest/quick-start.html