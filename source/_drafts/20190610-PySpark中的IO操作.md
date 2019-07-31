---
title: PySpark中的IO操作
summary: 关键词：PySaprk IO
date: 2019-06-10 14:40:28
urlname: 2019061001
categories: 大数据
tags:
  - PySpark
  - 大数据
# img: /medias/featureimages/9.jpg
author: foochane
toc: true
mathjax: false
top: false
cover: false
---

## 本文内容

从不同类型的文件格式中读取数据并将结果保存到多个数据接收器中是数据分析中不可避免的一部分。本文将介绍如何从不同类型的数据源读取数据，以及如何将分析结果保存到不同的文件或者数据库中。

将包含如下内容：
- 1 读取CSV文件
- 2 读取JSON文件
- 3 保存DataFrame格式数据到CSV文件中
- 4 保存DataFrame格式数据到JSON文件中
- 5 读取ORC文件
- 6 读取Parquet文件
- 7 保存DataFrame格式数据到ORC文件中
- 8 保存DataFrame格式数据Parquet文件中
- 9 从MySQL数据库中读取数据
- 10 从PostgreSQL数据库中读取数据
- 11 从MongoDB数据库中读取数据
- 12 从MySQL数据库中读取数据
- 13 保存DataFrame格式数据到MySQL数据库中

下面进行具体的介绍。

## 1 读取CSV文件



读取一个文件名为`worker.csv`的`CSV`文件，具体内容如下。



|id|job|age|salary|
|--|--|--|--|
|221|seller|21|	3222|
|222|HR manager|32|5555|
|333|Buyer|44|4343|
|339|store keeper|32|3353|

我们使用`spark.read.csv()`函数读取CSV文件。在这里，spark是SparkSession类的对象。该函数需要指定两个参数，第一个指定CSV文件的路径，它必须由path参数读取。PySpark SQL DataFrame具有类似于RDBMS表的表结构。因此，第二个参数指定DataFrame的模式。

### 1.1 上传cvs文件
先将`worker.csv`文件上传到 `HDFS`上：
```bash
hadoop@Master:~$ hdfs dfs -put worker.csv /pyspark
hadoop@Master:~$ hdfs dfs -ls /pyspark
Found 2 items
-rw-r--r--   3 hadoop supergroup        184 2019-06-10 08:10 /pyspark/test.txt
-rw-r--r--   3 hadoop supergroup        204 2019-06-10 08:11 /pyspark/work.csv
```
### 1.2 创建数据表的Schema

数据表中有4列，先用 `StructField()`定义这些列。`PySpark SQL`有自己的数据类型，所有的数据类型都定义在`pyspark.sql.types`子模块中，所以先导入该模块。
创建过程如下：
```python
from pyspark.sql.types import *
idColumn = StructField("id",StringType(),True)
jobColumn = StructField("job",StringType(),True)
ageColumn = StructField("age",DoubleType(),True)
salaryColumn = StructField("salary",DoubleType(),True)
columnList = [idColumn, jobColumn, ageColumn,salaryColumn]
workerDfSchema = StructType(columnList)
```

workerDf = spark.read.csv('/pyspark/worker.csv',header=True,schema=workerDfSchema)
workerDf.show(3)
workerDf.printSchema()


```python
>>>
>>>
>>> idColumn = StructField("id",StringType(),True)
>>> jobColumn = StructField("job",StringType(),True)
>>> ageColumn = StructField("age",DoubleType(),True)
>>> salaryColumn = StructField("salary",DoubleType(),True)
>>> columnList = [idColumn, jobColumn, ageColumn,salaryColumn]
>>> workerDfSchema = StructType(columnList)
>>> workerDf = spark.read.csv('/pyspark/worker.csv',header=True,schema=workerDfSchema)
>>> workerDf.show(3)
+---+----------+----+------+
| id|       job| age|salary|
+---+----------+----+------+
|221|    seller|21.0|3222.0|
|222|HR manager|32.0|5555.0|
|333|     Buyer|44.0|4343.0|
+---+----------+----+------+
only showing top 3 rows

>>> workerDf.show(2)
+---+----------+----+------+
| id|       job| age|salary|
+---+----------+----+------+
|221|    seller|21.0|3222.0|
|222|HR manager|32.0|5555.0|
+---+----------+----+------+
only showing top 2 rows

>>> workerDf.printSchema()
root
 |-- id: string (nullable = true)
 |-- job: string (nullable = true)
 |-- age: double (nullable = true)
 |-- salary: double (nullable = true)

>>>
```

### 1.3

```python
workerDf1 = spark.read.csv('/pyspark/worker.csv',header=True, inferSchema=True)
workerDf1.show(4)
workerDf1.printSchema()
```

```python
>>> workerDf1 = spark.read.csv('/pyspark/worker.csv',header=True, inferSchema=True)
>>> workerDf1.show(4)
+---+------------+---+------+
| id|         job|age|salary|
+---+------------+---+------+
|221|      seller| 21|  3222|
|222|  HR manager| 32|  5555|
|333|       Buyer| 44|  4343|
|339|store keeper| 32|  3353|
+---+------------+---+------+

>>> workerDf1.printSchema()
root
 |-- id: integer (nullable = true)
 |-- job: string (nullable = true)
 |-- age: integer (nullable = true)
 |-- salary: integer (nullable = true)

>>>
```
## 2 读取JSON文件

## 3 保存DataFrame格式数据到CSV文件中

## 4 保存DataFrame格式数据到JSON文件中

## 5 读取ORC文件

## 6 读取Parquet文件

## 7 保存DataFrame格式数据到ORC文件中

## 8 保存DataFrame格式数据Parquet文件中

## 9 从MySQL数据库中读取数据

## 10 从PostgreSQL数据库中读取数据

## 11 从MongoDB数据库中读取数据

## 12 从MySQL数据库中读取数据

## 13 保存DataFrame格式数据到MySQL数据库中