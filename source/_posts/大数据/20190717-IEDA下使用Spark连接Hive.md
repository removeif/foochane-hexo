---
title: IDEA下使用Spark连接Hive
summary: 关键词：IDEA Spark 连接 Hive
date: 2019-07-17 17:33:28
urlname: 2019071701
categories: 大数据
tags:
  - 大数据
  - sqoop
img: /medias/featureimages/21.jpg 
author: foochane
toc: true
mathjax: false
top: false
cover: false
---

## 1 Spark配置
### 1.1 复制hive-site.xml文件到spark中
```sh
$ cp /usr/local/bigdata/hive-2.3.5/conf/hive-site.xml /usr/local/bigdata/spark-2.4.3/conf/

```

### 1.2 spark中安装mysql-connector-java

```sh
$ sudo apt-get install libmysql-java
$ ln -s /usr/share/java/mysql-connector-java-5.1.45.jar /usr/local/bigdata/spark-2.4.3/jars

```

## 2 IDEA中新建maven工程

### 2.1 将hive-site.xml拷贝到resources目录下

### 2.2 添加依赖

```xml
<dependency>
	<groupId>org.apache.hadoop</groupId>
	<artifactId>hadoop-client</artifactId>
	<version>2.4.3</version>
</dependency>

<dependency>
	<groupId>org.scala-lang</groupId>
	<artifactId>scala-library</artifactId>
	<version>2.4.3</version>
</dependency>

<dependency>
	<groupId>org.apache.spark</groupId>
	<artifactId>spark-core_2.11</artifactId>
	<version>2.4.3</version>
</dependency>
<dependency>
	<groupId>org.apache.spark</groupId>
	<artifactId>spark-hive_2.11</artifactId>
	<version>2.4.3</version>
</dependency>
<dependency>
	<groupId>org.apache.spark</groupId>
	<artifactId>spark-sql_2.11</artifactId>
	<version>2.4.3</version>
</dependency>

```


### 2.3 添加代码

```scala
import org.apache.spark.sql.SparkSession

object SparkHiveDemo {
  def main(args: Array[String]): Unit = {

    //设置HADOOP_USER_NAME，否则会有权限问题
    System.setProperty("HADOOP_USER_NAME", "hadoop")

    val spark = SparkSession
      .builder()
      .appName("SparkHiveDemo")
      .master("spark://Node02:7077")
      .enableHiveSupport()
      .config("spark.sql.warehouse.dir", "/user/hive/warehouse/")
      .getOrCreate()

    spark.sql("show databases").show()
    spark.close()
  }
}
```

## 3 问题

问题1
```
Exception in thread "main" java.lang.IllegalArgumentException: Unable to instantiate SparkSession with Hive support because Hive classes are not found.
	at org.apache.spark.sql.SparkSession$Builder.enableHiveSupport(SparkSession.scala:869)
	at DataFrames3$.main(DataFrames3.scala:30)
	at DataFrames3.main(DataFrames3.scala)
```

解决1
找不到"org.apache.hadoop.hive.conf.HiveConf" 配置文件

添加依赖

```xml
<dependency>
	<groupId>org.apache.spark</groupId>
	<artifactId>spark-hive_2.11</artifactId>
	<version>2.4.3</version>
</dependency>
```


问题2
```
Exception in thread "main" org.apache.spark.sql.AnalysisException: java.lang.RuntimeException: org.apache.hadoop.security.AccessControlException: Permission denied: user=fucheng, access=WRITE, inode="/user/hive/tmp":hadoop:supergroup:drwxr-xr-x
```
解决：
方法一：修改hadoop配置文件conf/hdfs-core.xml
```xml
<property>
<name>dfs.permissions</name>
<value>false</value>
</property>
```

方法二：直接修改要访问的文件的权限
```
hdfs dfs chmod 777 /test
```

方法三：代码里设置HADOOP_USER_NAME

在代码里添加
```
System.setProperty("HADOOP_USER_NAME", "hdp");
```


问题3：
```
MetaException(message:Hive Schema version 2.3.0 does not match metastore's schema version 1.2.0 Metastore is not upgraded or corrupt)

```

方法一：
在mysql中执行

```
mysql> use metastore
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
mysql> UPDATE VERSION SET SCHEMA_VERSION='2.3.0', VERSION_COMMENT='fix conflict' where VER_ID=1;
Query OK, 1 row affected (0.05 sec)
Rows matched: 1  Changed: 1  Warnings: 0
```

方法二：

在hive-site.xml中关闭版本验证
```xml
<property>
    <name>hive.metastore.schema.verification</name>
    <value>false</value>
</property>
```