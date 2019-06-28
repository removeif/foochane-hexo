---
title: zookeeper的安装和使用
summary: 关键词：zookeeper数据存储形式 zookeeper安装  zookeeper命令行客户端的使用
date: 2019-06-26 09:25:28
urlname: 2019062601
categories: 大数据
tags:
  - 大数据
  - zookeeper
# img: /medias/featureimages/11.jpg
author: foochane
toc: true
mathjax: false
top: false
cover: false
---


<!-- 
文章作者：[foochane](https://foochane.cn/) 
</br>
原文链接：[https://foochane.cn/article/2019062601.html](https://foochane.cn/article/2019062601.html)  
-->

>zookeeper数据存储形式 zookeeper安装  zookeeper命令行客户端的使用


## 1 zookeeper数据存储形式

`zookeeper`中对用户的数据采用`kv`形式存储

`key`：是以路径的形式表示的，各key之间有父子关系，比如 `/ `是顶层`key`

用户建的`key`只能在/ 下作为子节点，比如建一个key： `/aa`  这个`key`可以带`value`数据

也可以建一个`key`：   `/bb`

也可以建多个`key`： `/aa/xx `

`zookeeper`中，对每一个数据`key`，称作一个`znode`

 

## 2 znode类型
`zookeeper`中的`znode`有多种类型：

- 1、`PERSISTENT`  持久的：创建者就算跟集群断开联系，该类节点也会持久存在与`zk`集群中
- 2、`EPHEMERAL`  短暂的：创建者一旦跟集群断开联系，`zk`就会将这个节点删除
- 3、`SEQUENTIAL`  带序号的：这类节点，`zk`会自动拼接上一个序号，而且序号是递增的

组合类型：
- `PERSISTENT`  ：持久不带序号
- `EPHEMERAL`  ：短暂不带序号 
- `PERSISTENT`  且 `SEQUENTIAL`   ：持久且带序号
- `EPHEMERAL`  且 `SEQUENTIAL`  ：短暂且带序号


## 3 安装zookeeper
解压安装包 `zookeeper-3.4.6.tar.gz`

修改`conf/zoo.cfg`

```
# The number of milliseconds of each tick
tickTime=2000
# The number of ticks that the initial 
# synchronization phase can take
initLimit=10
# The number of ticks that can pass between 
# sending a request and getting an acknowledgement
syncLimit=5
# the directory where the snapshot is stored.
# do not use /tmp for storage, /tmp here is just 
# example sakes.
dataDir=/usr/local/bigdata/data/zkdata
# the port at which the clients will connect
clientPort=2181
# the maximum number of client connections.
# increase this if you need to handle more clients
#maxClientCnxns=60
#
# Be sure to read the maintenance section of the 
# administrator guide before turning on autopurge.
#
# http://zookeeper.apache.org/doc/current/zookeeperAdmin.html#sc_maintenance
#
# The number of snapshots to retain in dataDir
#autopurge.snapRetainCount=3
# Purge task interval in hours
# Set to "0" to disable auto purge feature
#autopurge.purgeInterval=1
server.1=Master:2888:3888
server.2=Slave01:2888:3888
server.3=Slave02:2888:3888
```

对3台节点，都创建目录` /usr/local/bigdata/data/zkdata`

对3台节点，在工作目录中生成`myid`文件，但内容要分别为各自的`id`： `1`,`2`,`3`
```shell
Master上：   echo 1 > /usr/local/bigdata/data/zkdata/myid
Slave01上：  echo 2 > /usr/local/bigdata/data/zkdata/myid
Slave02上：  echo 3 > /usr/local/bigdata/data/zkdata/myid
```



## 4 启动zookeeper集群
`zookeeper`没有提供自动批量启动脚本，需要手动一台一台地起`zookeeper`进程
在每一台节点上，运行命令：
 
```shell
$ bin/zkServer.sh start

```

启动后，用`jps`应该能看到一个进程：`QuorumPeerMain`

查看状态
```shell
$ bin/zkServer.sh status

```

## 5 编写启动脚本zkmanage.sh

`zookeeper`没有提供批量脚本，不能像`hadoop`一样在一台机器上同时启动所有节点，可以自己编写脚本批量启动。

```c
#!/bin/bash
for host in Master Slave01 Slave02
do
echo "${host}:${1}ing....."
ssh $host "source ~/.bashrc;/usr/local/bigdata/zookeeper-3.4.6/bin/zkServer.sh $1"
done

sleep 2

for host in Master Slave01 Slave02
do
ssh $host "source ~/.bashrc;/usr/local/bigdata/zookeeper-3.4.6/bin/zkServer.sh status"
done

```

- $1 :指接收第一个参数

运行命令：

```shell
sh zkmanage.sh start #启动
sh zkmanage.sh stop  #停止

```

## 6 zookeeper命令行客户端

启动本地客户端：
```shell
$ bin/zkCli.sh

```
启动其他机器的客户端：
```shell
$ bin/zkCli.sh -server Master:2181

```

基本命令:

- 查看帮助：`help`
- 查看目录：`ls /`
- 查看节点数据：`get /zookeeper`
- 插入数据： `create /节点  数据` ， 如：`create /aa hello`
- 更改某节点数据： `set /aa helloworld`
- 删除数据：`rmr /aa/bb`
- 注册监听：`get /aa watch` -->数据发生改变会通知 ； `ls /aa watch` -->目录发现改变也会通知




