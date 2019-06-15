---
title: Windows下jekyll的安装和使用
summary: jekyll是一个简单的免费的Blog生成工具，类似WordPress。但是和WordPress又有很大的不同，原因是jekyll只是一个生成静态网页的工具，不需要数据库支持。
author: foochane
categories: 博客
date: 2019-05-19 10:56
urlname: 2019051905
tags:
  - jekyll
  - 博客
---

>jekyll是一个简单的免费的Blog生成工具，类似WordPress。但是和WordPress又有很大的不同，原因是jekyll只是一个生成静态网页的工具，不需要数据库支持。但是可以配合第三方服务,例如Disqus。最关键的是jekyll可以免费部署在Github上，而且可以绑定自己的域名。

## 1 安装 Ruby+devkit
下载地址：https://rubyinstaller.org/downloads/

下载安装包：rubyinstaller-devkit-2.5.5-1-x64.exe

点击安装即可，在安装结束时，不要勾选`ridk install`的选项，后面再手动安装

检查ruby是否正常安装，会出现版本号
```
ruby -v
```
检查gem是否安装完毕：

>gem -v

## 2 安装MSYS2

输入命令：
```
ridk install
```
输入“ridk install”进行MSYS2的安装，会出现然你选择123，你选3就行。这个过程会下载很多安装包什么的，耐心等待，一定要耐心，要完整装完才行，装好会让你再做一次123选择，这个时候不需要选了，直接enter退出就行了。

## 3 安装bundler
输入
```
gem install bundler
```
执行安装

## 4 安装jekyll

输入命令：
```
gem install jekyll
```
检查jekyll是否安装成功
```
jekyll -v
```

如果没什么问题，会显示版本信息说明安装成功。

具体可以参考jekyll官方文档：https://jekyllrb.com/docs/installation/windows/


## 5 使用jekyll创建简单的博客
### 5.1 创建博客
输入命令：
```
jekyll new myblog
```

### 5.2 本地运行博客
切换到`myblog`目录下，输入如下命令
```
bundle exec jekyll serve
```
打开浏览器，访问[http://localhost:4000](http://localhost:4000),即可查看博客。