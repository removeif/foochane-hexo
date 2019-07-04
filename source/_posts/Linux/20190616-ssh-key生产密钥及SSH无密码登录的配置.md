---
title: ssh-key生成密钥及SSH无密码登录的配置
summary: 关键词： SSH连接 ssh-key生成秘钥 无密码登录服务器 无密码访问git仓库
date: 2019-06-16 9:33:23
urlname: 2019061601
categories: linux
tags:
  - linux
  - git
  - ssh
img: /medias/featureimages/24.jpg 
author: foochane
toc: true 
mathjax: false
top: false 
cover: false
---

>SSH连接 ssh-key生成秘钥 无密码登录服务器 无密码访问git仓库

<!-- 
文章作者：[foochane](https://foochane.cn/) 

原文链接：[https://foochane.cn/article/2019061601.html](https://foochane.cn/article/2019061601.html)  
-->




## 1 ssh-keygen命令

`ssh-keygen`命令说明：
- -t ：指定加密类型（如：rea,dsa)
- -C : 指定注释,用于识别这个密钥



其他参数具体可以查看帮助
```bash
$ ssh-keygen help
Too many arguments.
usage: ssh-keygen [-q] [-b bits] [-t dsa | ecdsa | ed25519 | rsa]
                  [-N new_passphrase] [-C comment] [-f output_keyfile]
       ssh-keygen -p [-P old_passphrase] [-N new_passphrase] [-f keyfile]
       ssh-keygen -i [-m key_format] [-f input_keyfile]
       ssh-keygen -e [-m key_format] [-f input_keyfile]
       ssh-keygen -y [-f input_keyfile]
       ssh-keygen -c [-P passphrase] [-C comment] [-f keyfile]
       ssh-keygen -l [-v] [-E fingerprint_hash] [-f input_keyfile]
       ssh-keygen -B [-f input_keyfile]
       ssh-keygen -F hostname [-f known_hosts_file] [-l]
       ssh-keygen -H [-f known_hosts_file]
       ssh-keygen -R hostname [-f known_hosts_file]
       ssh-keygen -r hostname [-f input_keyfile] [-g]
       ssh-keygen -G output_file [-v] [-b bits] [-M memory] [-S start_point]
       ssh-keygen -T output_file -f input_file [-v] [-a rounds] [-J num_lines]
                  [-j start_line] [-K checkpt] [-W generator]
       ssh-keygen -s ca_key -I certificate_identity [-h] [-U]
                  [-D pkcs11_provider] [-n principals] [-O option]
                  [-V validity_interval] [-z serial_number] file ...
       ssh-keygen -L [-f input_keyfile]
       ssh-keygen -A
       ssh-keygen -k -f krl_file [-u] [-s ca_public] [-z version_number]
                  file ...
       ssh-keygen -Q -f krl_file file ...
```	   
实际情况中也用不到那么多参数，指定加密类型和注释即可。
例如：
```bash
$ ssh-keygen -t rsa -C "myname@163.com"
Generating public/private rsa key pair.
Enter file in which to save the key (C:\Users\fucheng/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in C:\Users\fucheng/.ssh/id_rsa.
Your public key has been saved in C:\Users\fucheng/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:9OlHGn5uIlELfGIYXdWectiEV5XS2quWpD1qpd2QJC8 myname@163.com
The key's randomart image is:
+---[RSA 2048]----+
|       . ....o..=|
|      . .   ..+o |
|       +.    *+. |
|      ..=.oooo=. |
|       .S=+.=o. .|
|        .o.E * . |
|         .+ @ =  |
|        . .B.B . |
|         ..++ .  |
+----[SHA256]-----+
```
一般情况下不需要输入密码，直接回车即可。

执行完`ssh-keygen`之后会在，用户目录下的`.ssh`文件下，生成一个`id_rsa`文件和`id_rsa.pub`文件。
- `id_rsa`文件是私钥，要保存好，放在本地，私钥可以生产公钥，反之不行。
- `id_rsa.pub`文件是公钥，可以用于发送到其他服务器，或者git上。
	   
## 2 ssh设置无密码登录服务器

将之前在本地生成的公钥`id_rsa.pub`,发送到需要无密码登录的服务器，然后将`id_rsa.pub`的内容追加到服务器的`~/.ssh/authorized_keys `文件中即可。

如果没有.ssh目录，创建一个就好，或者执行`ssh localhost`登录本地，ssh会自动创建。

可以使用如下命令进行操作：
```bash
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys 
```

另外如果想要无密码登录本地localhost，那么在本地执行上面的命令即可，执行之后再 `ssh localhost` 就不需要输入密码了。


**使用 ssh-copy-id命令**

使用 `ssh-copy-id 主机名@IP`可以直接完成以上两步的操作。


## 3 设置ssh无密码访问git仓库

注意这里访问的主要是私有仓库。

以`github`为例，找到个人主页，点击`[settings]`,找到`[SSH and GPG keys]` ，新建`SSH keys`,将本地`id_rsa.pub`的内容复制到`key`里面，`tittle`可以随便填写，这样就配置好了。

找到要访问的仓库主页，点击`Clone or Download` 将`use Http`换成`use SSH`,然后就会显示对应的仓库地址如：`git@github.com:uername/xxxxx.git`

使用该地址就可以在本地进行无密码访问仓库了。


