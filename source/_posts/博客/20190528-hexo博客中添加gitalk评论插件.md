---
title: hexo博客中添加gitalk评论插件
summary: 关键词： hexo 博客 gitalk 评论 插件
author: foochane
top: false
cover: false
categories: 博客
date: 2019-05-28 20:39
urlname: 2019052801
tags:
  - hexo
  - 博客
---


## 1 新建仓库
申请地址：[https://github.com/settings/applications/new](https://github.com/settings/applications/new)


填写申请内容：
![](/images/2019/003.jpg))


记住`Client ID`和 `Client Secret`
![](/images/2019/004.jpg))

## 2 添加代码

在`*/layout/_partial/`下新建一个`comment.ejs`，添加如下内容：
```
<link rel="stylesheet" href="<%- theme.libs.css.gitalk %>">
<link rel="stylesheet" href="/css/my-gitalk.css">

<div class="card gitalk-card" data-aos="fade-up">
    <div id="gitalk-container" class="card-content"></div>
</div>

<script src="<%- theme.libs.js.gitalk %>"></script>
<script>
    let gitalk = new Gitalk({
        clientID: '<%- theme.gitalk.oauth.clientId %>',
        clientSecret: '<%- theme.gitalk.oauth.clientSecret %>',
        repo: '<%- theme.gitalk.repo %>',
        owner: '<%- theme.gitalk.owner %>',
        admin: <%- JSON.stringify(theme.gitalk.admin) %>,
        id: '<%- date(page.date, 'YYYY-MM-DDTHH-mm-ss') %>',
        distractionFreeMode: false  // Facebook-like distraction free mode
    });

    gitalk.render('gitalk-container');
</script>
```


在post页添加：
```
{% elseif theme.gitalk.enable %}
    <%- partial('_partial/comment') %>
{% endif %}
```

## 3 修改配置文件
在` */_config.yml`中增加以下内容:
```
gitalk:
  enable: true #用来做启用判断可以不用
  owner: #Github 用户名,
  repo: #储存评论issue的github仓库名
  admin: #Github 用户名,
  oauth:
    clientID: #`Github Application clientID`
    clientSecret: #`Github Application clientSecret`
```
## 4 查看效果
部署代码后，访问博客页面

第一次访问需要登陆，github账号。
![](/images/2019/005.jpg))

关联账号，授权：
![](/images/2019/006.jpg))


之后就可以使用评论了：

![](/images/2019/007.jpg))

到这里，gitalk插件就添加成功了。

