


## 1 IDEA安装Scala插件
点击[File]-->[Setings],找到[Plugins]，然后搜索`scala`,点击[Search in repositories]
![IDEA安装Scala插件](https://foochane.cn/images/2019/008.jpg)


## 2 新建工程


## 3 添加依赖
这里用的`spark`版本是`2.4.3`
```xml
  <dependencies>
    <dependency> <!-- Spark dependency -->
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-sql_2.12</artifactId>
      <version>2.4.3</version>
      <scope>provided</scope>
    </dependency>
  </dependencies>
```


https://www.cnblogs.com/xxbbtt/p/8143593.html
https://blog.csdn.net/shaock2018/article/details/89349249#1__32