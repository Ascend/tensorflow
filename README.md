# Tensorflow Adapter For Ascend

[View English](README.en.md)

Tensorflow Adapter For Ascend（简称TF_Adapter）致力于将昇腾AI处理器卓越的运算能力，便捷地提供给使用Tensorflow框架的开发者。
开发者只需安装TF_Adapter插件，并在现有Tensorflow脚本中添加少量配置，即可实现在昇腾AI处理器上加速自己的训练任务。

![tfadapter](https://images.gitee.com/uploads/images/2020/1027/094640_8f305b88_8175427.jpeg "framework.jpg")

您可以通过阅读 [TF_Adapter接口文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha002/tfmoddevg/tfmigr1/atlasmprtg_13_0009.html) 获取更多使用细节。
## 安装
### 从源码安装

您可以从源代码构建 TF_Adapter 软件包并将其安装在昇腾AI处理器环境上。
> TF_Adapter 插件与 Tensorflow 有严格的匹配关系，从源码构建前，您需要确保已经正确安装了 [Tensorflow v1.15.0
>版本](https://www.tensorflow.org/install/pip) 。


tfadapter也支持由源码编译，进行源码编译前，首先确保你有昇腾910 AI处理器的环境，同时系统满足以下要求：
- Linux OS
- GCC >= 7.3.0
- CMake >= 3.14.0
- [SWIG](http://www.swig.org/download.html) 
  
#### 下载源码

```
git clone https://gitee.com/ascend/tensorflow.git
cd tensorflow
```
#### 适配定制tensorflow
在部分场景下，您可能会把自己定制或者修改过的tensorflow与我们的TF_Adapter软件包配合使用，由于TF_Adapter默认链接的是tensorflow官方网站的源码，因此您在使用TF_Adapter软件包的时候，可能会因为符号不匹配而出现coredump问题。为了使TF_Adapter能适配您的tensorflow源码，我们提供了一种修改建议。
您只需要将TF_Adapter源码下的tensorflow/cmake/tensorflow.cmake文件稍作修改，便可实现这一目的。

![修改前TF_Adapter链接的是tensorflow官网源码](https://gitee.com/guopeian/tensorflow/raw/fix_readme/tf_adapter/docs/tensorflow_cmake.png "tensorflow_cmake.png")

修改图中FetchContent_Declare下的URL和URL_HASH MD5，将其替换成您自己环境上的tensorflow软件包的地址和MD5值。
例如，您的tensorflow软件包如果放在/opt/hw路径下，则您此处tensorflow.cmake的源码可以修改为

![修改后TF_Adapter链接您环境上的tensorflow定制源码](https://gitee.com/guopeian/tensorflow/raw/fix_readme/tf_adapter/docs/revise_tensorflow.png "revise_tensorflow.png")

#### 修改源码
如果您想对TF_Adapter的源码进行修改，比如添加链接路径，或链接其他so等操作，您可以修改TF_Adapter源码下的tensorflow/CMakeLists.txt文件，只需要将ENABLE_OPEN_SRC分支下的编译配置做修改，便可以生效

![CMakeList.txt文件](https://gitee.com/guopeian/tensorflow/raw/fix_readme/tf_adapter/docs/cmake.png "cmake.png")

#### 执行脚本生成安装包
```
chmod +x build.sh
./build.sh
```

脚本执行成功后，会在output目录生成tfadapter.tar压缩文件。

#### 安装插件包
解压tfadapter.tar文件，生成npu_bridge-1.15.0-py3-none-any.whl,
然后使用 pip 安装 TF_Adapter 插件。
```
pip install npu_bridge-1.15.0-py3-none-any.whl
```
需要注意的是， 您应当保证安装路径与您编译时指定的 python
 解释器搜索路径是一致的。

## 贡献

欢迎参与贡献。

## 社区版本规划
https://gitee.com/ascend/tensorflow/wikis/Home?sort_id=3076366

## Release Notes

Release Notes请参考[RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
