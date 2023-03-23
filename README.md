# TensorFlow Adapter For Ascend

[View English](README.en.md)

## 简介
TensorFlow Adapter For Ascend（简称TF Adapter）致力于将昇腾AI处理器卓越的运算能力，便捷地提供给使用Tensorflow框架的开发者。
开发者只需安装TF Adapter插件，并在现有TensorFlow脚本中添加少量配置，即可实现在昇腾AI处理器上加速自己的训练任务。

![tfadapter](https://images.gitee.com/uploads/images/2020/1027/094640_8f305b88_8175427.jpeg "framework.jpg")

您可以通过阅读[昇腾社区文档中心](https://www.hiascend.com/zh/document)的《TensorFlow 迁移与训练》手册获取更多使用细节。

## 源码安装
您可以通过此仓中的源代码构建TF Adapter软件包并将其部署在昇腾AI处理器所在环境上。

### 安装前准备
1. 通过源码方式安装TF Adaptet前，请确保环境已参见[昇腾社区文档中心](https://www.hiascend.com/zh/document)中心的《CANN软件安装》手册完成以下安装与配置：

   a. 完成gcc、python等依赖的安装。

   b. 完成开发套件包Ascend-cann-toolkit*_{version}*_linux-*{arch}*.run的安装。

   c. 完成开发套件包的环境变量配置，假设以root用户使用默认安装路径进行安装，则环境变量配置命令：source  /usr/local/Ascend/ascend-toolkit/set_env.sh。

2. TF Adapter 插件与 Tensorflow 有严格的匹配关系，通过源码构建TF Adapter软件包前，您需要确保已经正确安装了 [Tensorflow v1.15.0版本](https://www.tensorflow.org/install/pip) ，安装方式可参见[昇腾社区文档中心](https://www.hiascend.com/zh/document)中心《CANN软件安装》手册中的“安装深度学习框架 > 安装TensorFlow”章节。

3. TF Adapter源码编译要求CMake软件版本 >=3.14.0，若您系统上的CMake版本不满足要求，可从[CMake官网](https://cmake.org/download/)下载配套操作系统的CMake软件包。
   安装示例：

   a. 这里是列表文本解压缩CMake软件包

   ​    `tar -zxvf cmake-3.19.3-Linux-x86_64.tar.gz`

   ​     解压后当前目录下生成cmake-3.19.3-Linux-x86_64的文件夹。

   b. 设置CMake的环境变量。

   ​    `export PATH=/home/xxx/xxx/cmake-3.19.3-Linux-x86_64/bin:$PATH`

   ​    以上路径请替换为CMake的实际部署路径。

   c. 检查是否安装成功。
   
   ​    执行echo $PATH查看环境变量是否设置成功。
   
   ​    执行如下命令，查看测试是否安装成功：
   
   ​    `cmake --version`

 4. TF Adapter源码编译依赖SWIG。
       可执行如下命令进行SWIG(http://www.swig.org/download.html))的安装：

       `pip3 install swig`

### TF Adapter源码下载

```
git clone https://gitee.com/ascend/tensorflow.git
cd tensorflow
```
### TensorFlow源码定制（可选）
在部分场景下，您可能会把自己定制或者修改过的TensorFlow与TF Adapter软件包配合使用，由于TF Adapter默认链接的是TensorFlow官方网站的源码，因此您在使用TF Adapter软件包的时候，可能会因为符号不匹配而出现coredump问题。为了使TF Adapter能适配您的TensorFlow源码，您需要将TF Adapter源码下的tensorflow/cmake/tensorflow.cmake文件稍作修改，详细修改点如下：

![修改前TF_Adapter链接的是tensorflow官网源码](https://gitee.com/guopeian/tensorflow/raw/fix_readme/tf_adapter/docs/tensorflow_cmake.png "tensorflow_cmake.png")

修改图中FetchContent_Declare下的URL和URL_HASH MD5，将其替换成您自己环境上的tensorflow软件包的地址和MD5值。
例如，您的tensorflow软件包如果放在/opt/hw路径下，则您此处tensorflow.cmake的源码可以修改为

![修改后TF_Adapter链接您环境上的tensorflow定制源码](https://gitee.com/guopeian/tensorflow/raw/fix_readme/tf_adapter/docs/revise_tensorflow.png "revise_tensorflow.png")

### TF Adapter源码定制（可选）
如果您想对TF Adapter的源码进行修改，比如添加链接路径，或链接其他so等操作，您可以修改TF Adapter源码下的tensorflow/CMakeLists.txt文件，只需要将ENABLE_OPEN_SRC分支下的编译配置做修改，便可以生效

![CMakeList.txt文件](https://gitee.com/guopeian/tensorflow/raw/fix_readme/tf_adapter/docs/cmake.png "cmake.png")

### 编译TF Adapter源码生成安装包
执行如下命令，对TF Adapter源码进行编译：
```
chmod +x build.sh
./build.sh
```
> 请注意：执行编译命令前，请确保环境中已配置开发套件包Ascend-cann-toolkit*_{version}*_linux-*{arch}*.run的环境变量。

编译结束后，TF Adapter安装包生成在如下路径：

```
./build/tfadapter/dist/python/dist/npu_bridge-1.15.0-py3-none-any.whl
```

### 安装TF Adapter
执行如下命令安装TF Adapter。
```
pip3 install ./build/tfadapter/dist/python/dist/npu_bridge-1.15.0-py3-none-any.whl --upgrade
```
执行完成后，TF Adapter相关文件安装到python解释器搜索路径下，例如“/usr/local/python3.7.5/lib/python3.7/siite-packages”路径，安装后文件夹为“npu_bridge”与“npu_bridge-1.15.0.dist-info”。

## 贡献
欢迎参与贡献。

## 社区版本规划
https://gitee.com/ascend/tensorflow/wikis/Home?sort_id=3076366

## Release Notes

Release Notes请参考[RELEASE](RELEASE.md).

## FAQ
#### 1. 执行./build.sh时提示配置swig的路径
需要执行以下命令安装swig
```
pip3 install swig
```
#### 2. Ubuntu系统中执行./build.sh时提示“Could not import the lzma module”

​     执行如下命令进行lzma的安装：

​     `apt-get install liblzma-dev`

​      需要注意，此依赖需要在Python安装之前安装，如果用户操作系统中已经安装满足要求的Python环境，在此之后再安装liblzma-dev，则需要重新编译Python环境。


## License

[Apache License 2.0](LICENSE)
