# 电力人工智能数据竞赛决赛——安全帽未佩戴行为目标检测赛道基准模型
以下为电力人工智能数据竞赛决赛-安全帽未佩戴行为目标检测赛道基准模型介绍。其中包含了智源联邦学习框架的简化版本（真实版本后期会以论文的形式发布）和基于 [YOLOv3模型](https://github.com/eriklindernoren/PyTorch-YOLOv3) 完成的实验。  
* 决赛和初赛的区别：
    * 联邦学习框架
        * 决赛升级了联邦学习通信框架，当前基于 [fastapi](https://github.com/tiangolo/fastapi) 和 [aiohttp](https://github.com/aio-libs/aiohttp) 的实现不仅提升了通信效率和安全，而且解决了超时问题
        * 决赛联邦学习框架会启动 [差分隐私模块](https://github.com/pytorch/opacus) 保护通信过程中模型参数隐私性
    * 数据集
        * [决赛数据集](https://open.baai.ac.cn/data-set-detail/MTI2NTE=/Njk=/true) 为国网电力真实场景的安全帽图片（包含没有安全帽的背景图），共计15000张
    * 选手权限
        * 决赛选手无法访问和接触决赛数据集
        * 决赛选手只能操作`contestant-server`（基于智源提供的`CPU`机器，每位选手提供一台），智源和国网电力操作`baai-client`和`sgcc-client`（基于智源和国网电力提供的`GPU`机器，两个`client`处于不同的`GPU`机器，每个`client`提供一块`Tesla V100 32GB`），以完成真实场景的联邦学习
        * 决赛选手完成提交之后，主办方会提供具体访问`CPU`和`GPU`机器的方式
* 选手流程：
    * 将新的联邦学习框架跟初赛的数据集和模型结合，于本地跑通整个流程
    * 在官网完成zip压缩文件的提交，主办方下载本日最后一次提交，于次日通知选手开始联邦学习
    * 完成联邦学习的训练、测试、检测流程之后，自动更新决赛榜单分数
    * 选手根据整体情况更新优化项目，再次提交，开启下一次联邦学习
    * 注意事项：
        * 选手在提交zip文件时一定按照要求（初赛提交时部分选手不符合，决赛缺一不可）
            * 提交次数：每天最多提交5次
            * 提交大小限制：不超过500M
            * 提交文件：一个zip压缩文件，包括以下部分：可复现结果的代码（基于联邦学习框架），预训练模型，详细说明文档（可以独立写文档或者写在README.md中，包括队伍名称、选手单位、选手姓名和联系方式，项目解读，运行环境（系统型号、CUDA版本、GPU版本和配置），需要安装的python库（代码所使用的库和版本可写在requirements.txt中），运行方式等等）

## 环境要求
服务器环境：  
* nvidia/cuda:10.2-devel-ubuntu18.04

Python库环境：
* aiohttp==3.7.3
* fastapi==0.62.0
* Flask==1.1.1
* loguru==0.5.3
* numpy==1.18.1
* Pillow==7.0.0
* Python==3.7.6
* python-multipart==0.0.5
* torch==1.7.1
* terminaltables==3.1.0
* torchvision==0.8.2
* tqdm==4.42.1
* uvicorn==0.13.1

详情请参考`finals-helmet-baseline`下面的`requirements.txt`

## 项目结构
```
.
├── README.md
├── baai-client  # 智源联邦学习客户端
│   ├── api
│   │   └── client_api.py  # 选手联邦学习服务端需要调用智源联邦学习客户端的函数（训练、测试等）
│   ├── config
│   │   ├── base_conf.py  # 本机IP地址和端口号配置
│   │   ├── finals_contest_helmet_federal_conf
│   │   │   ├── create_finals_contest_helmet_federal_model_conf.sh  # 构建yolov3模型的bash脚本
│   │   │   └── finals_contest_helmet_federal_conf.data  # 决赛安全帽数据集路径信息
│   │   ├── options_conf.py  # 初始化参数函数
│   │   └── routes_conf.py  # 配置调用方法
│   ├── main.py  # 智源联邦学习客户端启动主函数
│   ├── module
│   │   └── models_module.py  # yolov3模型相关类
│   ├── service
│   │   └── federated
│   │       └── client_service.py  # 智源联邦学习客户端类
│   └── utils
│       ├── common_utils.py  # 常用功能函数
│       ├── custom_error_utils.py  # 自定义错误信息类
│       ├── data_utils.py  # 处理加载数据集相关函数
│       ├── result_utils.py  # 通信结果封装类
│       └── tool_utils.py  # 辅助功能函数
├── contestant-server  # 选手联邦学习服务端
│   ├── config
│   │   ├── finals_contest_helmet_federal_conf
│   │   │   └── create_finals_contest_helmet_federal_model_conf.sh  # 构建yolov3模型的bash脚本
│   │   └── options_conf.py  # 初始化参数函数
│   ├── http_request
│   │   └── async_request.py  # 异步调用方法类
│   ├── module
│   │   └── models_module.py  # yolov3模型相关类
│   ├── service
│   │   └── federated
│   │       └── server_service.py  # 选手联邦学习服务端类
│   └── utils
│       ├── common_utils.py  # 常用功能函数
│       ├── custom_error_utils.py  # 自定义错误信息类
│       ├── data_utils.py  # 处理加载数据集相关函数
│       ├── result_utils.py  # 通信结果封装类
│       └── tool_utils.py  # 辅助功能函数
├── requirements.txt  # 需要安装的python库
└── sgcc-client  # 国网电力联邦学习客户端（功能与智源联邦学习客户端相同，此处不展开介绍）
    ├── api
    │   └── client_api.py
    ├── config
    │   ├── base_conf.py
    │   ├── finals_contest_helmet_federal_conf
    │   │   ├── create_finals_contest_helmet_federal_model_conf.sh
    │   │   └── finals_contest_helmet_federal_conf.data
    │   ├── options_conf.py
    │   └── routes_conf.py
    ├── main.py
    ├── module
    │   └── models_module.py
    ├── service
    │   └── federated
    │       └── client_service.py
    └── utils
        ├── common_utils.py
        ├── custom_error_utils.py
        ├── data_utils.py
        ├── result_utils.py
        └── tool_utils.py
```

## 下载地址
* [决赛安全帽数据集](https://open.baai.ac.cn/data-set-detail/MTI2NTE=/MzA=/true)    
    * 决赛选手不可访问决赛安全帽数据集，此处给出统计信息说明（周一更新）
* [yolov3预训练模型](http://dorc-data.ks3-cn-beijing.ksyun.com/2015682aasdf154asdfe5d5aq961fa6eg/weights_yolov3_pre_model/weights.tar.gz)  
    * 当前主要采用`weights`下面的`darknet53.conv.74`
## 运行方式
### 智源联邦学习客户端
* 进入`baai-client/config/finals_contest_helmet_federal_conf`目录
  * 修改`finals_contest_helmet_federal_conf.data`当中的数据路径
  * 生成`yolov3`模型的`cfg`文件（先删除旧的`cfg`文件）  
  `bash create_finals_contest_helmet_federal_model_conf.sh 2`
* 进入`baai-client/config`
  * 修改`base_conf.py`当中的`host`和`port`
* 进入`baai-client/config`目录
  * 配置`options_conf.py`当中的参数，特别是`data_config`，`model_def`，`server_ip`，`server_port`，`client_ip`，`client_port`

* 启动智源联邦学习客户端
  * 进入`baai-client`，运行以下指令  
  `python main.py`

### 国网电力联邦学习客户端
* 进入`sgcc-client/config/finals_contest_helmet_federal_conf`目录
  * 修改`finals_contest_helmet_federal_conf.data`当中的数据路径
  * 生成`yolov3`模型的`cfg`文件（先删除旧的`cfg`文件）  
  `bash create_finals_contest_helmet_federal_model_conf.sh 2`
* 进入`sgcc-client/config`
  * 修改`base_conf.py`当中的`host`和`port`
* 进入`sgcc-client/config`目录
  * 配置`options_conf.py`参数，特别是`data_config`，`model_def`，`server_ip`，`server_port`，`client_ip`，`client_port`

* 启动国网电力联邦学习客户端
  * 进入`sgcc-client`，运行以下指令  
  `python main.py`

### 选手联邦学习服务端
* 把下载好的 [yolov3预训练模型](http://dorc-data.ks3-cn-beijing.ksyun.com/2015682aasdf154asdfe5d5aq961fa6eg/weights_yolov3_pre_model/weights.tar.gz) 拷贝到`contestant-server/service/federated/weights`
* 进入`contestant-server/config/finals_contest_helmet_federal_conf`目录
  * 生成`yolov3`模型的`cfg`文件（先删除旧的`cfg`文件）  
  `bash create_finals_contest_helmet_federal_model_conf.sh 2`
* 进入`contestant-server/config`目录
  * 配置`options_conf.py`参数，特别是`pretrained_weights`，`model_def`，`server_ip`，`server_port`，`client_ips`，`client_ports`

* 启动选手联邦学习服务端
  * 进入`contestant-server`，运行以下指令  
    `PYTHONPATH=your/project/path/contestant-server python service/federated/server.py`

## 实验指标 
* 国网电力指标：[电力人工智能数据竞赛指标计算方法和自动评测脚本的详细介绍](https://github.com/AIOpenData/baai-federated-learning-baseline-metric)

## 实验结果
基于默认实验参数，决赛安全帽测试集基于YOLOv3模型的结果： 
<div class="table">
<table border="1" cellspacing="0" cellpadding="10" width="100%">
<thead>
<tr class="firstHead">  
<th colspan="1" rowspan="1">false detection rate</th> <th>missed detection rate</th> <th>object detection correct rate</th> <th>sgcc helmet image score</th>
 </tr>
</thead>
<tbody>
<tr>
<td>0.7039337474120083</td>
<td>0.18664383561643835</td> <td>0.6044609665427509</td> <td>0.6163901512767285</td>
</tr>
</tbody>
</table>
</div>