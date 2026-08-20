#

# Improved-DeepLabV3Plus-for-Potato-Leaf-Disease-Segmentation

> 官方PyTorch 实现 | 论文处于投刊阶段，标题：《Research on Potato Leaf Disease Segmentation and Recognition Method Based on Improved DeepLabv3+》  

> 提出改进的DeepLabV3+模型，融合ToST双注意力与FDAM频域动态调制模块，实现土豆早疫病、晚疫病叶片病斑的高精度像素级分割，助力智慧农业病害智能监测。  

## 1. 研究背景与模型定位

土豆作为全球重要的粮菜兼用作物，其叶片病害（如早疫病、晚疫病）会严重削弱光合作用，导致产量大幅下降。传统的人工检查依赖经验，效率低且主观性强。  

本文提出**基于改进DeepLabV3+的语义分割模型**，通过创新的全局上下文增强与频域边缘细节恢复机制，解决土豆叶片病害“病斑边缘模糊、小目标漏检及全局语义缺失”的问题。模型基于 PyTorch 框架实现，包含轻量化 MobileNetV2 骨干网络、ToST 双注意力模块与 FDAM 频域动态调制模块，在自建土豆早疫病与晚疫病数据集上实现优异的像素级分割性能，为智慧农业病害智能监测提供高效解决方案。

## 2.模型核心创新点

1. **双模块注意力协同机制**：

  

   - **ToST双注意力模块**：基于最大编码率降低原理推导TSSA算子，将传统自注意力增强全局语义建模能力，弥补原生DeepLabV3+对远距离上下文依赖提取不足的缺陷；

   - **FDAM频域动态调制模块**：提出注意力反转策略构造互补高通滤波器，结合频域动态缩放恢复被低通滤波抑制的高频边缘细节，改善病斑边界清晰度与分割精度。

2. **全局上下文增强**：

  

   采用MobileNetV2作为轻量级骨干网络，大幅降低参数量与计算开销；将解码器中两层3×3卷积精简为单层并引入Dropout(0.3)，减少冗余参数，缓解过拟合风险。  

3. **全局上下文增强**：

  

   针对早疫病（同心轮纹、边缘黄晕）与晚疫病（水渍状、形态不规则）在病斑尺度与形态上的差异，引入ASPP空洞空间金字塔池化，通过多膨胀率卷积并行采样，捕获从细小病斑到大面积病害的多尺度上下文信息，降低相似病斑的混淆程度。  

## 3. 实验数据集：土豆叶片病害数据集

### 3.1 数据集概况

本研究基于**土豆叶片病害数据集**，涵盖两种典型土豆叶部病害：早疫病（Early Blight）和晚疫病（Late Blight），图像主要收集自公开网络资源并经过人工筛选与像素级标注。数据集统计如下：  

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |

|------------|-------------------------|----------|------------|-----------------------|

| 土豆叶片病害数据集 | 早疫病（Early Blight）和晚疫病（Late Blight） + 健康叶片（Healthy） | 5,000+ | 统一resize至256×256（适配模型输入） | 7:2:1 |  

标注类别（4类）：

0: 背景（黑色）

1: 健康叶片（棕色 #aa5500）

2: 晚疫病病斑（绿色 #55aa7f）

3: 早疫病病斑（粉色 #ffaaff）

### 3.2 数据集获取与结构

1. **下载链接**：

  

   github链接：   

https://github.com/SCD-ykd/Potato-Leaf-Disease-Segmentation-and-Recognition/tree/main/potato

2. **文件夹组织**（下载后解压至项目根目录，结构如下）：

  

```

potato/

├── train/

│   ├── Early blight/          

│   │   ├── images/          # 如 Early_blight_1.jpg

│   │   └── masks/           # 如 Early_blight_1_pseudo.png

│   └── Late blight/

│       ├── images/

│       └── masks/

├── val/

│   └── (同上)

└── test/

    └── (同上)  

```

## 4. 实验环境配置

### 4.1 依赖安装

推荐使用 Anaconda 创建虚拟环境（Python 3.8+），并安装 PyTorch 及其依赖：  

```bash

# 创建并激活虚拟环境

conda create -n deeplab_potato python=3.8

conda activate deeplab_potato

# 安装 PyTorch（CPU 或 GPU 版本，本例使用 CPU）

pip install torch==2.4.1 torchvision==0.19.1

# 安装其他依赖库

pip install numpy opencv-python pillow tqdm matplotlib scikit-learn pandas  

```

## 5 代码使用说明

### 5.1 模型训练

运行`train.py`脚本启动训练，支持通过参数调整训练配置，示例命令：  

```bash

python train.py \

  --dataset_root ./split_dataset \        # 数据集根目录

  --num_classes 4 \                       # 类别数（背景+健康叶片+早疫病+晚疫病）

  --backbone mobilenet \                  # 骨干网络

  --input_shape 256 256 \                 # 输入图像尺寸

  --Freeze_Epoch 50 \                     # 冻结骨干训练轮数

  --UnFreeze_Epoch 1000 \                 # 全网络训练轮数

  --Freeze_batch_size 8 \                 # 冻结阶段批次大小

  --Unfreeze_batch_size 4 \               # 解冻阶段批次大小

  --Init_lr 1e-3 \                        # 初始学习率

  --optimizer_type sgd \                  # 优化器类型

  --dice_loss True \                      # 启用Dice损失

  --focal_loss True                       # 启用Focal损失  

```

## 5 代码使用说明

### 5.1 模型训练

运行`train.py`脚本启动训练，支持通过参数调整训练配置，示例命令：  

```bash

python train.py \

  --dataset_root ./potato \        # 数据集根目录

  --num_classes 4 \                       # 类别数（背景+健康叶片+早疫病+晚疫病）

  --backbone mobilenet \                  # 骨干网络

  --input_shape 256 256 \                 # 输入图像尺寸

  --Freeze_Epoch 50 \                     # 冻结骨干训练轮数

  --UnFreeze_Epoch 1000 \                 # 全网络训练轮数

  --Freeze_batch_size 8 \                 # 冻结阶段批次大小

  --Unfreeze_batch_size 4 \               # 解冻阶段批次大小

  --Init_lr 1e-3 \                        # 初始学习率

  --optimizer_type sgd \                  # 优化器类型

  --dice_loss True \                      # 启用Dice损失

  --focal_loss True                       # 启用Focal损失  

```

#### 关键参数说明：

| 参数名 | 含义 | 默认值 |

|-----------------|---------------------------------------|-----------------|

| `--data_rootr` | 数据集根目录路径 | 需自行设置 |

| `--backbone` | 骨干网络类型 | 60 |

| `--num_classes` | 类别数 | 4 |

| `--Init_lr` | 初始学习率 | 1e-3 |

| `--save_dir` | 训练权重保存目录（.h5格式） | `./logs` |

| `--Freeze_Epoch` | 冻结骨干训练轮数 | 50 |  

| `--UnFreeze_Epoch` | 全网络训练轮数 |1000 |

| `--Freeze_batch_size` | 冻结阶段批次大小 | 8 |

| `--Unfreeze_batch_size` | 解冻阶段批次大小 | 4 |

### 5.2 模型预测

使用训练好的权重在测试集上计算mIoU、Kappa等指标，运行get_miou.py脚本，示例命令：  

```bash

python get_miou.py  

```

#### 评估输出示例：

```

mIoU: 0.8748

Kappa coefficient: 0.8623

每类IoU: [0.9360, 0.8950, 0.8120, 0.8360]

```

## 6. 项目文件结构

```

Improved-DeepLabV3Plus-Potato/

├── deeplabv3_plus.py          # 改进后的DeepLabV3+主模型

├── tost.py                    # ToST双注意力模块

├── fdam.py                    # FDAM频域动态调制模块

├── mobilenetv2.py             # MobileNetV2骨干

├── train.py                       # 训练脚本

├── split.py                       # 数据集划分脚本

├── get_miou.py                    # 测试评估脚本（含Kappa）

└── detection-results-color.py     # 预测掩码彩色化

```

## 7. 已知问题与注意事项

1. **框架适配**：本项目基于 PyTorch 2.4.1 实现，不兼容 TensorFlow 环境，请确保使用 PyTorch 框架运行训练和评估脚本；；  

2. **输入尺寸**：模型固定输入为 256×256×3（可通过 input_shape 参数调整），训练和预测时会自动缩放图像，建议原始图像分辨率 ≥256×256 以保留病斑细节；  

3. **数据集扩展**：如需新增病害类别（如灰斑病），需在数据集中添加对应的图像与掩码，并修改模型定义中的 num_classes 参数（位于 train.py 和 nets/deeplabv3_plus.py），同时更新颜色映射和评估脚本中的类别设置。

## 8. 引用与联系方式

### 8.1 引用方式

论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：  

```bibtex

@article{potato_deeplab_2026,

  title={Research on Potato Leaf Disease Segmentation and Recognition Method Based on Improved DeepLabv3+},

  author={[作者姓名待补充]},

  journal={[期刊名称待补充]},

  year={2026},

  note={Manuscript submitted for publication}

}  

```

### 8.2 联系方式

若遇到代码运行问题或学术交流需求，请联系：  

- 邮箱：yukaidi@huuc.edu.cn  

- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
