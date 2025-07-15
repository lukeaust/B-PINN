# B-PINN: 基于业务流程信息约束神经网络的剩余时间预测

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-%231A2B5F.svg?style=flat)](https://optuna.org/)

**状态: 论文投稿中 (Paper Under Submission)**

> **注意**: 本项目为支持相关学术论文而开发。在论文被接收并公开发表之前，源代码将保持私有。

---

## 1. 项目简介与核心思想

本代码库是论文 **《基于业务流程信息约束神经网络的剩余时间预测》(Remaining Time Prediction Based on Business-Process-Informed Neural Networks, B-PINN)** 的官方PyTorch实现。

我们的工作引入了B-PINN，一个将领域知识注入深度学习模型的新颖框架，用于业务流程剩余时间预测。通过将流程规则形式化为“流程物理”并集成到损失函数中，B-PINN在预测精度和鲁棒性上均取得了显著提升，尤其是在数据稀疏的场景下。

## 2. 框架核心特性

- **模块化代码库**: 所有核心逻辑（数据处理、模型、损失、训练引擎）均被封装在`src/`目录下的独立模块中，清晰且易于维护。
- **配置驱动的实验**: 所有实验均由一个独立的`experiments.yaml`文件定义，实现了代码与实验配置的完全分离，易于管理和扩展。
- **端到端自动化**: 提供一个总控脚本`run_master.py`，可一键完成**数据预处理、多组实验训练、结果记录和图表生成**的全过程。
- **高级超参搜索**: 内置基于`Optuna`的自动化超参数搜索脚本，支持持久化存储、中断续跑和智能剪枝。
- **出版物级可视化**: 自动生成一套信息密度高、富有洞察力的分析图表，可直接用于论文撰写。
- **交互式分析与洞察**: 提供独立的Jupyter Notebook，可连接到超参数搜索数据库，实时、交互式地探索优化过程。

## 3. 项目结构与文件功能

```

B-PINN-RTP/
│
├── data/
│   ├── raw/                  \# [输入] 存放所有原始数据 (.xes.csv)
│   └── processed/            \# [输出] 存放预处理后的数据 (.pkl) 和归一化模型 (*scaler.pkl)
│
├── models/                   \# [输出] 按实验组存放训练好的模型权重 (.pth)
│
├── paper\_figures/            \# [输出] 存放最终生成的论文图表 (.png)
│   └── optuna\_reports/     \#   [输出] 存放Optuna生成的HTML分析报告
│
├── results/
│   ├── bpic2012\_final\_runs/  \# [输出] 按实验组存放结果的子文件夹
│   │   ├── summary.csv       \#   该实验组的聚合结果
│   │   └── predictions*\*.csv \#   该实验组每次运行的详细预测日志
│   └── \*.db                \# [输出] Optuna超参搜索的数据库文件
│
├── src/                      \# 【核心代码目录】
│   ├── utils/
│   │   └── xes\_to\_csv.py     \# (可选) XES到CSV的转换工具
│   ├── preprocessing.py      \# 【模块一】数据预处理脚本
│   ├── dataset.py            \# 【模块二】PyTorch数据集类
│   ├── model.py              \# 【模块三】B-PINN模型架构
│   ├── loss.py               \# 【模块四】"流程物理"损失函数
│   ├── engine.py             \# 【模块五】训练与评估引擎
│   ├── main.py               \# 【模块六】单次实验执行入口
│   ├── visualize2.py         \# 【模块七】实验结果可视化脚本
│   └── hyperparameter\_search\_advanced.py \# 【模块八】高级自动化超参数搜索脚本
│
├── experiments.yaml          \# 【核心】实验配置文件，定义所有实验
│
├── run\_master.py               \# 【核心】项目总控脚本，自动化执行所有任务
│
├── notebooks/
│   └── analyze\_tuning\_results.ipynb \# 【核心】超参搜索结果的交互式分析笔记本
│
├── requirements.txt            \# 项目Python依赖库
│
└── README.md                   \# 项目说明文档

````

## 4. 环境配置与安装

### 环境要求

- **Anaconda/Miniconda**
- **Python**: 3.9 或更高版本
- **NVIDIA GPU** + **CUDA**: 强烈推荐

### 安装步骤

1.  **克隆本代码库**。
2.  **使用Conda创建并激活环境**:
    ```bash
    conda create -n bpinn python=3.9 -y
    conda activate bpinn
    ```
3.  **安装PyTorch (GPU版)**:
    访问 [PyTorch官网](https://pytorch.org/get-started/locally/)，根据您系统的CUDA版本，获取并运行对应的`conda`安装命令。

4.  **安装其他依赖**:
    ```bash
    pip install -r requirements.txt
    ```

## 5. 使用指南：三大工作流模式

本框架支持多种工作模式，从全自动到精细手动，满足不同需求。

### **模式一：一键自动化工作流 (推荐)**

这是最主要、最便捷的使用方式，能够一键复现论文中的所有实验并生成图表。

1.  **准备数据**: 将原始数据文件 (如 `bpic2012.xes.csv`) 放置在 `data/raw/` 目录下。
2.  **定义实验**: (可选) 打开 `experiments.yaml` 文件，修改或添加您想运行的实验配置。
3.  **一键执行**: 在项目根目录的终端中，运行总控脚本：
    ```bash
    python run_master.py
    ```
    - **只运行特定实验组**: `python run_master.py <实验组名称1> <实验组名称2>`
    - **跳过实验，仅重新生成图表**: `python run_master.py --skip_experiments`

### **模式二：自动化超参数搜索工作流 (进阶用法)**

当您需要为您的模型系统性地寻找最优超参数组合时，请使用`src/hyperparameter_search_advanced.py`脚本。它集成了**Hyperband剪枝**和**并行化**等高级策略，能以数倍的效率进行搜索。

1.  **确保数据已预处理**。
2.  **启动搜索**:
    ```bash
    # 示例：为 bpic2012 的 LSTM 模型运行 100 次超参数搜索试验，并使用所有CPU核心并行加速
    python src/hyperparameter_search_advanced.py --dataset_name bpic2012 --model_type lstm --study_name bpic2012_lstm_tuning --n_trials 100 --n_jobs -1
    ```
3.  **实时监控与分析结果**:
    - **实时日志**: 脚本会**每完成一次试验，就在终端打印一行日志**，让您随时了解进度。
    - **交互式分析**: 在搜索**正在进行时**或结束后，打开 `notebooks/analyze_tuning_results.ipynb`，修改`STUDY_NAME`为您正在运行的研究名称，然后运行所有单元格。您将看到**可交互的**优化历史、参数重要性等图表。
4.  **应用最优参数**: 将找到的最佳参数更新到您的`experiments.yaml`配置文件中，然后使用**模式一**来运行最终实验。

### **模式三：手动分步工作流 (用于调试或快速验证)**

如果您想深入细节、进行调试，或者只想运行一个临时的实验，可以按照以下步骤手动操作。

1.  **预处理数据** (只需对每个数据集运行一次):
    ```bash
    python src/preprocessing.py --input data/raw/bpic2012.xes.csv --output_dir data/processed/
    ```

2.  **运行单次训练**: 直接调用`main.py`并指定所有必需的参数。
    ```bash
    # 示例：运行一个完整的B-PINN模型
    # 在Windows的CMD中，可以用^换行；在PowerShell中，用`；在Linux/macOS中，用\
    python src/main.py ^
        --dataset_name bpic2012 ^
        --data_dir data/processed/ ^
        --output_dir models/manual_runs/ ^
        --results_dir results/manual_runs/ ^
        --tag manual_bpinn_test ^
        --lr 0.001 ^
        --lambda_mono 0.01 ^
        --lambda_bound 0.01
    ```
3.  **手动生成图表** (用于检查样式):
    ```bash
    # 使用内置的模拟数据，快速生成所有图表以检查样式
    python src/visualize2.py --use-mock-data
    ```

## 6. 引用

如果您在您的研究中发现这项工作有用，请考虑引用我们的论文：

```bibtex
@inproceedings{yourname2025bpinn,
  title={Remaining Time Prediction Based on Business-Process-Informed Neural Networks (B-PINN)},
  author={Your, Name and Co-author, Name},
  booktitle={Proceedings of the ... Conference ...},
  year={2025}
}
````

## 7\. License

This project is distributed under the MIT License.

