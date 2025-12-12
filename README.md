# 数据预处理工具包 (Data Preprocessing Toolkit)

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

用于RUL（剩余使用寿命）预测的多元时序数据分析和预处理工具包。

## 📋 目录

- [功能特性](#功能特性)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [详细文档](#详细文档)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [依赖项](#依赖项)
- [贡献指南](#贡献指南)

## 🎯 功能特性

本工具包专为基于深度学习的RUL预测项目设计，提供完整的数据预处理流程：

### 1. 数据加载与管理 (`DataLoader`)
- ✅ 单个/多个CSV文件读取
- ✅ 数据筛选和过滤（行/列）
- ✅ 多文件数据合并（concat/merge）
- ✅ 原始数据可视化
- ✅ 数据信息摘要

### 2. 统计特性分析 (`StatisticalAnalyzer`)
- ✅ 基础统计量（均值、方差、极值等）
- ✅ 分布特征分析（偏度、峰度、正态性检验）
- ✅ 缺失值分析
- ✅ 异常值检测（IQR、Z-score方法）
- ✅ 时序特性分析
- ✅ 多种统计可视化（分布图、箱线图、热图）
- ✅ 生成综合统计报告

### 3. 相关性分析 (`CorrelationAnalyzer`)
- ✅ 多种相关性方法：
  - Pearson相关（线性相关）
  - Spearman相关（单调相关）
  - Kendall相关（适用于小样本）
  - 互信息（捕获非线性关系）
- ✅ 目标变量相关性分析
- ✅ 特征选择和排序
- ✅ 相关性可视化（热图、柱状图、散点矩阵）
- ✅ 生成相关性分析表格

### 4. 数据预处理 (`DataPreprocessor`)
- ✅ 数据清洗（去重、去异常值）
- ✅ 多种缺失值处理方法：
  - 统计插补（均值、中位数、众数）
  - 前向/后向填充
  - 插值法
  - KNN插补
- ✅ 数据标准化/归一化：
  - Z-score标准化
  - Min-Max归一化
  - 鲁棒标准化
- ✅ 滑动窗口构建
- ✅ RUL标签自动构建
- ✅ 健康因子构建：
  - PCA主成分法
  - 加权求和法
  - 单调性分析法
- ✅ 数据集划分

## 📦 安装说明

### 环境要求
- Python 3.7+
- pip

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/w1627977696-alt/Data_Preprocessing.git
cd Data_Preprocessing
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 基础使用示例

```python
import sys
sys.path.insert(0, 'src')

from data_preprocessing import DataLoader, StatisticalAnalyzer, CorrelationAnalyzer, DataPreprocessor

# 1. 加载数据
loader = DataLoader()
data = loader.read_csv('data/raw/your_data.csv')
loader.display_info()

# 2. 统计分析
analyzer = StatisticalAnalyzer(data)
analyzer.basic_statistics()
analyzer.visualize_distribution()

# 3. 相关性分析
corr_analyzer = CorrelationAnalyzer(data)
corr_analyzer.pearson_correlation()
corr_analyzer.visualize_correlation_matrix()

# 4. 数据预处理
preprocessor = DataPreprocessor(data)
preprocessor.handle_missing_values(strategy='mean')
preprocessor.normalize(method='standard')
preprocessor.construct_rul_labels(unit_column='unit')
```

### 运行完整示例

```bash
cd examples
python example_usage.py
```

这将运行一个完整的示例，展示所有功能模块的使用。

## 📚 详细文档

### 数据加载器 (DataLoader)

```python
from data_preprocessing import DataLoader

loader = DataLoader()

# 读取CSV文件
data = loader.read_csv('data.csv', encoding='utf-8')

# 读取多个CSV文件
data_dict = loader.read_multiple_csv(['file1.csv', 'file2.csv'])

# 筛选列
loader.filter_columns(columns=['col1', 'col2', 'col3'])

# 筛选行
loader.filter_rows(condition='col1 > 10', start_index=0, end_index=1000)

# 合并数据
merged_data = loader.merge_data([df1, df2], method='concat', axis=0)

# 可视化
loader.visualize_raw_data(columns=['sensor1', 'sensor2'], save_path='plot.png')
```

### 统计分析器 (StatisticalAnalyzer)

```python
from data_preprocessing import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(data)

# 基础统计
basic_stats = analyzer.basic_statistics()

# 分布分析
dist_analysis = analyzer.distribution_analysis()

# 缺失值分析
missing = analyzer.missing_value_analysis()

# 异常值检测
outliers = analyzer.outlier_detection(method='iqr')

# 时序分析
ts_stats = analyzer.time_series_analysis(window_size=10)

# 可视化
analyzer.visualize_distribution(save_path='dist.png')
analyzer.visualize_boxplot(save_path='boxplot.png')
analyzer.visualize_correlation_heatmap(save_path='heatmap.png')

# 生成报告
report = analyzer.generate_report(save_path='report.xlsx')
```

### 相关性分析器 (CorrelationAnalyzer)

```python
from data_preprocessing import CorrelationAnalyzer

corr_analyzer = CorrelationAnalyzer(data)

# Pearson相关性
pearson = corr_analyzer.pearson_correlation()

# 与目标变量的相关性
target_corr = corr_analyzer.correlation_with_target(
    target='RUL',
    methods=['pearson', 'spearman', 'mutual_info']
)

# 特征选择
selected = corr_analyzer.feature_selection(
    target='RUL',
    method='pearson',
    top_k=10
)

# 可视化
corr_analyzer.visualize_correlation_matrix(save_path='corr_matrix.png')
corr_analyzer.visualize_correlation_with_target(
    target='RUL',
    save_path='corr_target.png'
)

# 生成表格
table = corr_analyzer.generate_correlation_table(
    target='RUL',
    save_path='correlation_table.csv'
)
```

### 数据预处理器 (DataPreprocessor)

```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(data)

# 数据清洗
preprocessor.remove_duplicates()
preprocessor.remove_outliers(method='iqr')

# 缺失值处理
preprocessor.handle_missing_values(
    strategy='mean',  # 'mean', 'median', 'mode', 'ffill', 'bfill', 'interpolate', 'knn'
    columns=['sensor1', 'sensor2']
)

# 标准化
preprocessor.normalize(
    columns=['sensor1', 'sensor2'],
    method='standard'  # 'standard', 'minmax', 'robust'
)

# 构建RUL标签
preprocessor.construct_rul_labels(
    unit_column='unit',
    time_column='cycle',
    max_rul=125
)

# 构建健康指标
preprocessor.construct_health_indicator(
    feature_columns=['sensor1', 'sensor2', 'sensor3'],
    method='pca_first'  # 'pca_first', 'weighted_sum', 'monotonicity'
)

# 创建滑动窗口
X, y = preprocessor.create_sliding_windows(
    window_size=30,
    step_size=1,
    feature_columns=['sensor1', 'sensor2'],
    target_column='RUL'
)

# 划分数据集
train_data, test_data = preprocessor.split_by_unit(
    unit_column='unit',
    train_ratio=0.8
)

# 导出数据
preprocessor.export_processed_data('processed_data.csv')
```

## 💡 使用示例

### 示例1：完整的RUL预测数据预处理流程

```python
import pandas as pd
from data_preprocessing import DataLoader, CorrelationAnalyzer, DataPreprocessor

# 1. 加载数据
loader = DataLoader()
data = loader.read_csv('turbofan_data.csv')

# 2. 分析与目标的相关性
corr_analyzer = CorrelationAnalyzer(data)
selected_features = corr_analyzer.feature_selection(
    target='RUL',
    method='pearson',
    top_k=15
)

# 3. 预处理
preprocessor = DataPreprocessor(data)
preprocessor.handle_missing_values(strategy='mean')
preprocessor.construct_rul_labels(unit_column='unit', max_rul=125)
preprocessor.normalize(columns=selected_features, method='standard')

# 4. 划分数据集并创建滑动窗口
train_data, test_data = preprocessor.split_by_unit('unit', train_ratio=0.8)

train_preprocessor = DataPreprocessor(train_data)
X_train, y_train = train_preprocessor.create_sliding_windows(
    window_size=30,
    feature_columns=selected_features,
    target_column='RUL'
)

# 现在可以用于模型训练
print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
```

### 示例2：健康因子构建

```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(data)

# 方法1: PCA主成分法
preprocessor.construct_health_indicator(
    feature_columns=['temp', 'vibration', 'pressure'],
    method='pca_first'
)

# 方法2: 加权求和法
preprocessor.construct_health_indicator(
    feature_columns=['temp', 'vibration', 'pressure'],
    method='weighted_sum',
    weights=[0.5, 0.3, 0.2]
)

# 方法3: 单调性分析法
preprocessor.construct_health_indicator(
    feature_columns=['temp', 'vibration', 'pressure'],
    method='monotonicity'
)
```

## 📁 项目结构

```
Data_Preprocessing/
├── src/
│   └── data_preprocessing/
│       ├── __init__.py              # 包初始化
│       ├── data_loader.py           # 数据加载模块
│       ├── statistical_analysis.py  # 统计分析模块
│       ├── correlation_analysis.py  # 相关性分析模块
│       └── preprocessing.py         # 数据预处理模块
├── examples/
│   └── example_usage.py             # 使用示例
├── data/
│   ├── raw/                         # 原始数据
│   └── processed/                   # 处理后的数据
├── docs/                            # 文档
├── tests/                           # 测试文件
├── requirements.txt                 # 依赖项
└── README.md                        # 本文件
```

## 📋 依赖项

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
openpyxl>=3.0.0
```

## 🎨 特性亮点

### 代码质量
- ✨ 清晰的代码逻辑和结构
- 📝 详细的中文注释和文档字符串
- 🔧 模块化设计，易于扩展
- 🎯 面向对象的设计模式

### 功能完整性
- 📊 涵盖数据预处理全流程
- 🔍 多种分析方法可选
- 📈 丰富的可视化功能
- 💾 支持多种数据格式导出

### 易用性
- 🚀 简洁的API设计
- 📖 完整的使用示例
- 🎓 详细的使用文档
- ⚡ 开箱即用

## 📝 使用场景

本工具包特别适用于以下场景：

1. **设备剩余寿命预测**
   - 涡扇发动机RUL预测
   - 轴承寿命预测
   - 电池健康状态评估

2. **时序数据分析**
   - 工业传感器数据分析
   - 设备监测数据处理
   - 退化趋势分析

3. **深度学习预处理**
   - LSTM/GRU模型数据准备
   - CNN-LSTM模型输入构建
   - Transformer模型数据预处理

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 👥 作者

Data Analysis Expert

## 🙏 致谢

感谢所有为本项目做出贡献的开发者！

## 📮 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发起 Pull Request

---

**注意**: 本工具包专为RUL预测场景设计，但也可用于其他时序数据分析任务。根据具体需求调整参数即可。
