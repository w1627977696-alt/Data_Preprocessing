# 用户指南 (User Guide)

## 目录
1. [简介](#简介)
2. [安装与配置](#安装与配置)
3. [模块详解](#模块详解)
4. [实战案例](#实战案例)
5. [常见问题](#常见问题)
6. [最佳实践](#最佳实践)

## 简介

本工具包是为RUL（剩余使用寿命）预测项目设计的综合数据预处理解决方案。它提供了从数据加载到模型输入准备的完整流程，特别适用于基于深度学习的预测性维护场景。

### 适用场景
- 工业设备健康监测
- 预测性维护
- 时序数据分析
- 退化建模
- 深度学习模型数据准备

## 安装与配置

### 系统要求
- Python 3.7 或更高版本
- 建议使用虚拟环境

### 安装步骤

1. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

### 验证安装

```python
import sys
sys.path.insert(0, 'src')

from data_preprocessing import DataLoader, StatisticalAnalyzer, CorrelationAnalyzer, DataPreprocessor
print("安装成功！")
```

## 模块详解

### 1. DataLoader - 数据加载器

#### 核心功能
数据加载器负责CSV文件的读取、筛选、合并和基本可视化。

#### 主要方法

##### read_csv() - 读取单个CSV文件
```python
loader = DataLoader()
data = loader.read_csv('data.csv', encoding='utf-8')
```

**参数说明：**
- `file_path`: CSV文件路径
- `encoding`: 文件编码，默认'utf-8'
- `**kwargs`: pandas.read_csv的其他参数

**使用场景：**
- 读取单个传感器数据文件
- 加载标准格式的时序数据

##### filter_columns() - 筛选列
```python
# 保留指定列
loader.filter_columns(columns=['sensor1', 'sensor2', 'sensor3'])

# 排除指定列
loader.filter_columns(exclude_columns=['noise_col'])
```

**使用场景：**
- 去除噪声特征
- 选择感兴趣的传感器

##### merge_data() - 合并数据
```python
# 垂直合并（添加更多行）
merged = loader.merge_data([df1, df2], method='concat', axis=0)

# 水平合并（添加更多列）
merged = loader.merge_data([df1, df2], method='concat', axis=1)

# 根据键合并
merged = loader.merge_data([df1, df2], method='merge', on='unit_id', how='inner')
```

**使用场景：**
- 合并多个设备的数据
- 整合不同时间段的数据
- 结合多个数据源

##### visualize_raw_data() - 可视化原始数据
```python
loader.visualize_raw_data(
    columns=['temp', 'vibration', 'pressure'],
    sample_size=1000,
    figsize=(15, 10),
    save_path='raw_data.png'
)
```

**使用场景：**
- 初步数据探索
- 识别数据模式
- 检测明显异常

### 2. StatisticalAnalyzer - 统计分析器

#### 核心功能
提供全面的统计分析，帮助理解数据特征和分布。

#### 主要方法

##### basic_statistics() - 基础统计
```python
analyzer = StatisticalAnalyzer(data)
stats = analyzer.basic_statistics()
```

**输出包括：**
- 均值、中位数、标准差
- 最小值、最大值、范围
- 百分位数
- 偏度、峰度
- 变异系数

**使用场景：**
- 了解数据的集中趋势和离散程度
- 比较不同特征的变化范围
- 识别潜在的数据问题

##### distribution_analysis() - 分布分析
```python
dist = analyzer.distribution_analysis()
```

**功能：**
- 正态性检验
- 分布类型判断
- 分布参数估计

**使用场景：**
- 确定是否需要数据变换
- 选择合适的统计方法
- 理解数据生成机制

##### outlier_detection() - 异常值检测
```python
# IQR方法
outliers_iqr = analyzer.outlier_detection(method='iqr')

# Z-score方法
outliers_z = analyzer.outlier_detection(method='zscore', threshold=3.0)
```

**使用场景：**
- 识别传感器故障
- 检测数据采集错误
- 发现异常工况

##### time_series_analysis() - 时序分析
```python
ts_stats = analyzer.time_series_analysis(window_size=10)
```

**功能：**
- 趋势分析
- 滑动统计
- 变化点检测

**使用场景：**
- 分析退化趋势
- 识别周期性模式
- 评估时间依赖性

### 3. CorrelationAnalyzer - 相关性分析器

#### 核心功能
多方法相关性分析，支持特征选择和关系可视化。

#### 主要方法

##### pearson_correlation() - Pearson相关
```python
corr_analyzer = CorrelationAnalyzer(data)

# 完整相关矩阵
corr_matrix = corr_analyzer.pearson_correlation()

# 与目标的相关性
target_corr = corr_analyzer.pearson_correlation(target='RUL')
```

**适用于：**
- 线性关系检测
- 快速特征筛选

##### spearman_correlation() - Spearman相关
```python
spearman = corr_analyzer.spearman_correlation(target='RUL')
```

**适用于：**
- 单调关系检测
- 非线性关系初步评估

##### mutual_information() - 互信息
```python
mi = corr_analyzer.mutual_information(target='RUL', n_neighbors=3)
```

**适用于：**
- 捕获非线性关系
- 复杂依赖关系分析

##### feature_selection() - 特征选择
```python
# 按阈值选择
selected = corr_analyzer.feature_selection(
    target='RUL',
    method='pearson',
    threshold=0.3
)

# 选择Top K特征
selected = corr_analyzer.feature_selection(
    target='RUL',
    method='pearson',
    top_k=10
)
```

**使用场景：**
- 降低模型复杂度
- 提高训练效率
- 减少过拟合风险

### 4. DataPreprocessor - 数据预处理器

#### 核心功能
完整的数据预处理流程，从清洗到模型输入准备。

#### 主要方法

##### handle_missing_values() - 处理缺失值
```python
preprocessor = DataPreprocessor(data)

# 均值插补
preprocessor.handle_missing_values(strategy='mean')

# KNN插补
preprocessor.handle_missing_values(strategy='knn')

# 前向填充
preprocessor.handle_missing_values(strategy='ffill')

# 插值
preprocessor.handle_missing_values(strategy='interpolate', method='linear')
```

**策略选择指南：**
- `mean/median`: 适用于正态分布数据
- `knn`: 考虑特征间关系
- `ffill/bfill`: 保持时序连续性
- `interpolate`: 平滑时序数据

##### normalize() - 标准化
```python
# Z-score标准化
preprocessor.normalize(method='standard')

# Min-Max归一化
preprocessor.normalize(method='minmax', feature_range=(0, 1))

# 鲁棒标准化
preprocessor.normalize(method='robust')
```

**方法选择指南：**
- `standard`: 默认选择，假设数据近似正态分布
- `minmax`: 需要固定范围时使用
- `robust`: 数据含异常值时使用

##### construct_rul_labels() - 构建RUL标签
```python
preprocessor.construct_rul_labels(
    unit_column='unit_id',
    time_column='cycle',
    max_rul=125
)
```

**参数说明：**
- `unit_column`: 设备/单元标识列
- `time_column`: 时间/周期列（可选）
- `max_rul`: 最大RUL值（用于截断）

**使用场景：**
- 自动生成RUL标签
- 处理截尾问题
- 统一标签尺度

##### construct_health_indicator() - 构建健康指标
```python
# PCA方法
preprocessor.construct_health_indicator(
    feature_columns=['sensor1', 'sensor2', 'sensor3'],
    method='pca_first'
)

# 加权和方法
preprocessor.construct_health_indicator(
    feature_columns=['sensor1', 'sensor2', 'sensor3'],
    method='weighted_sum',
    weights=[0.5, 0.3, 0.2]
)

# 单调性方法
preprocessor.construct_health_indicator(
    feature_columns=['sensor1', 'sensor2', 'sensor3'],
    method='monotonicity'
)
```

**方法选择指南：**
- `pca_first`: 无先验知识时的默认选择
- `weighted_sum`: 有领域知识时使用
- `monotonicity`: 关注退化趋势时使用

##### create_sliding_windows() - 创建滑动窗口
```python
X, y = preprocessor.create_sliding_windows(
    window_size=30,
    step_size=1,
    feature_columns=['sensor1', 'sensor2'],
    target_column='RUL'
)
```

**参数调优：**
- `window_size`: 取决于时序模式的时间尺度
- `step_size`: 权衡数据量和计算效率
  - step_size=1: 最大数据量，计算开销大
  - step_size=window_size: 无重叠，数据量少

**输出格式：**
- X: (样本数, 窗口大小, 特征数)
- y: (样本数,) 或 (样本数, 1)

## 实战案例

### 案例1：涡扇发动机RUL预测

```python
# 1. 加载数据
loader = DataLoader()
data = loader.read_csv('turbofan_FD001.csv')

# 2. 探索性分析
analyzer = StatisticalAnalyzer(data)
analyzer.basic_statistics()
analyzer.visualize_distribution(save_path='dist.png')

# 3. 相关性分析和特征选择
corr_analyzer = CorrelationAnalyzer(data)
selected_features = corr_analyzer.feature_selection(
    target='RUL',
    method='pearson',
    top_k=14
)

# 4. 数据预处理
preprocessor = DataPreprocessor(data)
preprocessor.handle_missing_values(strategy='mean')
preprocessor.remove_outliers(method='iqr')
preprocessor.normalize(columns=selected_features, method='standard')

# 5. 构建RUL标签
preprocessor.construct_rul_labels(
    unit_column='unit',
    time_column='cycle',
    max_rul=125
)

# 6. 划分数据集
train_data, test_data = preprocessor.split_by_unit('unit', train_ratio=0.8)

# 7. 创建滑动窗口
train_prep = DataPreprocessor(train_data)
X_train, y_train = train_prep.create_sliding_windows(
    window_size=30,
    step_size=1,
    feature_columns=selected_features,
    target_column='RUL'
)

print(f"训练数据: X={X_train.shape}, y={y_train.shape}")
# 现在可以用于LSTM/GRU模型训练
```

### 案例2：轴承退化分析

```python
# 1. 加载多个轴承数据
loader = DataLoader()
bearing_files = ['bearing1.csv', 'bearing2.csv', 'bearing3.csv']
data_dict = loader.read_multiple_csv(bearing_files)

# 2. 合并数据
data_list = []
for i, (name, df) in enumerate(data_dict.items(), 1):
    df['bearing_id'] = i
    data_list.append(df)

merged_data = loader.merge_data(data_list, method='concat', axis=0)

# 3. 构建健康指标
preprocessor = DataPreprocessor(merged_data)
preprocessor.construct_health_indicator(
    feature_columns=['vibration_x', 'vibration_y', 'temperature'],
    method='pca_first'
)

# 4. 可视化健康指标趋势
import matplotlib.pyplot as plt
for bearing_id in merged_data['bearing_id'].unique():
    bearing_data = merged_data[merged_data['bearing_id'] == bearing_id]
    plt.plot(bearing_data.index, bearing_data['health_indicator'], 
             label=f'Bearing {bearing_id}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Health Indicator')
plt.title('Bearing Health Degradation')
plt.savefig('health_indicator.png')
plt.show()
```

## 常见问题

### Q1: 如何处理不同单元周期长度不同的情况？
**A:** RUL标签构建会自动处理。每个单元独立计算RUL，不受其他单元影响。

### Q2: 滑动窗口大小如何选择？
**A:** 
- 考虑物理意义：覆盖关键退化过程的时间尺度
- 经验值：10-50个时间步
- 实验验证：尝试多个值，根据模型性能选择

### Q3: 数据量很大时如何提高处理速度？
**A:**
- 增大滑动窗口步长
- 使用数据采样
- 并行处理多个单元
- 考虑使用Dask处理大数据

### Q4: 如何处理高度不平衡的RUL分布？
**A:**
- 使用max_rul参数截断
- 分段RUL标签（早期/中期/晚期）
- 采样平衡技术
- 使用加权损失函数

### Q5: 缺失值很多时如何处理？
**A:**
- 缺失率<5%: 任意插补方法
- 5-20%: KNN或插值
- >20%: 考虑删除该特征或使用深度学习方法

## 最佳实践

### 1. 数据预处理流程建议

```
加载数据 → 初步探索 → 异常值处理 → 缺失值处理 
→ 特征选择 → 标准化 → 标签构建 → 滑动窗口 → 数据集划分
```

### 2. 特征工程建议

- **保留原始特征**: 深度学习可以自动学习特征
- **构建健康指标**: 作为额外特征输入
- **考虑领域知识**: 物理意义的特征组合
- **时间特征**: 周期、阶段等

### 3. 可视化建议

- 每个步骤后都进行可视化验证
- 对比处理前后的数据分布
- 检查标签分布的合理性
- 可视化滑动窗口样本

### 4. 代码组织建议

```python
# 创建处理流程类
class RULPreprocessingPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.analyzer = None
        self.corr_analyzer = None
        self.preprocessor = None
    
    def load_data(self, file_path):
        self.data = self.loader.read_csv(file_path)
        return self.data
    
    def analyze(self):
        self.analyzer = StatisticalAnalyzer(self.data)
        self.analyzer.generate_report('report.xlsx')
    
    def preprocess(self, config):
        self.preprocessor = DataPreprocessor(self.data)
        # 根据config执行预处理步骤
        ...
    
    def prepare_model_input(self, window_size):
        X, y = self.preprocessor.create_sliding_windows(
            window_size=window_size,
            ...
        )
        return X, y

# 使用
pipeline = RULPreprocessingPipeline()
pipeline.load_data('data.csv')
pipeline.analyze()
pipeline.preprocess(config)
X, y = pipeline.prepare_model_input(window_size=30)
```

### 5. 性能优化建议

- 使用`sample_size`参数减少可视化数据量
- 批量处理多个文件
- 缓存中间结果
- 使用更高效的数据格式（Parquet, HDF5）

---

更多信息请参考完整文档或提交Issue。
