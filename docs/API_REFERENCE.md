# API 参考文档

## 模块概览

```python
from data_preprocessing import (
    DataLoader,           # 数据加载
    StatisticalAnalyzer,  # 统计分析
    CorrelationAnalyzer,  # 相关性分析
    DataPreprocessor      # 数据预处理
)
```

---

## DataLoader - 数据加载器

### 初始化

```python
loader = DataLoader()
```

### 方法

#### read_csv(file_path, encoding='utf-8', **kwargs)
读取单个CSV文件

**参数:**
- `file_path` (str): CSV文件路径
- `encoding` (str): 文件编码，默认'utf-8'
- `**kwargs`: pandas.read_csv的其他参数

**返回:** `pd.DataFrame`

#### read_multiple_csv(file_paths, encoding='utf-8', **kwargs)
读取多个CSV文件

**参数:**
- `file_paths` (List[str]): CSV文件路径列表
- `encoding` (str): 文件编码
- `**kwargs`: pandas.read_csv的其他参数

**返回:** `Dict[str, pd.DataFrame]`

#### filter_columns(columns=None, exclude_columns=None)
筛选数据列

**参数:**
- `columns` (List[str], optional): 要保留的列名列表
- `exclude_columns` (List[str], optional): 要排除的列名列表

**返回:** `pd.DataFrame`

#### filter_rows(condition=None, start_index=None, end_index=None)
筛选数据行

**参数:**
- `condition` (str, optional): 筛选条件（query表达式）
- `start_index` (int, optional): 起始索引
- `end_index` (int, optional): 结束索引

**返回:** `pd.DataFrame`

#### merge_data(data_list, method='concat', on=None, how='outer', axis=0)
合并多个数据框

**参数:**
- `data_list` (List[pd.DataFrame]): 要合并的数据框列表
- `method` (str): 合并方法 ('concat' 或 'merge')
- `on` (str|List[str], optional): merge方法的连接键
- `how` (str): merge方法的连接方式
- `axis` (int): concat方法的轴向 (0=行, 1=列)

**返回:** `pd.DataFrame`

#### visualize_raw_data(columns=None, sample_size=None, figsize=(15,10), save_path=None)
可视化原始数据

**参数:**
- `columns` (List[str], optional): 要可视化的列名列表
- `sample_size` (int, optional): 采样大小
- `figsize` (tuple): 图形大小
- `save_path` (str, optional): 保存路径

---

## StatisticalAnalyzer - 统计分析器

### 初始化

```python
analyzer = StatisticalAnalyzer(data)
```

**参数:**
- `data` (pd.DataFrame): 要分析的数据框

### 方法

#### basic_statistics(percentiles=[0.25, 0.5, 0.75])
计算基础统计量

**参数:**
- `percentiles` (List[float]): 要计算的百分位数列表

**返回:** `pd.DataFrame`

#### distribution_analysis()
分布特征分析

**返回:** `Dict[str, pd.DataFrame]`

#### missing_value_analysis()
缺失值分析

**返回:** `pd.DataFrame`

#### outlier_detection(method='iqr', threshold=3.0)
异常值检测

**参数:**
- `method` (str): 检测方法 ('iqr' 或 'zscore')
- `threshold` (float): 阈值（zscore方法使用）

**返回:** `Dict[str, pd.DataFrame]`

#### time_series_analysis(window_size=10)
时序特性分析

**参数:**
- `window_size` (int): 滑动窗口大小

**返回:** `Dict[str, pd.DataFrame]`

#### visualize_distribution(columns=None, figsize=(15,10), save_path=None)
分布可视化（直方图+KDE）

#### visualize_boxplot(columns=None, figsize=(15,6), save_path=None)
箱线图可视化

#### visualize_correlation_heatmap(method='pearson', figsize=(12,10), save_path=None)
相关性热图

#### generate_report(save_path=None)
生成综合统计分析报告

**返回:** `Dict`

---

## CorrelationAnalyzer - 相关性分析器

### 初始化

```python
corr_analyzer = CorrelationAnalyzer(data)
```

**参数:**
- `data` (pd.DataFrame): 要分析的数据框

### 方法

#### pearson_correlation(target=None)
Pearson相关性分析（线性相关）

**参数:**
- `target` (str, optional): 目标变量名

**返回:** `pd.DataFrame`

#### spearman_correlation(target=None)
Spearman相关性分析（单调相关）

**参数:**
- `target` (str, optional): 目标变量名

**返回:** `pd.DataFrame`

#### kendall_correlation(target=None)
Kendall相关性分析（适用于小样本）

**参数:**
- `target` (str, optional): 目标变量名

**返回:** `pd.DataFrame`

#### mutual_information(target, n_neighbors=3, random_state=42)
互信息分析（捕获非线性关系）

**参数:**
- `target` (str): 目标变量名
- `n_neighbors` (int): 最近邻数量
- `random_state` (int): 随机种子

**返回:** `pd.DataFrame`

#### correlation_with_target(target, methods=['pearson', 'spearman', 'mutual_info'])
使用多种方法分析与目标变量的相关性

**参数:**
- `target` (str): 目标变量名
- `methods` (List[str]): 要使用的方法列表

**返回:** `pd.DataFrame`

#### feature_selection(target, method='pearson', threshold=None, top_k=None)
基于相关性的特征选择

**参数:**
- `target` (str): 目标变量名
- `method` (str): 相关性方法
- `threshold` (float, optional): 相关性阈值
- `top_k` (int, optional): 选择前k个特征

**返回:** `List[str]`

#### visualize_correlation_matrix(method='pearson', figsize=(12,10), save_path=None)
可视化相关性矩阵

#### visualize_correlation_with_target(target, method='pearson', top_k=15, figsize=(12,8), save_path=None)
可视化与目标变量的相关性

#### generate_correlation_table(target, methods=['pearson', 'spearman', 'mutual_info'], save_path=None)
生成综合相关性分析表格

**返回:** `pd.DataFrame`

---

## DataPreprocessor - 数据预处理器

### 初始化

```python
preprocessor = DataPreprocessor(data)
```

**参数:**
- `data` (pd.DataFrame): 要预处理的数据框

### 方法

#### remove_duplicates(subset=None, keep='first')
删除重复行

**参数:**
- `subset` (List[str], optional): 用于识别重复的列
- `keep` (str): 保留哪个重复值 ('first', 'last', False)

**返回:** `pd.DataFrame`

#### remove_outliers(columns=None, method='iqr', threshold=3.0)
删除异常值

**参数:**
- `columns` (List[str], optional): 要处理的列
- `method` (str): 异常值检测方法 ('iqr', 'zscore', 'percentile')
- `threshold` (float): 阈值

**返回:** `pd.DataFrame`

#### handle_missing_values(strategy='mean', columns=None, fill_value=None, method=None)
处理缺失值

**参数:**
- `strategy` (str): 插补策略 ('mean', 'median', 'mode', 'constant', 'ffill', 'bfill', 'interpolate', 'knn')
- `columns` (List[str], optional): 要处理的列
- `fill_value` (Any, optional): 当strategy='constant'时使用的填充值
- `method` (str, optional): 插值方法（当strategy='interpolate'时）

**返回:** `pd.DataFrame`

#### normalize(columns=None, method='standard', feature_range=(0,1))
数据标准化/归一化

**参数:**
- `columns` (List[str], optional): 要标准化的列
- `method` (str): 标准化方法 ('standard', 'minmax', 'robust')
- `feature_range` (Tuple[float, float]): MinMax缩放的范围

**返回:** `pd.DataFrame`

#### create_sliding_windows(window_size, step_size=1, feature_columns=None, target_column=None)
创建滑动窗口数据

**参数:**
- `window_size` (int): 窗口大小
- `step_size` (int): 步长
- `feature_columns` (List[str], optional): 特征列
- `target_column` (str, optional): 目标列

**返回:** `Tuple[np.ndarray, Optional[np.ndarray]]` - (X, y)

#### construct_rul_labels(unit_column, time_column=None, max_rul=None)
构建RUL（剩余使用寿命）标签

**参数:**
- `unit_column` (str): 单元/设备标识列
- `time_column` (str, optional): 时间列
- `max_rul` (int, optional): 最大RUL值（用于截断）

**返回:** `pd.DataFrame`

#### construct_health_indicator(feature_columns, method='pca_first', weights=None)
构建健康因子/健康指标

**参数:**
- `feature_columns` (List[str]): 用于构建健康指标的特征列
- `method` (str): 构建方法 ('pca_first', 'weighted_sum', 'monotonicity')
- `weights` (List[float], optional): 加权和方法的权重

**返回:** `pd.DataFrame`

#### split_by_unit(unit_column, train_ratio=0.8)
按单元划分训练集和测试集

**参数:**
- `unit_column` (str): 单元标识列
- `train_ratio` (float): 训练集比例

**返回:** `Tuple[pd.DataFrame, pd.DataFrame]` - (train_data, test_data)

#### export_processed_data(file_path)
导出处理后的数据

**参数:**
- `file_path` (str): 保存路径

#### get_processing_log()
获取处理日志

**返回:** `List[str]`

#### print_processing_summary()
打印处理摘要

---

## 使用示例

### 完整流程示例

```python
from data_preprocessing import *

# 1. 加载
loader = DataLoader()
data = loader.read_csv('data.csv')

# 2. 分析
analyzer = StatisticalAnalyzer(data)
stats = analyzer.basic_statistics()

# 3. 相关性
corr_analyzer = CorrelationAnalyzer(data)
features = corr_analyzer.feature_selection(target='RUL', top_k=10)

# 4. 预处理
preprocessor = DataPreprocessor(data)
preprocessor.handle_missing_values(strategy='mean')
preprocessor.normalize(method='standard')
preprocessor.construct_rul_labels(unit_column='unit')

# 5. 滑窗
X, y = preprocessor.create_sliding_windows(
    window_size=30,
    feature_columns=features,
    target_column='RUL'
)
```

---

更多详细信息，请参考[用户指南](USER_GUIDE.md)和[快速入门](QUICKSTART.md)。
