# 快速入门指南

## 5分钟快速上手

### 第一步：安装

```bash
# 克隆仓库
git clone https://github.com/w1627977696-alt/Data_Preprocessing.git
cd Data_Preprocessing

# 安装依赖
pip install -r requirements.txt
```

### 第二步：运行示例

```bash
cd examples
python example_usage.py
```

这将自动：
1. 生成示例数据（模拟5个设备的传感器数据）
2. 进行完整的数据分析和预处理
3. 生成各种可视化图表
4. 创建用于模型训练的滑动窗口数据

### 第三步：查看结果

运行完成后，检查以下文件：
- `data/raw/sample_data.csv` - 原始示例数据
- `data/processed/*.png` - 各种分析图表
- `data/processed/statistical_report.xlsx` - 统计分析报告
- `data/processed/correlation_table.csv` - 相关性分析表格
- `data/processed/*.csv` - 处理后的数据
- `data/processed/*.npy` - 滑动窗口数据（可直接用于模型训练）

## 用自己的数据

### 最简单的使用方式

```python
import sys
sys.path.insert(0, 'src')
from data_preprocessing import DataLoader, DataPreprocessor

# 1. 加载你的数据
loader = DataLoader()
data = loader.read_csv('你的数据.csv')

# 2. 预处理
preprocessor = DataPreprocessor(data)
preprocessor.handle_missing_values(strategy='mean')  # 处理缺失值
preprocessor.normalize(method='standard')            # 标准化

# 3. 构建RUL标签（如果需要）
preprocessor.construct_rul_labels(
    unit_column='设备ID列名',    # 设备标识列
    time_column='时间列名',      # 时间或周期列（可选）
    max_rul=125                  # 最大RUL值
)

# 4. 创建滑动窗口用于模型训练
X, y = preprocessor.create_sliding_windows(
    window_size=30,                          # 窗口大小
    step_size=1,                             # 步长
    feature_columns=['传感器1', '传感器2'],   # 特征列
    target_column='RUL'                      # 目标列
)

print(f"训练数据准备完成: X形状={X.shape}, y形状={y.shape}")
# 现在可以用X和y训练LSTM/GRU等模型了
```

### 完整的预处理流程

```python
from data_preprocessing import (
    DataLoader, 
    StatisticalAnalyzer, 
    CorrelationAnalyzer, 
    DataPreprocessor
)

# 1. 加载和探索数据
loader = DataLoader()
data = loader.read_csv('你的数据.csv')
loader.display_info()

# 2. 统计分析
analyzer = StatisticalAnalyzer(data)
analyzer.basic_statistics()                    # 基础统计
analyzer.visualize_distribution()              # 分布可视化
analyzer.generate_report('分析报告.xlsx')      # 生成完整报告

# 3. 相关性分析（用于特征选择）
corr_analyzer = CorrelationAnalyzer(data)
selected_features = corr_analyzer.feature_selection(
    target='RUL',      # 你的目标变量
    method='pearson',  # 相关性方法
    top_k=10          # 选择前10个最相关的特征
)

# 4. 数据预处理
preprocessor = DataPreprocessor(data)

# 处理缺失值
preprocessor.handle_missing_values(
    strategy='mean',                    # 或 'median', 'knn', 'interpolate'
    columns=selected_features
)

# 标准化
preprocessor.normalize(
    columns=selected_features,
    method='standard'                   # 或 'minmax', 'robust'
)

# 构建RUL标签
preprocessor.construct_rul_labels(
    unit_column='unit',
    max_rul=125
)

# 5. 划分训练集和测试集
train_data, test_data = preprocessor.split_by_unit(
    unit_column='unit',
    train_ratio=0.8
)

# 6. 创建滑动窗口
train_prep = DataPreprocessor(train_data)
X_train, y_train = train_prep.create_sliding_windows(
    window_size=30,
    feature_columns=selected_features,
    target_column='RUL'
)

test_prep = DataPreprocessor(test_data)
X_test, y_test = test_prep.create_sliding_windows(
    window_size=30,
    feature_columns=selected_features,
    target_column='RUL'
)

# 保存处理后的数据
import numpy as np
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("数据预处理完成！可以开始训练模型了。")
```

## 常见使用场景

### 场景1：只需要数据清洗和标准化

```python
preprocessor = DataPreprocessor(data)
preprocessor.remove_duplicates()                           # 去重
preprocessor.handle_missing_values(strategy='mean')        # 处理缺失值
preprocessor.normalize(method='standard')                  # 标准化
preprocessor.export_processed_data('clean_data.csv')       # 导出
```

### 场景2：需要构建健康指标

```python
preprocessor = DataPreprocessor(data)

# 使用PCA方法构建健康指标
preprocessor.construct_health_indicator(
    feature_columns=['温度', '振动', '压力'],
    method='pca_first'
)

# 健康指标保存在 data['health_indicator'] 列中
```

### 场景3：特征选择

```python
corr_analyzer = CorrelationAnalyzer(data)

# 方法1：按相关性阈值选择
selected = corr_analyzer.feature_selection(
    target='RUL',
    method='pearson',
    threshold=0.3  # 选择相关性>0.3的特征
)

# 方法2：选择Top K特征
selected = corr_analyzer.feature_selection(
    target='RUL',
    method='pearson',
    top_k=10  # 选择前10个特征
)
```

### 场景4：合并多个数据文件

```python
loader = DataLoader()

# 读取多个文件
file_list = ['device1.csv', 'device2.csv', 'device3.csv']
data_dict = loader.read_multiple_csv(file_list)

# 合并所有数据
all_data = []
for name, df in data_dict.items():
    all_data.append(df)

merged = loader.merge_data(all_data, method='concat', axis=0)
```

## 数据格式要求

### 输入数据格式

你的CSV文件应该包含以下列：

```
unit, cycle, sensor_1, sensor_2, sensor_3, ...
1,    1,     60.5,     0.3,      100.2,    ...
1,    2,     61.2,     0.35,     99.8,     ...
1,    3,     62.1,     0.38,     98.5,     ...
2,    1,     59.8,     0.28,     101.3,    ...
...
```

必需列：
- **unit/设备ID列**: 标识不同的设备或运行周期
- **cycle/时间列**: 时间步或周期（可选，如果没有会自动生成）
- **传感器列**: 各种传感器测量值

### 输出数据格式

滑动窗口输出：
- **X**: numpy数组，形状为 (样本数, 窗口大小, 特征数)
  - 例如：(1000, 30, 10) 表示1000个样本，每个样本是30个时间步的10个特征
- **y**: numpy数组，形状为 (样本数,)
  - 例如：(1000,) 表示1000个RUL标签

这个格式可以直接用于：
- LSTM/GRU模型：`model.fit(X, y)`
- CNN-LSTM模型
- Transformer模型

## 下一步

- 阅读完整的[用户指南](docs/USER_GUIDE.md)
- 查看详细的[README](README.md)
- 根据你的需求调整参数
- 开始训练RUL预测模型！

## 遇到问题？

1. 检查数据格式是否正确
2. 确保所有依赖都已安装
3. 查看[常见问题](docs/USER_GUIDE.md#常见问题)
4. 提交Issue寻求帮助

祝你使用愉快！🚀
