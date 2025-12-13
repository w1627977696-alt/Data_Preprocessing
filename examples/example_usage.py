"""
数据预处理工具包使用示例
演示所有主要功能的使用方法
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from data_preprocessing import DataLoader, StatisticalAnalyzer, CorrelationAnalyzer, DataPreprocessor

# 设置随机种子
np.random.seed(42)


def create_sample_data():
    """创建示例数据集（模拟RUL预测场景）"""
    print("\n" + "="*60)
    print("1. 创建示例数据")
    print("="*60)
    
    n_units = 5  # 设备数量
    cycles_per_unit = 200  # 每个设备的周期数
    
    data_list = []
    
    for unit_id in range(1, n_units + 1):
        cycles = cycles_per_unit + np.random.randint(-20, 20)
        
        # 生成时序数据（模拟传感器数据）
        time = np.arange(cycles)
        
        # 模拟退化过程
        degradation = time / cycles
        
        # 传感器1: 温度（随退化上升）
        sensor_1 = 60 + 20 * degradation + np.random.normal(0, 2, cycles)
        
        # 传感器2: 振动（随退化上升）
        sensor_2 = 0.3 + 0.5 * degradation + np.random.normal(0, 0.05, cycles)
        
        # 传感器3: 压力（随退化下降）
        sensor_3 = 100 - 20 * degradation + np.random.normal(0, 3, cycles)
        
        # 传感器4: 噪声特征（无明显趋势）
        sensor_4 = np.random.normal(50, 10, cycles)
        
        # 传感器5: 周期性特征
        sensor_5 = 30 + 10 * np.sin(time / 10) + np.random.normal(0, 1, cycles)
        
        unit_data = pd.DataFrame({
            'unit': unit_id,
            'cycle': time + 1,
            'sensor_1': sensor_1,
            'sensor_2': sensor_2,
            'sensor_3': sensor_3,
            'sensor_4': sensor_4,
            'sensor_5': sensor_5
        })
        
        data_list.append(unit_data)
    
    data = pd.concat(data_list, ignore_index=True)
    
    # 添加一些缺失值
    missing_indices = np.random.choice(len(data), size=int(len(data) * 0.02), replace=False)
    data.loc[missing_indices, 'sensor_4'] = np.nan
    
    print(f"创建示例数据: {data.shape}")
    print(f"设备数量: {n_units}")
    print(f"总周期数: {len(data)}")
    
    # 保存数据
    os.makedirs('../data/raw', exist_ok=True)
    data.to_csv('../data/raw/sample_data.csv', index=False)
    print("数据已保存至: ../data/raw/sample_data.csv")
    
    return data


def example_data_loading():
    """示例1: 数据加载和基本操作"""
    print("\n" + "="*60)
    print("2. 数据加载示例")
    print("="*60)
    
    # 初始化数据加载器
    loader = DataLoader()
    
    # 读取CSV文件
    data = loader.read_csv('../data/raw/sample_data.csv')
    
    # 显示数据信息
    loader.display_info()
    
    # 筛选特定列
    sensor_columns = ['unit', 'cycle', 'sensor_1', 'sensor_2', 'sensor_3']
    filtered_data = loader.filter_columns(columns=sensor_columns)
    
    # 可视化原始数据
    print("\n可视化原始数据...")
    loader.visualize_raw_data(
        columns=['sensor_1', 'sensor_2', 'sensor_3'],
        sample_size=500,
        figsize=(15, 8),
        save_path='../data/processed/raw_data_visualization.png'
    )
    
    return loader.data


def example_statistical_analysis(data):
    """示例2: 统计分析"""
    print("\n" + "="*60)
    print("3. 统计分析示例")
    print("="*60)
    
    # 初始化统计分析器
    analyzer = StatisticalAnalyzer(data)
    
    # 基础统计
    basic_stats = analyzer.basic_statistics()
    
    # 分布分析
    dist_analysis = analyzer.distribution_analysis()
    
    # 缺失值分析
    missing_analysis = analyzer.missing_value_analysis()
    
    # 异常值检测
    outliers = analyzer.outlier_detection(method='iqr')
    
    # 时序分析
    ts_analysis = analyzer.time_series_analysis(window_size=10)
    
    # 可视化
    print("\n生成统计可视化...")
    analyzer.visualize_distribution(
        columns=['sensor_1', 'sensor_2', 'sensor_3'],
        save_path='../data/processed/distribution_plot.png'
    )
    
    analyzer.visualize_boxplot(
        columns=['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5'],
        save_path='../data/processed/boxplot.png'
    )
    
    # 生成完整报告
    report = analyzer.generate_report(save_path='../data/processed/statistical_report.xlsx')
    
    return analyzer


def example_correlation_analysis(data):
    """示例3: 相关性分析"""
    print("\n" + "="*60)
    print("4. 相关性分析示例")
    print("="*60)
    
    # 初始化相关性分析器
    corr_analyzer = CorrelationAnalyzer(data)
    
    # Pearson相关性
    print("\n计算Pearson相关性...")
    pearson_corr = corr_analyzer.pearson_correlation()
    
    # Spearman相关性
    print("\n计算Spearman相关性...")
    spearman_corr = corr_analyzer.spearman_correlation()
    
    # 与特定目标的相关性分析
    # 假设sensor_1是我们关心的目标变量
    print("\n分析与sensor_1的相关性...")
    target_corr = corr_analyzer.correlation_with_target(
        target='sensor_1',
        methods=['pearson', 'spearman']
    )
    
    # 特征选择
    selected_features = corr_analyzer.feature_selection(
        target='sensor_1',
        method='pearson',
        top_k=3
    )
    
    # 可视化
    print("\n生成相关性可视化...")
    corr_analyzer.visualize_correlation_matrix(
        method='pearson',
        save_path='../data/processed/correlation_matrix.png'
    )
    
    corr_analyzer.visualize_correlation_with_target(
        target='sensor_1',
        method='pearson',
        top_k=4,
        save_path='../data/processed/correlation_with_target.png'
    )
    
    # 生成相关性表格
    corr_table = corr_analyzer.generate_correlation_table(
        target='sensor_1',
        methods=['pearson', 'spearman'],
        save_path='../data/processed/correlation_table.csv'
    )
    
    return corr_analyzer


def example_preprocessing(data):
    """示例4: 数据预处理"""
    print("\n" + "="*60)
    print("5. 数据预处理示例")
    print("="*60)
    
    # 初始化预处理器
    preprocessor = DataPreprocessor(data)
    
    # 1. 删除重复值
    print("\n1. 删除重复值...")
    preprocessor.remove_duplicates()
    
    # 2. 处理缺失值
    print("\n2. 处理缺失值...")
    preprocessor.handle_missing_values(
        strategy='mean',
        columns=['sensor_4']
    )
    
    # 3. 删除异常值（可选，根据实际需求）
    # print("\n3. 删除异常值...")
    # preprocessor.remove_outliers(method='iqr')
    
    # 4. 构建RUL标签
    print("\n4. 构建RUL标签...")
    preprocessor.construct_rul_labels(
        unit_column='unit',
        time_column='cycle',
        max_rul=125
    )
    
    # 5. 构建健康指标
    print("\n5. 构建健康指标...")
    preprocessor.construct_health_indicator(
        feature_columns=['sensor_1', 'sensor_2', 'sensor_3'],
        method='pca_first'
    )
    
    # 6. 数据标准化
    print("\n6. 数据标准化...")
    sensor_columns = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
    preprocessor.normalize(
        columns=sensor_columns,
        method='standard'
    )
    
    # 7. 按单元划分数据集
    print("\n7. 划分训练集和测试集...")
    train_data, test_data = preprocessor.split_by_unit(
        unit_column='unit',
        train_ratio=0.8
    )
    
    # 8. 创建滑动窗口
    print("\n8. 创建滑动窗口...")
    window_size = 30
    step_size = 1
    
    # 为训练集创建滑动窗口
    train_preprocessor = DataPreprocessor(train_data)
    X_train, y_train = train_preprocessor.create_sliding_windows(
        window_size=window_size,
        step_size=step_size,
        feature_columns=sensor_columns,
        target_column='RUL'
    )
    
    print(f"训练集窗口形状: X={X_train.shape}, y={y_train.shape}")
    
    # 为测试集创建滑动窗口
    test_preprocessor = DataPreprocessor(test_data)
    X_test, y_test = test_preprocessor.create_sliding_windows(
        window_size=window_size,
        step_size=step_size,
        feature_columns=sensor_columns,
        target_column='RUL'
    )
    
    print(f"测试集窗口形状: X={X_test.shape}, y={y_test.shape}")
    
    # 导出处理后的数据
    print("\n9. 导出处理后的数据...")
    preprocessor.export_processed_data('../data/processed/preprocessed_data.csv')
    train_data.to_csv('../data/processed/train_data.csv', index=False)
    test_data.to_csv('../data/processed/test_data.csv', index=False)
    
    # 保存窗口数据
    np.save('../data/processed/X_train.npy', X_train)
    np.save('../data/processed/y_train.npy', y_train)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_test.npy', y_test)
    print("窗口数据已保存为.npy格式")
    
    # 打印处理摘要
    preprocessor.print_processing_summary()
    
    return preprocessor, X_train, y_train, X_test, y_test


def main():
    """主函数：运行所有示例"""
    print("="*60)
    print("数据预处理工具包 - 完整示例")
    print("用于RUL预测的多元时序数据分析和预处理")
    print("="*60)
    
    # 创建输出目录
    os.makedirs('../data/processed', exist_ok=True)
    
    # 1. 创建或加载示例数据
    data = create_sample_data()
    
    # 2. 数据加载示例
    data = example_data_loading()
    
    # 3. 统计分析示例
    analyzer = example_statistical_analysis(data)
    
    # 4. 相关性分析示例
    corr_analyzer = example_correlation_analysis(data)
    
    # 5. 数据预处理示例
    preprocessor, X_train, y_train, X_test, y_test = example_preprocessing(data)
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    print("\n生成的文件:")
    print("- 数据文件: ../data/raw/sample_data.csv")
    print("- 处理后数据: ../data/processed/preprocessed_data.csv")
    print("- 训练集: ../data/processed/train_data.csv")
    print("- 测试集: ../data/processed/test_data.csv")
    print("- 窗口数据: ../data/processed/X_train.npy, y_train.npy, etc.")
    print("- 统计报告: ../data/processed/statistical_report.xlsx")
    print("- 相关性表格: ../data/processed/correlation_table.csv")
    print("- 可视化图表: ../data/processed/*.png")
    print("\n数据预处理完成，可以用于后续的RUL预测模型训练！")


if __name__ == "__main__":
    main()
