"""
数据预处理模块 (Data Preprocessing Module)
用于RUL预测的数据预处理，包括清洗、插补、标准化、滑窗、标签构建等
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    数据预处理器
    
    功能：
    1. 数据清洗（去除异常值、重复值等）
    2. 缺失值插补（多种方法）
    3. 数据标准化/归一化
    4. 滑动窗口构建
    5. RUL标签构建
    6. 健康因子构建
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化预处理器
        
        参数:
            data: 要预处理的数据框
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.scaler = None
        self.imputer = None
        self.processing_log = []
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        删除重复行
        
        参数:
            subset: 用于识别重复的列
            keep: 保留哪个重复值 ('first', 'last', False)
            
        返回:
            pd.DataFrame: 去重后的数据
        """
        original_len = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        removed = original_len - len(self.data)
        
        log_msg = f"删除重复行: {removed} 行"
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return self.data
    
    def remove_outliers(self,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        删除异常值
        
        参数:
            columns: 要处理的列，None表示所有数值列
            method: 异常值检测方法 ('iqr', 'zscore', 'percentile')
            threshold: 阈值
            
        返回:
            pd.DataFrame: 处理后的数据
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        original_len = len(self.data)
        mask = pd.Series([True] * len(self.data))
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            col_data = self.data[col]
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask &= (col_data >= lower) & (col_data <= upper)
                
            elif method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                z_scores = np.abs((col_data - mean) / std)
                mask &= z_scores <= threshold
                
            elif method == 'percentile':
                lower = col_data.quantile(0.01)
                upper = col_data.quantile(0.99)
                mask &= (col_data >= lower) & (col_data <= upper)
        
        self.data = self.data[mask]
        removed = original_len - len(self.data)
        
        log_msg = f"删除异常值 ({method}方法): {removed} 行"
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return self.data
    
    def handle_missing_values(self,
                             strategy: str = 'mean',
                             columns: Optional[List[str]] = None,
                             fill_value: Optional[Union[float, str]] = None,
                             method: Optional[str] = None) -> pd.DataFrame:
        """
        处理缺失值
        
        参数:
            strategy: 插补策略 ('mean', 'median', 'mode', 'constant', 'ffill', 'bfill', 'interpolate', 'knn')
            columns: 要处理的列
            fill_value: 当strategy='constant'时使用的填充值
            method: 插值方法（当strategy='interpolate'时）
            
        返回:
            pd.DataFrame: 处理后的数据
        """
        if columns is None:
            columns = self.data.columns.tolist()
        
        missing_before = self.data[columns].isnull().sum().sum()
        
        if strategy in ['mean', 'median', 'mode', 'constant']:
            # 使用SimpleImputer
            numeric_cols = [col for col in columns if col in self.data.select_dtypes(include=[np.number]).columns]
            
            if strategy == 'mode':
                strategy = 'most_frequent'
            
            self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            self.data[numeric_cols] = self.imputer.fit_transform(self.data[numeric_cols])
            
        elif strategy in ['ffill', 'bfill']:
            # 前向/后向填充
            self.data[columns] = self.data[columns].fillna(method=strategy)
            
        elif strategy == 'interpolate':
            # 插值
            method = method if method else 'linear'
            numeric_cols = [col for col in columns if col in self.data.select_dtypes(include=[np.number]).columns]
            self.data[numeric_cols] = self.data[numeric_cols].interpolate(method=method)
            
        elif strategy == 'knn':
            # KNN插补
            numeric_cols = [col for col in columns if col in self.data.select_dtypes(include=[np.number]).columns]
            self.imputer = KNNImputer(n_neighbors=5)
            self.data[numeric_cols] = self.imputer.fit_transform(self.data[numeric_cols])
        
        missing_after = self.data[columns].isnull().sum().sum()
        
        log_msg = f"缺失值处理 ({strategy}): 处理前 {missing_before} 个缺失值, 处理后 {missing_after} 个"
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return self.data
    
    def normalize(self,
                 columns: Optional[List[str]] = None,
                 method: str = 'standard',
                 feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """
        数据标准化/归一化
        
        参数:
            columns: 要标准化的列
            method: 标准化方法 ('standard', 'minmax', 'robust')
            feature_range: MinMax缩放的范围
            
        返回:
            pd.DataFrame: 标准化后的数据
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            # Z-score标准化
            self.scaler = StandardScaler()
        elif method == 'minmax':
            # Min-Max归一化
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            # 鲁棒标准化（对异常值不敏感）
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        self.data[columns] = self.scaler.fit_transform(self.data[columns])
        
        log_msg = f"数据标准化 ({method}): {len(columns)} 个特征"
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return self.data
    
    def create_sliding_windows(self,
                              window_size: int,
                              step_size: int = 1,
                              feature_columns: Optional[List[str]] = None,
                              target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        创建滑动窗口数据
        
        参数:
            window_size: 窗口大小
            step_size: 步长
            feature_columns: 特征列
            target_column: 目标列
            
        返回:
            Tuple[np.ndarray, np.ndarray]: (X, y) 窗口特征和目标
        """
        if feature_columns is None:
            feature_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in feature_columns:
                feature_columns.remove(target_column)
        
        data_array = self.data[feature_columns].values
        
        X = []
        y = [] if target_column else None
        
        for i in range(0, len(data_array) - window_size + 1, step_size):
            X.append(data_array[i:i + window_size])
            
            if target_column:
                y.append(self.data[target_column].iloc[i + window_size - 1])
        
        X = np.array(X)
        y = np.array(y) if target_column else None
        
        log_msg = f"创建滑动窗口: 窗口大小={window_size}, 步长={step_size}, 生成 {len(X)} 个样本"
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return X, y
    
    def construct_rul_labels(self,
                            unit_column: str,
                            time_column: Optional[str] = None,
                            max_rul: Optional[int] = None) -> pd.DataFrame:
        """
        构建RUL（剩余使用寿命）标签
        
        参数:
            unit_column: 单元/设备标识列
            time_column: 时间列（如果有）
            max_rul: 最大RUL值（用于截断）
            
        返回:
            pd.DataFrame: 添加了RUL列的数据
        """
        if unit_column not in self.data.columns:
            raise ValueError(f"单元列 '{unit_column}' 不存在")
        
        rul_list = []
        
        # 按单元分组计算RUL
        for unit_id in self.data[unit_column].unique():
            unit_data = self.data[self.data[unit_column] == unit_id]
            max_cycle = len(unit_data)
            
            # RUL = 最大周期 - 当前周期
            if time_column and time_column in unit_data.columns:
                current_cycles = unit_data[time_column].values
                rul = max_cycle - current_cycles
            else:
                rul = np.arange(max_cycle, 0, -1)
            
            # 应用最大RUL限制
            if max_rul is not None:
                rul = np.minimum(rul, max_rul)
            
            rul_list.extend(rul)
        
        self.data['RUL'] = rul_list
        
        log_msg = f"构建RUL标签: {len(self.data[unit_column].unique())} 个单元, 最大RUL={max_rul if max_rul else '无限制'}"
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return self.data
    
    def construct_health_indicator(self,
                                  feature_columns: List[str],
                                  method: str = 'pca_first',
                                  weights: Optional[List[float]] = None) -> pd.DataFrame:
        """
        构建健康因子/健康指标
        
        参数:
            feature_columns: 用于构建健康指标的特征列
            method: 构建方法 ('pca_first', 'weighted_sum', 'monotonicity')
            weights: 加权和方法的权重
            
        返回:
            pd.DataFrame: 添加了健康指标列的数据
        """
        if not all(col in self.data.columns for col in feature_columns):
            raise ValueError("部分特征列不存在")
        
        feature_data = self.data[feature_columns].fillna(0)
        
        if method == 'pca_first':
            # 使用PCA第一主成分作为健康指标
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            hi = pca.fit_transform(feature_data)
            self.data['health_indicator'] = hi.flatten()
            
            log_msg = f"构建健康指标 (PCA): 解释方差比={pca.explained_variance_ratio_[0]:.4f}"
            
        elif method == 'weighted_sum':
            # 加权求和
            if weights is None:
                weights = [1.0] * len(feature_columns)
            
            if len(weights) != len(feature_columns):
                raise ValueError("权重数量与特征数量不匹配")
            
            hi = np.zeros(len(feature_data))
            for i, col in enumerate(feature_columns):
                hi += feature_data[col].values * weights[i]
            
            self.data['health_indicator'] = hi
            log_msg = f"构建健康指标 (加权和): 使用 {len(feature_columns)} 个特征"
            
        elif method == 'monotonicity':
            # 基于单调性的健康指标
            # 计算每个特征的趋势
            hi = np.zeros(len(feature_data))
            
            for col in feature_columns:
                # 标准化特征
                col_data = feature_data[col].values
                col_normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min() + 1e-10)
                
                # 计算趋势（正向或负向）
                correlation = np.corrcoef(np.arange(len(col_normalized)), col_normalized)[0, 1]
                
                if correlation > 0:
                    hi += col_normalized
                else:
                    hi += (1 - col_normalized)
            
            # 归一化健康指标
            hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
            self.data['health_indicator'] = hi
            
            log_msg = f"构建健康指标 (单调性): 使用 {len(feature_columns)} 个特征"
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return self.data
    
    def split_by_unit(self,
                     unit_column: str,
                     train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按单元划分训练集和测试集
        
        参数:
            unit_column: 单元标识列
            train_ratio: 训练集比例
            
        返回:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 测试集)
        """
        if unit_column not in self.data.columns:
            raise ValueError(f"单元列 '{unit_column}' 不存在")
        
        units = self.data[unit_column].unique()
        n_train = int(len(units) * train_ratio)
        
        # 随机划分单元
        np.random.seed(42)
        train_units = np.random.choice(units, size=n_train, replace=False)
        
        train_data = self.data[self.data[unit_column].isin(train_units)]
        test_data = self.data[~self.data[unit_column].isin(train_units)]
        
        log_msg = f"数据划分: 训练集 {len(train_units)} 个单元 ({len(train_data)} 行), 测试集 {len(units)-len(train_units)} 个单元 ({len(test_data)} 行)"
        self.processing_log.append(log_msg)
        print(log_msg)
        
        return train_data, test_data
    
    def export_processed_data(self, file_path: str):
        """
        导出处理后的数据
        
        参数:
            file_path: 保存路径
        """
        self.data.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"处理后的数据已保存至: {file_path}")
    
    def get_processing_log(self) -> List[str]:
        """
        获取处理日志
        
        返回:
            List[str]: 处理步骤列表
        """
        return self.processing_log
    
    def print_processing_summary(self):
        """打印处理摘要"""
        print("\n" + "="*60)
        print("数据预处理摘要")
        print("="*60)
        print(f"原始数据形状: {self.original_data.shape}")
        print(f"处理后数据形状: {self.data.shape}")
        print(f"\n处理步骤:")
        for i, log in enumerate(self.processing_log, 1):
            print(f"{i}. {log}")
        print("="*60 + "\n")
