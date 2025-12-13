"""
数据加载模块 (Data Loader Module)
用于多元时序CSV文件的读取、筛选、合并及可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union, Dict
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataLoader:
    """
    多元时序数据加载器
    
    功能：
    1. 读取单个或多个CSV文件
    2. 数据筛选和过滤
    3. 多文件数据合并
    4. 原始数据可视化
    """
    
    def __init__(self):
        """初始化数据加载器"""
        self.data = None
        self.file_path = None
        self.data_info = {}
        
    def read_csv(self, 
                 file_path: Union[str, Path],
                 encoding: str = 'utf-8',
                 **kwargs) -> pd.DataFrame:
        """
        读取单个CSV文件
        
        参数:
            file_path: CSV文件路径
            encoding: 文件编码，默认utf-8
            **kwargs: pandas.read_csv的其他参数
            
        返回:
            pd.DataFrame: 读取的数据框
        """
        try:
            self.file_path = Path(file_path)
            self.data = pd.read_csv(file_path, encoding=encoding, **kwargs)
            
            # 记录数据信息
            self.data_info['file_path'] = str(file_path)
            self.data_info['shape'] = self.data.shape
            self.data_info['columns'] = list(self.data.columns)
            self.data_info['dtypes'] = self.data.dtypes.to_dict()
            
            print(f"成功读取文件: {file_path}")
            print(f"数据形状: {self.data.shape}")
            print(f"列名: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            print(f"读取文件失败: {e}")
            raise
    
    def read_multiple_csv(self,
                         file_paths: List[Union[str, Path]],
                         encoding: str = 'utf-8',
                         **kwargs) -> Dict[str, pd.DataFrame]:
        """
        读取多个CSV文件
        
        参数:
            file_paths: CSV文件路径列表
            encoding: 文件编码
            **kwargs: pandas.read_csv的其他参数
            
        返回:
            Dict[str, pd.DataFrame]: 文件名到数据框的字典
        """
        data_dict = {}
        
        for file_path in file_paths:
            try:
                file_name = Path(file_path).stem
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                data_dict[file_name] = df
                print(f"成功读取: {file_name}, 形状: {df.shape}")
            except Exception as e:
                print(f"读取文件 {file_path} 失败: {e}")
                
        return data_dict
    
    def filter_columns(self,
                      columns: Optional[List[str]] = None,
                      exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        筛选数据列
        
        参数:
            columns: 要保留的列名列表
            exclude_columns: 要排除的列名列表
            
        返回:
            pd.DataFrame: 筛选后的数据框
        """
        if self.data is None:
            raise ValueError("请先读取数据")
        
        if columns is not None:
            # 保留指定列
            valid_columns = [col for col in columns if col in self.data.columns]
            self.data = self.data[valid_columns]
            print(f"保留列: {valid_columns}")
            
        elif exclude_columns is not None:
            # 排除指定列
            self.data = self.data.drop(columns=exclude_columns, errors='ignore')
            print(f"排除列: {exclude_columns}")
            
        return self.data
    
    def filter_rows(self,
                   condition: Optional[str] = None,
                   start_index: Optional[int] = None,
                   end_index: Optional[int] = None) -> pd.DataFrame:
        """
        筛选数据行
        
        参数:
            condition: 筛选条件（字符串形式的查询表达式）
            start_index: 起始索引
            end_index: 结束索引
            
        返回:
            pd.DataFrame: 筛选后的数据框
        """
        if self.data is None:
            raise ValueError("请先读取数据")
        
        original_shape = self.data.shape
        
        if condition is not None:
            # 使用query方法筛选
            self.data = self.data.query(condition)
            print(f"应用条件筛选: {condition}")
            
        if start_index is not None or end_index is not None:
            # 索引切片
            start = start_index if start_index is not None else 0
            end = end_index if end_index is not None else len(self.data)
            self.data = self.data.iloc[start:end]
            print(f"索引切片: [{start}:{end}]")
        
        print(f"筛选前: {original_shape}, 筛选后: {self.data.shape}")
        return self.data
    
    def merge_data(self,
                  data_list: List[pd.DataFrame],
                  method: str = 'concat',
                  on: Optional[Union[str, List[str]]] = None,
                  how: str = 'outer',
                  axis: int = 0) -> pd.DataFrame:
        """
        合并多个数据框
        
        参数:
            data_list: 要合并的数据框列表
            method: 合并方法 ('concat' 或 'merge')
            on: merge方法的连接键
            how: merge方法的连接方式
            axis: concat方法的轴向 (0=行, 1=列)
            
        返回:
            pd.DataFrame: 合并后的数据框
        """
        if not data_list:
            raise ValueError("数据列表不能为空")
        
        if method == 'concat':
            # 使用concat连接
            self.data = pd.concat(data_list, axis=axis, ignore_index=True)
            print(f"使用concat方法合并 {len(data_list)} 个数据框")
            
        elif method == 'merge':
            # 使用merge合并
            if on is None:
                raise ValueError("merge方法需要指定连接键'on'")
            
            result = data_list[0]
            for df in data_list[1:]:
                result = pd.merge(result, df, on=on, how=how)
            
            self.data = result
            print(f"使用merge方法合并 {len(data_list)} 个数据框")
        else:
            raise ValueError(f"不支持的合并方法: {method}")
        
        print(f"合并后数据形状: {self.data.shape}")
        return self.data
    
    def visualize_raw_data(self,
                          columns: Optional[List[str]] = None,
                          sample_size: Optional[int] = None,
                          figsize: tuple = (15, 10),
                          save_path: Optional[str] = None):
        """
        可视化原始数据
        
        参数:
            columns: 要可视化的列名列表，None表示所有数值列
            sample_size: 采样大小，用于大数据集
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.data is None:
            raise ValueError("请先读取数据")
        
        # 选择要可视化的数据
        plot_data = self.data.copy()
        
        if sample_size is not None and len(plot_data) > sample_size:
            plot_data = plot_data.sample(n=sample_size, random_state=42).sort_index()
        
        # 选择数值列
        if columns is None:
            numeric_cols = plot_data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in plot_data.columns]
        
        if not numeric_cols:
            print("没有可视化的数值列")
            return
        
        # 创建子图
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        # 绘制每个特征的时序图
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            ax.plot(plot_data.index, plot_data[col], linewidth=0.8)
            ax.set_title(f'{col}', fontsize=10)
            ax.set_xlabel('时间步 (Time Step)', fontsize=8)
            ax.set_ylabel('值 (Value)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # 隐藏多余的子图
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")
        
        plt.show()
        
    def get_data_summary(self) -> Dict:
        """
        获取数据摘要信息
        
        返回:
            Dict: 包含数据基本信息的字典
        """
        if self.data is None:
            raise ValueError("请先读取数据")
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        return summary
    
    def display_info(self):
        """显示数据信息"""
        if self.data is None:
            print("未加载数据")
            return
        
        print("\n" + "="*50)
        print("数据信息概览")
        print("="*50)
        print(f"数据形状: {self.data.shape}")
        print(f"列数: {self.data.shape[1]}")
        print(f"行数: {self.data.shape[0]}")
        print(f"\n列名和类型:")
        print(self.data.dtypes)
        print(f"\n缺失值统计:")
        print(self.data.isnull().sum())
        print(f"\n数据预览:")
        print(self.data.head())
        print("="*50 + "\n")
