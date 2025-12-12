"""
统计分析模块 (Statistical Analysis Module)
用于数据的综合统计特性分析和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class StatisticalAnalyzer:
    """
    统计分析器
    
    功能：
    1. 基础统计量计算（均值、方差、极值等）
    2. 分布特征分析（偏度、峰度、正态性检验）
    3. 时序特性分析（趋势、周期性）
    4. 统计可视化（分布图、箱线图、热图等）
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化统计分析器
        
        参数:
            data: 要分析的数据框
        """
        self.data = data.copy()
        self.numeric_data = data.select_dtypes(include=[np.number])
        self.stats_results = {}
        
    def basic_statistics(self, percentiles: List[float] = [0.25, 0.5, 0.75]) -> pd.DataFrame:
        """
        计算基础统计量
        
        参数:
            percentiles: 要计算的百分位数列表
            
        返回:
            pd.DataFrame: 统计量表格
        """
        stats_df = self.numeric_data.describe(percentiles=percentiles).T
        
        # 添加额外的统计量
        stats_df['variance'] = self.numeric_data.var()
        stats_df['skewness'] = self.numeric_data.skew()
        stats_df['kurtosis'] = self.numeric_data.kurtosis()
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['cv'] = stats_df['std'] / stats_df['mean']  # 变异系数
        
        self.stats_results['basic'] = stats_df
        
        print("\n基础统计量:")
        print(stats_df.round(4))
        
        return stats_df
    
    def distribution_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        分布特征分析
        
        返回:
            Dict: 包含各列分布特征的字典
        """
        results = {}
        
        for col in self.numeric_data.columns:
            col_data = self.numeric_data[col].dropna()
            
            if len(col_data) < 3:
                continue
            
            # 正态性检验
            try:
                stat, p_value = stats.shapiro(col_data) if len(col_data) <= 5000 else stats.normaltest(col_data)
                normality = "正态" if p_value > 0.05 else "非正态"
            except:
                stat, p_value = np.nan, np.nan
                normality = "未知"
            
            results[col] = {
                '均值': col_data.mean(),
                '中位数': col_data.median(),
                '标准差': col_data.std(),
                '偏度': col_data.skew(),
                '峰度': col_data.kurtosis(),
                '最小值': col_data.min(),
                '最大值': col_data.max(),
                '正态性检验p值': p_value,
                '分布类型': normality
            }
        
        results_df = pd.DataFrame(results).T
        self.stats_results['distribution'] = results_df
        
        print("\n分布特征分析:")
        print(results_df.round(4))
        
        return results
    
    def missing_value_analysis(self) -> pd.DataFrame:
        """
        缺失值分析
        
        返回:
            pd.DataFrame: 缺失值统计表
        """
        missing_stats = pd.DataFrame({
            '缺失数量': self.data.isnull().sum(),
            '缺失比例': self.data.isnull().sum() / len(self.data) * 100
        })
        
        missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
        
        self.stats_results['missing'] = missing_stats
        
        if len(missing_stats) > 0:
            print("\n缺失值分析:")
            print(missing_stats.round(2))
        else:
            print("\n数据无缺失值")
        
        return missing_stats
    
    def outlier_detection(self, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, pd.DataFrame]:
        """
        异常值检测
        
        参数:
            method: 检测方法 ('iqr' 或 'zscore')
            threshold: 阈值（zscore方法使用，默认3倍标准差）
            
        返回:
            Dict: 包含异常值信息的字典
        """
        outliers_info = {}
        
        for col in self.numeric_data.columns:
            col_data = self.numeric_data[col].dropna()
            
            if method == 'iqr':
                # IQR方法
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
            elif method == 'zscore':
                # Z-score方法
                z_scores = np.abs(stats.zscore(col_data))
                outliers = col_data[z_scores > threshold]
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            if len(outliers) > 0:
                outliers_info[col] = {
                    '异常值数量': len(outliers),
                    '异常值比例': len(outliers) / len(col_data) * 100,
                    '异常值示例': outliers.head().tolist()
                }
        
        if outliers_info:
            outliers_df = pd.DataFrame(outliers_info).T
            self.stats_results['outliers'] = outliers_df
            
            print(f"\n异常值检测 ({method}方法):")
            print(outliers_df[['异常值数量', '异常值比例']].round(2))
        else:
            print("\n未检测到异常值")
        
        return outliers_info
    
    def time_series_analysis(self, window_size: int = 10) -> Dict[str, pd.DataFrame]:
        """
        时序特性分析
        
        参数:
            window_size: 滑动窗口大小
            
        返回:
            Dict: 时序统计信息
        """
        ts_stats = {}
        
        for col in self.numeric_data.columns:
            col_data = self.numeric_data[col]
            
            # 计算滑动统计量
            rolling_mean = col_data.rolling(window=window_size).mean()
            rolling_std = col_data.rolling(window=window_size).std()
            
            # 趋势分析（线性拟合）
            x = np.arange(len(col_data))
            y = col_data.values
            valid_idx = ~np.isnan(y)
            
            if valid_idx.sum() > 2:
                slope, intercept = np.polyfit(x[valid_idx], y[valid_idx], 1)
                trend = "上升" if slope > 0 else "下降"
            else:
                slope = np.nan
                trend = "未知"
            
            ts_stats[col] = {
                '均值': col_data.mean(),
                '标准差': col_data.std(),
                '趋势斜率': slope,
                '趋势方向': trend,
                '最大值位置': col_data.idxmax(),
                '最小值位置': col_data.idxmin()
            }
        
        ts_df = pd.DataFrame(ts_stats).T
        self.stats_results['timeseries'] = ts_df
        
        print("\n时序特性分析:")
        print(ts_df.round(4))
        
        return ts_stats
    
    def visualize_distribution(self, 
                              columns: Optional[List[str]] = None,
                              figsize: tuple = (15, 10),
                              save_path: Optional[str] = None):
        """
        分布可视化（直方图+KDE）
        
        参数:
            columns: 要可视化的列，None表示所有数值列
            figsize: 图形大小
            save_path: 保存路径
        """
        if columns is None:
            columns = self.numeric_data.columns.tolist()
        else:
            columns = [col for col in columns if col in self.numeric_data.columns]
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            data = self.numeric_data[col].dropna()
            
            # 绘制直方图和KDE
            ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
            
            # KDE
            try:
                data.plot(kind='density', ax=ax, color='red', linewidth=2)
            except:
                pass
            
            ax.set_title(f'{col} 分布', fontsize=10)
            ax.set_xlabel('值', fontsize=8)
            ax.set_ylabel('密度', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分布图已保存至: {save_path}")
        
        plt.show()
    
    def visualize_boxplot(self,
                         columns: Optional[List[str]] = None,
                         figsize: tuple = (15, 6),
                         save_path: Optional[str] = None):
        """
        箱线图可视化
        
        参数:
            columns: 要可视化的列
            figsize: 图形大小
            save_path: 保存路径
        """
        if columns is None:
            columns = self.numeric_data.columns.tolist()
        else:
            columns = [col for col in columns if col in self.numeric_data.columns]
        
        # 标准化数据用于可视化
        plot_data = self.numeric_data[columns].copy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制箱线图
        plot_data.boxplot(ax=ax, rot=45)
        ax.set_title('特征箱线图 (Box Plot)', fontsize=12)
        ax.set_ylabel('值', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"箱线图已保存至: {save_path}")
        
        plt.show()
    
    def visualize_correlation_heatmap(self,
                                     method: str = 'pearson',
                                     figsize: tuple = (12, 10),
                                     save_path: Optional[str] = None):
        """
        相关性热图
        
        参数:
            method: 相关性计算方法
            figsize: 图形大小
            save_path: 保存路径
        """
        corr_matrix = self.numeric_data.corr(method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(f'特征相关性热图 ({method.capitalize()})', fontsize=12, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性热图已保存至: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict:
        """
        生成综合统计分析报告
        
        参数:
            save_path: 报告保存路径（可选）
            
        返回:
            Dict: 包含所有统计结果的字典
        """
        print("\n" + "="*60)
        print("综合统计分析报告")
        print("="*60)
        
        # 执行各项分析
        self.basic_statistics()
        self.distribution_analysis()
        self.missing_value_analysis()
        self.outlier_detection()
        self.time_series_analysis()
        
        report = {
            '数据形状': self.data.shape,
            '基础统计': self.stats_results.get('basic'),
            '分布特征': self.stats_results.get('distribution'),
            '缺失值': self.stats_results.get('missing'),
            '异常值': self.stats_results.get('outliers'),
            '时序特征': self.stats_results.get('timeseries')
        }
        
        if save_path:
            # 保存报告到Excel
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                for sheet_name, df in report.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name)
            
            print(f"\n报告已保存至: {save_path}")
        
        print("="*60 + "\n")
        
        return report
