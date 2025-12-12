"""
相关性分析模块 (Correlation Analysis Module)
使用多种方法进行数据相关性分析，支持特征选择和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CorrelationAnalyzer:
    """
    相关性分析器
    
    功能：
    1. 多种相关性分析方法（Pearson、Spearman、Kendall、互信息等）
    2. 目标变量相关性分析
    3. 特征筛选和排序
    4. 相关性可视化
    5. 生成相关性分析表格
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化相关性分析器
        
        参数:
            data: 要分析的数据框
        """
        self.data = data.copy()
        self.numeric_data = data.select_dtypes(include=[np.number])
        self.correlation_results = {}
        
    def pearson_correlation(self, target: Optional[str] = None) -> pd.DataFrame:
        """
        Pearson相关性分析（线性相关）
        
        参数:
            target: 目标变量名，如果指定则计算所有特征与目标的相关性
            
        返回:
            pd.DataFrame: 相关系数矩阵或相关系数序列
        """
        if target is not None:
            if target not in self.numeric_data.columns:
                raise ValueError(f"目标变量 '{target}' 不存在")
            
            corr_with_target = {}
            for col in self.numeric_data.columns:
                if col != target:
                    valid_data = self.numeric_data[[col, target]].dropna()
                    if len(valid_data) > 2:
                        corr, p_value = pearsonr(valid_data[col], valid_data[target])
                        corr_with_target[col] = {
                            'correlation': corr,
                            'abs_correlation': abs(corr),
                            'p_value': p_value,
                            'significant': '是' if p_value < 0.05 else '否'
                        }
            
            result = pd.DataFrame(corr_with_target).T
            result = result.sort_values('abs_correlation', ascending=False)
            
        else:
            # 计算完整的相关系数矩阵
            result = self.numeric_data.corr(method='pearson')
        
        self.correlation_results['pearson'] = result
        
        print("\nPearson相关性分析:")
        print(result.head(10))
        
        return result
    
    def spearman_correlation(self, target: Optional[str] = None) -> pd.DataFrame:
        """
        Spearman相关性分析（单调相关）
        
        参数:
            target: 目标变量名
            
        返回:
            pd.DataFrame: 相关系数矩阵或相关系数序列
        """
        if target is not None:
            if target not in self.numeric_data.columns:
                raise ValueError(f"目标变量 '{target}' 不存在")
            
            corr_with_target = {}
            for col in self.numeric_data.columns:
                if col != target:
                    valid_data = self.numeric_data[[col, target]].dropna()
                    if len(valid_data) > 2:
                        corr, p_value = spearmanr(valid_data[col], valid_data[target])
                        corr_with_target[col] = {
                            'correlation': corr,
                            'abs_correlation': abs(corr),
                            'p_value': p_value,
                            'significant': '是' if p_value < 0.05 else '否'
                        }
            
            result = pd.DataFrame(corr_with_target).T
            result = result.sort_values('abs_correlation', ascending=False)
            
        else:
            # 计算完整的相关系数矩阵
            result = self.numeric_data.corr(method='spearman')
        
        self.correlation_results['spearman'] = result
        
        print("\nSpearman相关性分析:")
        print(result.head(10))
        
        return result
    
    def kendall_correlation(self, target: Optional[str] = None) -> pd.DataFrame:
        """
        Kendall相关性分析（适用于小样本）
        
        参数:
            target: 目标变量名
            
        返回:
            pd.DataFrame: 相关系数矩阵
        """
        if target is not None:
            # 计算与目标变量的相关性
            result = self.numeric_data.corrwith(self.numeric_data[target], method='kendall')
            result = result.drop(target).sort_values(ascending=False, key=abs)
        else:
            result = self.numeric_data.corr(method='kendall')
        
        self.correlation_results['kendall'] = result
        
        print("\nKendall相关性分析:")
        print(result.head(10))
        
        return result
    
    def mutual_information(self, 
                          target: str,
                          n_neighbors: int = 3,
                          random_state: int = 42) -> pd.DataFrame:
        """
        互信息分析（捕获非线性关系）
        
        参数:
            target: 目标变量名
            n_neighbors: 最近邻数量
            random_state: 随机种子
            
        返回:
            pd.DataFrame: 互信息得分
        """
        if target not in self.numeric_data.columns:
            raise ValueError(f"目标变量 '{target}' 不存在")
        
        X = self.numeric_data.drop(columns=[target]).fillna(0)
        y = self.numeric_data[target].fillna(0)
        
        # 计算互信息
        mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=random_state)
        
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_information': mi_scores,
            'normalized_mi': mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
        }).sort_values('mutual_information', ascending=False)
        
        mi_df = mi_df.set_index('feature')
        
        self.correlation_results['mutual_info'] = mi_df
        
        print("\n互信息分析:")
        print(mi_df.head(10))
        
        return mi_df
    
    def correlation_with_target(self,
                               target: str,
                               methods: List[str] = ['pearson', 'spearman', 'mutual_info']) -> pd.DataFrame:
        """
        使用多种方法分析与目标变量的相关性
        
        参数:
            target: 目标变量名
            methods: 要使用的方法列表
            
        返回:
            pd.DataFrame: 综合相关性分析结果
        """
        results = {}
        
        for method in methods:
            if method == 'pearson':
                corr_df = self.pearson_correlation(target)
                if 'correlation' in corr_df.columns:
                    results['pearson_corr'] = corr_df['correlation']
                    
            elif method == 'spearman':
                corr_df = self.spearman_correlation(target)
                if 'correlation' in corr_df.columns:
                    results['spearman_corr'] = corr_df['correlation']
                    
            elif method == 'kendall':
                corr_series = self.kendall_correlation(target)
                results['kendall_corr'] = corr_series
                
            elif method == 'mutual_info':
                mi_df = self.mutual_information(target)
                results['mutual_info'] = mi_df['mutual_information']
        
        # 合并结果
        combined_df = pd.DataFrame(results)
        
        # 计算平均排名
        rankings = combined_df.rank(ascending=False)
        combined_df['avg_rank'] = rankings.mean(axis=1)
        combined_df['avg_score'] = combined_df.abs().mean(axis=1)
        
        combined_df = combined_df.sort_values('avg_rank')
        
        self.correlation_results['combined'] = combined_df
        
        print("\n综合相关性分析:")
        print(combined_df.round(4))
        
        return combined_df
    
    def feature_selection(self,
                         target: str,
                         method: str = 'pearson',
                         threshold: Optional[float] = None,
                         top_k: Optional[int] = None) -> List[str]:
        """
        基于相关性的特征选择
        
        参数:
            target: 目标变量名
            method: 相关性方法
            threshold: 相关性阈值
            top_k: 选择前k个特征
            
        返回:
            List[str]: 选择的特征列表
        """
        if method == 'pearson':
            corr_df = self.pearson_correlation(target)
        elif method == 'spearman':
            corr_df = self.spearman_correlation(target)
        elif method == 'mutual_info':
            corr_df = self.mutual_information(target)
            method = 'mutual_information'
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 根据阈值或top_k筛选
        if threshold is not None:
            if method in ['pearson', 'spearman']:
                selected_features = corr_df[corr_df['abs_correlation'] >= threshold].index.tolist()
            else:
                selected_features = corr_df[corr_df['mutual_information'] >= threshold].index.tolist()
        elif top_k is not None:
            selected_features = corr_df.head(top_k).index.tolist()
        else:
            selected_features = corr_df.index.tolist()
        
        print(f"\n选择的特征数量: {len(selected_features)}")
        print(f"选择的特征: {selected_features[:10]}...")
        
        return selected_features
    
    def visualize_correlation_matrix(self,
                                    method: str = 'pearson',
                                    figsize: tuple = (12, 10),
                                    save_path: Optional[str] = None):
        """
        可视化相关性矩阵
        
        参数:
            method: 相关性方法
            figsize: 图形大小
            save_path: 保存路径
        """
        corr_matrix = self.numeric_data.corr(method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix,
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(f'相关性矩阵 ({method.capitalize()})', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性矩阵图已保存至: {save_path}")
        
        plt.show()
    
    def visualize_correlation_with_target(self,
                                         target: str,
                                         method: str = 'pearson',
                                         top_k: int = 15,
                                         figsize: tuple = (12, 8),
                                         save_path: Optional[str] = None):
        """
        可视化与目标变量的相关性
        
        参数:
            target: 目标变量名
            method: 相关性方法
            top_k: 显示前k个特征
            figsize: 图形大小
            save_path: 保存路径
        """
        if method == 'pearson':
            corr_df = self.pearson_correlation(target)
            if 'correlation' in corr_df.columns:
                plot_data = corr_df.head(top_k)['correlation']
            else:
                plot_data = corr_df.head(top_k)
        elif method == 'spearman':
            corr_df = self.spearman_correlation(target)
            if 'correlation' in corr_df.columns:
                plot_data = corr_df.head(top_k)['correlation']
            else:
                plot_data = corr_df.head(top_k)
        elif method == 'mutual_info':
            corr_df = self.mutual_information(target)
            plot_data = corr_df.head(top_k)['mutual_information']
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制柱状图
        colors = ['red' if x < 0 else 'blue' for x in plot_data.values]
        plot_data.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        
        ax.set_xlabel('相关性系数', fontsize=10)
        ax.set_ylabel('特征', fontsize=10)
        ax.set_title(f'与 {target} 的相关性 (Top {top_k}, {method.capitalize()})', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性柱状图已保存至: {save_path}")
        
        plt.show()
    
    def visualize_scatter_matrix(self,
                                features: Optional[List[str]] = None,
                                figsize: tuple = (15, 15),
                                save_path: Optional[str] = None):
        """
        绘制散点矩阵
        
        参数:
            features: 要显示的特征列表
            figsize: 图形大小
            save_path: 保存路径
        """
        if features is None:
            features = self.numeric_data.columns.tolist()[:5]  # 默认显示前5个
        else:
            features = [f for f in features if f in self.numeric_data.columns]
        
        plot_data = self.numeric_data[features]
        
        # 创建散点矩阵
        pd.plotting.scatter_matrix(plot_data,
                                  figsize=figsize,
                                  alpha=0.6,
                                  diagonal='kde')
        
        plt.suptitle('特征散点矩阵', fontsize=14, y=1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"散点矩阵已保存至: {save_path}")
        
        plt.show()
    
    def generate_correlation_table(self,
                                  target: str,
                                  methods: List[str] = ['pearson', 'spearman', 'mutual_info'],
                                  save_path: Optional[str] = None) -> pd.DataFrame:
        """
        生成综合相关性分析表格
        
        参数:
            target: 目标变量名
            methods: 使用的方法列表
            save_path: 保存路径
            
        返回:
            pd.DataFrame: 相关性分析表格
        """
        # 执行综合相关性分析
        combined_df = self.correlation_with_target(target, methods)
        
        # 添加排名信息
        combined_df['rank'] = range(1, len(combined_df) + 1)
        
        # 重新排列列
        cols = ['rank', 'avg_score'] + [col for col in combined_df.columns if col not in ['rank', 'avg_score', 'avg_rank']]
        result_df = combined_df[cols]
        
        print("\n相关性分析表格:")
        print(result_df.round(4))
        
        if save_path:
            result_df.to_csv(save_path, index=True, encoding='utf-8-sig')
            print(f"\n相关性表格已保存至: {save_path}")
        
        return result_df
