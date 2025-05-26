"""
统计分析器
提供博弈实验数据的统计分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import kstest, normaltest, pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticalSummary:
    """统计摘要"""
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    q25: float  # 第一四分位数
    q75: float  # 第三四分位数
    skewness: float  # 偏度
    kurtosis: float  # 峰度
    variance: float  # 方差
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'median': self.median,
            'q25': self.q25,
            'q75': self.q75,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'variance': self.variance
        }


@dataclass
class CorrelationAnalysis:
    """相关性分析结果"""
    pearson_corr: float
    pearson_p_value: float
    spearman_corr: float
    spearman_p_value: float
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """检查相关性是否显著"""
        return self.pearson_p_value < alpha or self.spearman_p_value < alpha
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'pearson_correlation': self.pearson_corr,
            'pearson_p_value': self.pearson_p_value,
            'spearman_correlation': self.spearman_corr,
            'spearman_p_value': self.spearman_p_value
        }


@dataclass
class NormalityTest:
    """正态性检验结果"""
    shapiro_statistic: float
    shapiro_p_value: float
    ks_statistic: float
    ks_p_value: float
    dagostino_statistic: float
    dagostino_p_value: float
    
    def is_normal(self, alpha: float = 0.05) -> bool:
        """检查是否服从正态分布"""
        return (self.shapiro_p_value > alpha and 
                self.ks_p_value > alpha and 
                self.dagostino_p_value > alpha)
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'shapiro_statistic': self.shapiro_statistic,
            'shapiro_p_value': self.shapiro_p_value,
            'ks_statistic': self.ks_statistic,
            'ks_p_value': self.ks_p_value,
            'dagostino_statistic': self.dagostino_statistic,
            'dagostino_p_value': self.dagostino_p_value
        }


@dataclass
class StationarityTest:
    """平稳性检验结果"""
    adf_statistic: float
    adf_p_value: float
    adf_critical_values: Dict[str, float]
    is_stationary: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'adf_statistic': self.adf_statistic,
            'adf_p_value': self.adf_p_value,
            'adf_critical_values': self.adf_critical_values,
            'is_stationary': self.is_stationary
        }


class StatisticalAnalyzer:
    """
    统计分析器
    提供博弈实验数据的统计分析功能
    """
    
    def __init__(self):
        """初始化统计分析器"""
        self.logger = logging.getLogger(__name__)
    
    def descriptive_statistics(self, data: Union[List, np.ndarray, pd.Series]) -> StatisticalSummary:
        """
        计算描述性统计
        
        Args:
            data: 数据序列
            
        Returns:
            StatisticalSummary: 统计摘要
        """
        try:
            data = np.array(data)
            
            # 基本统计量
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            min_val = np.min(data)
            max_val = np.max(data)
            median = np.median(data)
            q25 = np.percentile(data, 25)
            q75 = np.percentile(data, 75)
            variance = np.var(data, ddof=1)
            
            # 分布特征
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            return StatisticalSummary(
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                median=median,
                q25=q25,
                q75=q75,
                skewness=skewness,
                kurtosis=kurtosis,
                variance=variance
            )
            
        except Exception as e:
            self.logger.error(f"计算描述性统计时出错: {e}")
            raise
    
    def correlation_analysis(self, x: Union[List, np.ndarray], 
                           y: Union[List, np.ndarray]) -> CorrelationAnalysis:
        """
        计算两个变量的相关性
        
        Args:
            x: 第一个变量
            y: 第二个变量
            
        Returns:
            CorrelationAnalysis: 相关性分析结果
        """
        try:
            x = np.array(x)
            y = np.array(y)
            
            # Pearson相关系数
            pearson_corr, pearson_p = pearsonr(x, y)
            
            # Spearman相关系数
            spearman_corr, spearman_p = spearmanr(x, y)
            
            return CorrelationAnalysis(
                pearson_corr=pearson_corr,
                pearson_p_value=pearson_p,
                spearman_corr=spearman_corr,
                spearman_p_value=spearman_p
            )
            
        except Exception as e:
            self.logger.error(f"计算相关性分析时出错: {e}")
            raise
    
    def normality_test(self, data: Union[List, np.ndarray]) -> NormalityTest:
        """
        正态性检验
        
        Args:
            data: 数据序列
            
        Returns:
            NormalityTest: 正态性检验结果
        """
        try:
            data = np.array(data)
            
            # Shapiro-Wilk检验
            shapiro_stat, shapiro_p = stats.shapiro(data)
            
            # Kolmogorov-Smirnov检验
            ks_stat, ks_p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            
            # D'Agostino和Pearson检验
            dagostino_stat, dagostino_p = normaltest(data)
            
            return NormalityTest(
                shapiro_statistic=shapiro_stat,
                shapiro_p_value=shapiro_p,
                ks_statistic=ks_stat,
                ks_p_value=ks_p,
                dagostino_statistic=dagostino_stat,
                dagostino_p_value=dagostino_p
            )
            
        except Exception as e:
            self.logger.error(f"正态性检验时出错: {e}")
            raise
    
    def stationarity_test(self, data: Union[List, np.ndarray]) -> StationarityTest:
        """
        平稳性检验（ADF检验）
        
        Args:
            data: 时间序列数据
            
        Returns:
            StationarityTest: 平稳性检验结果
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            data = np.array(data)
            
            # ADF检验
            result = adfuller(data)
            adf_stat = result[0]
            adf_p = result[1]
            critical_values = result[4]
            
            # 判断是否平稳（p值小于0.05则拒绝原假设，序列平稳）
            is_stationary = adf_p < 0.05
            
            return StationarityTest(
                adf_statistic=adf_stat,
                adf_p_value=adf_p,
                adf_critical_values=critical_values,
                is_stationary=is_stationary
            )
            
        except ImportError:
            self.logger.warning("未安装statsmodels，跳过平稳性检验")
            return StationarityTest(
                adf_statistic=0.0,
                adf_p_value=1.0,
                adf_critical_values={},
                is_stationary=False
            )
        except Exception as e:
            self.logger.error(f"平稳性检验时出错: {e}")
            raise
    
    def hypothesis_test(self, group1: Union[List, np.ndarray], 
                       group2: Union[List, np.ndarray],
                       test_type: str = 'ttest') -> Dict[str, float]:
        """
        假设检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            test_type: 检验类型 ('ttest', 'mannwhitney', 'ks')
            
        Returns:
            Dict: 检验结果
        """
        try:
            group1 = np.array(group1)
            group2 = np.array(group2)
            
            if test_type == 'ttest':
                # t检验
                statistic, p_value = stats.ttest_ind(group1, group2)
                test_name = 'T-test'
                
            elif test_type == 'mannwhitney':
                # Mann-Whitney U检验
                statistic, p_value = stats.mannwhitneyu(group1, group2)
                test_name = 'Mann-Whitney U test'
                
            elif test_type == 'ks':
                # Kolmogorov-Smirnov检验
                statistic, p_value = stats.ks_2samp(group1, group2)
                test_name = 'Kolmogorov-Smirnov test'
                
            else:
                raise ValueError(f"不支持的检验类型: {test_type}")
            
            return {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
            
        except Exception as e:
            self.logger.error(f"假设检验时出错: {e}")
            raise
    
    def time_series_analysis(self, data: Union[List, np.ndarray], 
                           window_size: int = 10) -> Dict[str, Any]:
        """
        时间序列分析
        
        Args:
            data: 时间序列数据
            window_size: 移动窗口大小
            
        Returns:
            Dict: 时间序列分析结果
        """
        try:
            data = np.array(data)
            
            # 计算移动平均
            moving_avg = []
            for i in range(len(data) - window_size + 1):
                window = data[i:i + window_size]
                moving_avg.append(np.mean(window))
            
            # 计算趋势
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            # 计算变化率
            change_rates = []
            for i in range(1, len(data)):
                rate = (data[i] - data[i-1]) / data[i-1] if data[i-1] != 0 else 0
                change_rates.append(rate)
            
            # 平稳性检验
            stationarity = self.stationarity_test(data)
            
            return {
                'trend_slope': slope,
                'trend_intercept': intercept,
                'trend_r_squared': r_value ** 2,
                'trend_p_value': p_value,
                'moving_average': moving_avg,
                'change_rates': change_rates,
                'mean_change_rate': np.mean(change_rates),
                'volatility': np.std(change_rates),
                'stationarity': stationarity.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"时间序列分析时出错: {e}")
            raise
    
    def outlier_detection(self, data: Union[List, np.ndarray], 
                         method: str = 'iqr') -> Dict[str, Any]:
        """
        异常值检测
        
        Args:
            data: 数据序列
            method: 检测方法 ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            Dict: 异常值检测结果
        """
        try:
            data = np.array(data)
            
            if method == 'iqr':
                # 四分位距方法
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
                
            elif method == 'zscore':
                # Z分数方法
                z_scores = np.abs(stats.zscore(data))
                threshold = 3
                outliers = data[z_scores > threshold]
                outlier_indices = np.where(z_scores > threshold)[0]
                
            elif method == 'modified_zscore':
                # 修正Z分数方法
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                threshold = 3.5
                outliers = data[np.abs(modified_z_scores) > threshold]
                outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
                
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
            
            return {
                'method': method,
                'outliers': outliers.tolist(),
                'outlier_indices': outlier_indices.tolist(),
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100
            }
            
        except Exception as e:
            self.logger.error(f"异常值检测时出错: {e}")
            raise
    
    def distribution_fitting(self, data: Union[List, np.ndarray]) -> Dict[str, Any]:
        """
        分布拟合
        
        Args:
            data: 数据序列
            
        Returns:
            Dict: 分布拟合结果
        """
        try:
            data = np.array(data)
            
            # 常见分布
            distributions = [
                stats.norm,      # 正态分布
                stats.expon,     # 指数分布
                stats.gamma,     # 伽马分布
                stats.beta,      # 贝塔分布
                stats.lognorm,   # 对数正态分布
                stats.uniform    # 均匀分布
            ]
            
            results = {}
            
            for dist in distributions:
                try:
                    # 拟合参数
                    params = dist.fit(data)
                    
                    # KS检验
                    ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
                    
                    # AIC和BIC
                    log_likelihood = np.sum(dist.logpdf(data, *params))
                    k = len(params)  # 参数个数
                    n = len(data)
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    results[dist.name] = {
                        'parameters': params,
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood
                    }
                    
                except Exception as e:
                    self.logger.warning(f"拟合{dist.name}分布时出错: {e}")
                    continue
            
            # 找到最佳拟合分布（基于AIC）
            if results:
                best_dist = min(results.keys(), key=lambda x: results[x]['aic'])
                results['best_distribution'] = best_dist
            
            return results
            
        except Exception as e:
            self.logger.error(f"分布拟合时出错: {e}")
            raise
    
    def comprehensive_analysis(self, data: Dict[str, Union[List, np.ndarray]]) -> Dict[str, Any]:
        """
        综合统计分析
        
        Args:
            data: 数据字典，key为变量名，value为数据序列
            
        Returns:
            Dict: 综合分析结果
        """
        try:
            results = {}
            
            # 描述性统计
            results['descriptive_statistics'] = {}
            for var_name, var_data in data.items():
                results['descriptive_statistics'][var_name] = self.descriptive_statistics(var_data).to_dict()
            
            # 相关性矩阵
            if len(data) > 1:
                variables = list(data.keys())
                corr_matrix = {}
                
                for i, var1 in enumerate(variables):
                    corr_matrix[var1] = {}
                    for j, var2 in enumerate(variables):
                        if i == j:
                            corr_matrix[var1][var2] = {'pearson_correlation': 1.0, 'pearson_p_value': 0.0}
                        elif i < j:
                            corr_analysis = self.correlation_analysis(data[var1], data[var2])
                            corr_matrix[var1][var2] = corr_analysis.to_dict()
                        else:
                            # 利用对称性
                            corr_matrix[var1][var2] = corr_matrix[var2][var1]
                
                results['correlation_matrix'] = corr_matrix
            
            # 正态性检验
            results['normality_tests'] = {}
            for var_name, var_data in data.items():
                results['normality_tests'][var_name] = self.normality_test(var_data).to_dict()
            
            # 异常值检测
            results['outlier_detection'] = {}
            for var_name, var_data in data.items():
                results['outlier_detection'][var_name] = self.outlier_detection(var_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"综合统计分析时出错: {e}")
            raise
    
    def analyze_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析实验结果数据
        
        Args:
            results: 实验结果字典
            
        Returns:
            Dict: 统计分析结果
        """
        try:
            analysis_results = {}
            
            # 提取需要分析的数据
            data_to_analyze = {}
            
            # 提取轮次结果，可能是对象列表或字典列表
            round_results = []
            if 'round_results' in results:
                round_results = results['round_results']
            elif 'round_results_data' in results:
                round_results = results['round_results_data']
            elif hasattr(results, 'round_results') and results.round_results:
                round_results = results.round_results
            
            if not round_results:
                self.logger.warning("未找到轮次结果数据")
                return {"error": "未找到轮次结果数据"}
            
            # 确定玩家列表
            players = results.get('players', ['player_a', 'player_b'])
            
            # 提取策略和奖励数据
            player_strategies = {}
            player_revenues = {}
            
            for player in players:
                player_strategies[f"{player}_strategy"] = []
                player_revenues[f"{player}_revenue"] = []
            
            for round_result in round_results:
                for player in players:
                    strategy_key = f"{player}_strategy"
                    revenue_key = f"{player}_revenue"
                    
                    if isinstance(round_result, dict):
                        # 字典格式
                        if strategy_key in round_result:
                            player_strategies[strategy_key].append(round_result[strategy_key])
                        elif player + '_strategy' in round_result:
                            player_strategies[strategy_key].append(round_result[player + '_strategy'])
                        
                        if revenue_key in round_result:
                            player_revenues[revenue_key].append(round_result[revenue_key])
                        elif player + '_revenue' in round_result:
                            player_revenues[revenue_key].append(round_result[player + '_revenue'])
                    else:
                        # 对象格式
                        if hasattr(round_result, strategy_key):
                            player_strategies[strategy_key].append(getattr(round_result, strategy_key))
                        elif hasattr(round_result, player + '_strategy'):
                            player_strategies[strategy_key].append(getattr(round_result, player + '_strategy'))
                        
                        if hasattr(round_result, revenue_key):
                            player_revenues[revenue_key].append(getattr(round_result, revenue_key))
                        elif hasattr(round_result, player + '_revenue'):
                            player_revenues[revenue_key].append(getattr(round_result, player + '_revenue'))
            
            data_to_analyze.update(player_strategies)
            data_to_analyze.update(player_revenues)
            
            # 提取纳什距离
            nash_distances = []
            for r in round_results:
                try:
                    if isinstance(r, dict):
                        if 'nash_distance' in r and r['nash_distance'] is not None:
                            nash_dist = r['nash_distance']
                            # 确保提取的是数值
                            if isinstance(nash_dist, (int, float)):
                                nash_distances.append(float(nash_dist))
                            elif isinstance(nash_dist, tuple) and nash_dist and isinstance(nash_dist[0], (int, float)):
                                nash_distances.append(float(nash_dist[0]))
                            elif isinstance(nash_dist, dict):
                                # 如果是字典，尝试获取第一个数值
                                for k, v in nash_dist.items():
                                    if isinstance(v, (int, float)):
                                        nash_distances.append(float(v))
                                        break
                    else:
                        if hasattr(r, 'nash_distance') and r.nash_distance is not None:
                            nash_dist = r.nash_distance
                            # 确保提取的是数值
                            if isinstance(nash_dist, (int, float)):
                                nash_distances.append(float(nash_dist))
                            elif isinstance(nash_dist, tuple) and nash_dist and isinstance(nash_dist[0], (int, float)):
                                nash_distances.append(float(nash_dist[0]))
                            elif isinstance(nash_dist, dict):
                                # 如果是字典，尝试获取第一个数值
                                for k, v in nash_dist.items():
                                    if isinstance(v, (int, float)):
                                        nash_distances.append(float(v))
                                        break
                except Exception as e:
                    self.logger.warning(f"处理nash_distance时出错: {e}")
                    continue
            
            if nash_distances:
                data_to_analyze['nash_distance'] = nash_distances
            
            # 如果找不到结构化数据，尝试从其他键提取数值序列
            if not data_to_analyze:
                for key, value in results.items():
                    if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
                        data_to_analyze[key] = value
            
            # 进行综合分析
            if data_to_analyze:
                analysis_results = self.comprehensive_analysis(data_to_analyze)
                
                # 添加时间序列分析（如果有足够的数据点）
                time_series_analysis = {}
                for var_name, var_data in data_to_analyze.items():
                    if len(var_data) >= 30:  # 至少30个数据点才进行时间序列分析
                        time_series_analysis[var_name] = self.time_series_analysis(var_data)
                
                if time_series_analysis:
                    analysis_results['time_series_analysis'] = time_series_analysis
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"分析数据时出错: {e}")
            # 返回空结果而不是抛出异常，以免中断整个分析流程
            return {"error": str(e)} 