import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class EnzymeActivityAssay:
    """
    转氨酶活性测定的模拟实验类
    """
    
    def __init__(self, output_dir='results'):
        """
        初始化活性测定实验
        
        Parameters:
        -----------
        output_dir : str
            结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 实验ID，用于标识不同的实验批次
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 日志文件
        self.log_file = os.path.join(output_dir, f"activity_assay_{self.experiment_id}.log")
        with open(self.log_file, 'w') as f:
            f.write(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def measure_activity(self, enzyme_variants, substrates, ph_range=(6.5, 9.0, 0.5), replicates=3):
        """
        测量不同变体在不同pH下的催化活性
        
        Parameters:
        -----------
        enzyme_variants : list of str
            酶变体ID列表
        substrates : list of str
            底物ID列表
        ph_range : tuple
            pH范围，格式为(起始值, 结束值, 步长)
        replicates : int
            重复测量次数
        
        Returns:
        --------
        pandas.DataFrame
            活性测定结果
        """
        # 生成pH列表
        ph_values = np.arange(ph_range[0], ph_range[1] + 0.01, ph_range[2])
        
        # 记录数据
        results = []
        
        # 模拟实验测量
        for variant in enzyme_variants:
            for substrate in substrates:
                for ph in ph_values:
                    # 模拟催化活性测量
                    activities = self._simulate_activity_measurement(variant, substrate, ph, replicates)
                    
                    # 计算均值和标准差
                    mean_activity = np.mean(activities)
                    std_activity = np.std(activities)
                    
                    # 记录结果
                    results.append({
                        'Variant': variant,
                        'Substrate': substrate,
                        'pH': ph,
                        'Activity_Mean': mean_activity,
                        'Activity_Std': std_activity,
                        'Activities': activities
                    })
                    
                    # 记录日志
                    self._log(f"测量 {variant} 对 {substrate} 在 pH {ph} 下的活性: {mean_activity:.2f} ± {std_activity:.2f} U/mg")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存结果
        output_file = os.path.join(self.output_dir, f"activity_results_{self.experiment_id}.csv")
        results_df.to_csv(output_file, index=False)
        
        return results_df
    
    def _simulate_activity_measurement(self, variant, substrate, ph, replicates):
        """
        模拟测量酶活性
        
        此处我们使用模拟数据，实际应用中应替换为真实的实验测量
        
        Parameters:
        -----------
        variant : str
            酶变体ID
        substrate : str
            底物ID
        ph : float
            pH值
        replicates : int
            重复测量次数
        
        Returns:
        --------
        list
            活性测量值列表(U/mg)
        """
        # 不同变体的基准活性（根据研究中的数据模拟）
        # 在实际应用中，这里应该是真实的实验测量
        base_activities = {
            '3FCR-3M': 10.0,
            '3FCR-3M-F168Y': 16.0,
            '3FCR-3M-L58M': 12.0,
            '3FCR-3M-W59C': 11.0,
            '3FCR-3M-F168Y-W59C': 18.0,
            '3FCR-3M-L58M-F168Y': 17.0
        }
        
        # 默认基准活性
        base_activity = base_activities.get(variant, 10.0)
        
        # 底物对活性的影响
        substrate_factor = 1.0
        if substrate == '1a':
            substrate_factor = 1.0
        elif substrate == '1f':
            substrate_factor = 0.8
        
        # pH对活性的影响 - 不同变体有不同的pH依赖性
        ph_factor = 0.0
        
        if variant == '3FCR-3M' or variant == '3FCR-3M-F168Y':
            # 标准pH曲线，在pH 9.0附近有最大活性
            ph_factor = -1.5 * (ph - 9.0) ** 2 + 1.0
        elif variant == '3FCR-3M-L58M' or variant == '3FCR-3M-L58M-F168Y':
            # 平缓的pH曲线，在较宽的pH范围内有高活性
            ph_factor = -0.7 * (ph - 8.5) ** 2 + 1.0
        elif variant == '3FCR-3M-W59C' or variant == '3FCR-3M-F168Y-W59C':
            # 平缓的pH曲线，在中性pH下有较好的活性
            ph_factor = -0.5 * (ph - 8.0) ** 2 + 0.9
        else:
            # 默认pH曲线
            ph_factor = -1.0 * (ph - 9.0) ** 2 + 1.0
        
        # 计算平均活性
        mean_activity = base_activity * substrate_factor * max(0.05, ph_factor)
        
        # 添加实验误差（正态分布）
        activities = np.random.normal(mean_activity, mean_activity * 0.05, replicates)
        
        # 确保活性非负
        activities = np.maximum(0.01, activities)
        
        return activities.tolist()
    
    def plot_ph_activity_profile(self, results_df, output_file=None):
        """
        绘制pH-活性曲线
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            活性测定结果
        output_file : str, optional
            输出文件路径
        
        Returns:
        --------
        matplotlib.figure.Figure
            图形对象
        """
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 设置风格
        sns.set_style('whitegrid')
        
        # 获取唯一的变体和底物
        variants = results_df['Variant'].unique()
        substrates = results_df['Substrate'].unique()
        
        # 为每个底物创建子图
        for i, substrate in enumerate(substrates):
            plt.subplot(1, len(substrates), i+1)
            
            # 筛选当前底物的数据
            substrate_data = results_df[results_df['Substrate'] == substrate]
            
            # 为每个变体绘制曲线
            for variant in variants:
                variant_data = substrate_data[substrate_data['Variant'] == variant]
                
                # 排序数据
                variant_data = variant_data.sort_values('pH')
                
                plt.errorbar(
                    variant_data['pH'],
                    variant_data['Activity_Mean'],
                    yerr=variant_data['Activity_Std'],
                    marker='o',
                    label=variant
                )
            
            plt.title(f'底物 {substrate} 的pH-活性曲线')
            plt.xlabel('pH')
            plt.ylabel('活性 (U/mg)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图形
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"ph_activity_profile_{self.experiment_id}.png")
        
        plt.savefig(output_file, dpi=300)
        
        return plt.gcf()
    
    def compare_variants_at_ph(self, results_df, ph_value, output_file=None):
        """
        比较不同变体在特定pH下的活性
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            活性测定结果
        ph_value : float
            目标pH值
        output_file : str, optional
            输出文件路径
        
        Returns:
        --------
        matplotlib.figure.Figure
            图形对象
        """
        # 筛选最接近目标pH的数据
        ph_diff = np.abs(results_df['pH'] - ph_value)
        min_diff = ph_diff.min()
        target_ph_data = results_df[ph_diff <= min_diff + 1e-6]
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 设置风格
        sns.set_style('whitegrid')
        
        # 获取唯一的底物和变体
        substrates = target_ph_data['Substrate'].unique()
        variants = target_ph_data['Variant'].unique()
        
        # 定义柱状图宽度和组间距
        bar_width = 0.2
        group_width = bar_width * len(substrates)
        
        # 创建柱状图
        for i, substrate in enumerate(substrates):
            substrate_data = target_ph_data[target_ph_data['Substrate'] == substrate]
            substrate_data = substrate_data.set_index('Variant')
            
            # 计算每个变体的x位置
            x_positions = np.arange(len(variants))
            
            # 计算当前底物的位置偏移
            offset = (i - (len(substrates) - 1) / 2) * bar_width
            
            # 绘制柱状图
            plt.bar(
                x_positions + offset,
                [substrate_data.loc[v, 'Activity_Mean'] if v in substrate_data.index else 0 for v in variants],
                width=bar_width,
                yerr=[substrate_data.loc[v, 'Activity_Std'] if v in substrate_data.index else 0 for v in variants],
                label=substrate
            )
        
        plt.title(f'不同变体在pH {target_ph_data["pH"].iloc[0]:.1f}下的活性比较')
        plt.xlabel('变体')
        plt.ylabel('活性 (U/mg)')
        plt.legend(title='底物')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴刻度
        plt.xticks(np.arange(len(variants)), variants, rotation=45)
        
        plt.tight_layout()
        
        # 保存图形
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"variant_comparison_pH{ph_value}_{self.experiment_id}.png")
        
        plt.savefig(output_file, dpi=300)
        
        return plt.gcf()
    
    def _log(self, message):
        """
        记录日志
        
        Parameters:
        -----------
        message : str
            日志消息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        print(log_message) 