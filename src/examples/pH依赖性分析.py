"""
pH依赖性分析示例脚本

本脚本演示如何分析不同转氨酶变体的pH依赖性曲线特征，并提取关键参数进行比较。
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.activity_assay import EnzymeActivityAssay

def ph_activity_model(ph, a, b, c, ph_opt):
    """
    pH-活性关系的高斯模型
    
    Parameters:
    -----------
    ph : float
        pH值
    a : float
        曲线高度参数
    b : float
        曲线宽度参数
    c : float
        基线活性参数
    ph_opt : float
        最适pH值
    
    Returns:
    --------
    float
        预测的酶活性
    """
    return a * np.exp(-(ph - ph_opt)**2 / (2 * b**2)) + c

def analyze_ph_dependence():
    """运行pH依赖性分析工作流程"""
    print("=== 转氨酶pH依赖性分析工作流程 ===")
    
    # 创建结果目录
    results_dir = 'examples_results/ph_analysis'
    os.makedirs(results_dir, exist_ok=True)
    
    # 步骤1：测量活性数据
    print("\n1. 测量不同pH下的活性数据")
    assay = EnzymeActivityAssay(output_dir=results_dir)
    
    # 定义要测试的变体和底物
    variants = [
        '3FCR-3M',           # 基础模板
        '3FCR-3M-F168Y',     # 单突变体，高活性
        '3FCR-3M-L58M',      # 单突变体，平缓的pH曲线
        '3FCR-3M-F168Y-W59C' # 双突变体，在中性pH下活性更高
    ]
    
    substrate = '1a'  # 选择一个底物进行分析
    
    # 测量不同pH条件下的活性，使用更精细的pH梯度
    results = assay.measure_activity(
        enzyme_variants=variants,
        substrates=[substrate],
        ph_range=(6.0, 9.5, 0.2)  # 从pH 6.0到9.5，步长0.2
    )
    
    # 步骤2：拟合pH-活性曲线
    print("\n2. 拟合pH-活性曲线模型")
    
    # 准备存储拟合参数的数据框
    fit_params = []
    
    # 为每个变体拟合曲线
    plt.figure(figsize=(12, 8))
    
    # 定义颜色映射
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, variant in enumerate(variants):
        # 获取变体数据
        variant_data = results[results['Variant'] == variant]
        ph_values = variant_data['pH'].values
        activities = variant_data['Activity_Mean'].values
        
        # 拟合曲线参数
        try:
            # 初始参数猜测：a=最大活性, b=1.0, c=最小活性, ph_opt=8.5
            initial_guess = [
                max(activities) - min(activities),
                1.0,
                min(activities),
                8.5
            ]
            
            # 约束参数范围
            bounds = (
                [0, 0.1, 0, 6.0],  # 下限
                [100, 5.0, 10, 10.0]  # 上限
            )
            
            params, _ = curve_fit(
                ph_activity_model, 
                ph_values, 
                activities, 
                p0=initial_guess,
                bounds=bounds
            )
            
            # 存储拟合参数
            a, b, c, ph_opt = params
            fit_params.append({
                'Variant': variant,
                'a': a,
                'b': b,
                'c': c,
                'ph_opt': ph_opt,
                'max_activity': a + c,
                'ph_range_width': 2 * b  # 活性超过最大值一半的pH范围
            })
            
            # 生成拟合曲线
            ph_fit = np.linspace(6.0, 9.5, 100)
            activity_fit = ph_activity_model(ph_fit, *params)
            
            # 绘制原始数据点和拟合曲线
            plt.errorbar(
                ph_values, 
                activities, 
                yerr=variant_data['Activity_Std'].values, 
                marker='o', 
                linestyle='none',
                color=colors[i],
                label=f'{variant} 实验数据'
            )
            plt.plot(
                ph_fit, 
                activity_fit, 
                '-', 
                color=colors[i], 
                linewidth=2,
                label=f'{variant} 拟合曲线 (pH_opt = {ph_opt:.2f})'
            )
            
        except RuntimeError:
            print(f"警告: 无法为变体 {variant} 拟合曲线")
    
    plt.title(f'转氨酶变体的pH-活性曲线拟合 (底物: {substrate})')
    plt.xlabel('pH')
    plt.ylabel('活性 (U/mg)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/ph_activity_fitting.png', dpi=300)
    
    # 步骤3：分析pH依赖性参数
    print("\n3. 分析pH依赖性参数")
    
    # 将拟合参数转换为DataFrame
    params_df = pd.DataFrame(fit_params)
    
    # 保存拟合参数
    params_df.to_csv(f'{results_dir}/ph_dependence_parameters.csv', index=False)
    
    # 绘制最适pH柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(variants)), 
        params_df['ph_opt'],
        color=[plt.cm.viridis(i/len(variants)) for i in range(len(variants))]
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.title('不同转氨酶变体的最适pH值')
    plt.xlabel('变体')
    plt.ylabel('最适pH值')
    plt.xticks(range(len(variants)), variants, rotation=45)
    plt.ylim(6, 10)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/optimal_ph_comparison.png', dpi=300)
    
    # 绘制pH适应范围宽度柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(variants)), 
        params_df['ph_range_width'],
        color=[plt.cm.plasma(i/len(variants)) for i in range(len(variants))]
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.title('不同转氨酶变体的pH适应范围宽度')
    plt.xlabel('变体')
    plt.ylabel('pH范围宽度')
    plt.xticks(range(len(variants)), variants, rotation=45)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/ph_range_width_comparison.png', dpi=300)
    
    # 步骤4：计算中性pH区域的相对活性
    print("\n4. 计算中性pH区域的相对活性")
    
    # 定义中性pH区域
    neutral_ph_min = 7.0
    neutral_ph_max = 7.8
    
    # 在中性pH范围内进行积分计算相对面积
    neutral_ph = np.linspace(neutral_ph_min, neutral_ph_max, 100)
    
    neutral_activities = []
    total_activities = []
    
    for _, row in params_df.iterrows():
        variant = row['Variant']
        params = [row['a'], row['b'], row['c'], row['ph_opt']]
        
        # 计算中性pH区域的活性
        neutral_activity = np.trapz(
            ph_activity_model(neutral_ph, *params),
            neutral_ph
        )
        neutral_activities.append(neutral_activity)
        
        # 计算全pH区域的活性
        all_ph = np.linspace(6.0, 9.5, 100)
        total_activity = np.trapz(
            ph_activity_model(all_ph, *params),
            all_ph
        )
        total_activities.append(total_activity)
    
    # 计算中性pH活性占比
    neutral_ratio = [n/t for n, t in zip(neutral_activities, total_activities)]
    
    # 添加到参数DataFrame
    params_df['neutral_activity'] = neutral_activities
    params_df['total_activity'] = total_activities
    params_df['neutral_ratio'] = neutral_ratio
    
    # 保存更新后的参数
    params_df.to_csv(f'{results_dir}/ph_dependence_parameters.csv', index=False)
    
    # 绘制中性pH活性比例柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(variants)), 
        params_df['neutral_ratio'],
        color=[plt.cm.coolwarm(i/len(variants)) for i in range(len(variants))]
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.2%}',
            ha='center',
            va='bottom'
        )
    
    plt.title(f'不同转氨酶变体在中性pH区域({neutral_ph_min}-{neutral_ph_max})的活性占比')
    plt.xlabel('变体')
    plt.ylabel('中性pH活性占比')
    plt.xticks(range(len(variants)), variants, rotation=45)
    plt.ylim(0, max(neutral_ratio) * 1.2)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/neutral_ph_activity_ratio.png', dpi=300)
    
    # 输出分析结果
    print("\n不同变体的pH依赖性参数:")
    for _, row in params_df.iterrows():
        print(f"变体: {row['Variant']}")
        print(f"  最适pH: {row['ph_opt']:.2f}")
        print(f"  pH适应范围宽度: {row['ph_range_width']:.2f}")
        print(f"  中性pH活性占比: {row['neutral_ratio']:.2%}")
        print(f"  最大活性: {row['max_activity']:.2f} U/mg")
    
    # 找出中性pH活性最高的变体
    best_neutral_idx = params_df['neutral_activity'].idxmax()
    best_neutral_variant = params_df.loc[best_neutral_idx]
    
    print(f"\n中性pH区域活性最高的变体: {best_neutral_variant['Variant']}")
    print(f"  中性pH活性积分: {best_neutral_variant['neutral_activity']:.2f}")
    print(f"  相对于野生型的提升: {best_neutral_variant['neutral_activity'] / params_df[params_df['Variant'] == '3FCR-3M']['neutral_activity'].values[0]:.2f} 倍")
    
    print("\npH依赖性分析工作流程完成！结果保存在 'examples_results/ph_analysis' 目录中")
    
    return params_df

if __name__ == "__main__":
    # 运行pH依赖性分析工作流程
    analyze_ph_dependence() 