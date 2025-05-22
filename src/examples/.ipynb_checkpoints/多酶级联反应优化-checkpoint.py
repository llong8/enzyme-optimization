"""
多酶级联反应优化示例脚本

本脚本演示如何优化多酶系统中的转氨酶，使其与乳酸脱氢酶(LDH)和葡萄糖脱氢酶(GDH)在中性pH下协同工作。
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.activity_assay import EnzymeActivityAssay

# 定义其他酶的活性模型
def ldh_activity_model(ph):
    """
    乳酸脱氢酶(LDH)的pH-活性关系模型
    
    Parameters:
    -----------
    ph : float or array
        pH值
    
    Returns:
    --------
    float or array
        LDH在给定pH下的相对活性
    """
    # LDH的最适pH约为7.5
    return np.exp(-(ph - 7.5)**2 / (2 * 0.8**2))

def gdh_activity_model(ph):
    """
    葡萄糖脱氢酶(GDH)的pH-活性关系模型
    
    Parameters:
    -----------
    ph : float or array
        pH值
    
    Returns:
    --------
    float or array
        GDH在给定pH下的相对活性
    """
    # GDH的最适pH约为7.75
    return np.exp(-(ph - 7.75)**2 / (2 * 0.9**2))

def optimize_cascade_reaction():
    """运行多酶级联反应优化工作流程"""
    print("=== 转氨酶多酶级联反应优化工作流程 ===")
    
    # 创建结果目录
    results_dir = 'examples_results/cascade_optimization'
    os.makedirs(results_dir, exist_ok=True)
    
    # 步骤1：测量转氨酶变体活性
    print("\n1. 测量不同转氨酶变体的活性")
    assay = EnzymeActivityAssay(output_dir=results_dir)
    
    # 定义要测试的变体和底物
    variants = [
        '3FCR-3M',           # 基础模板
        '3FCR-3M-F168Y',     # 单突变体，高活性
        '3FCR-3M-L58M',      # 单突变体，平缓的pH曲线
        '3FCR-3M-F168Y-W59C' # 双突变体，在中性pH下活性更高
    ]
    
    substrate = '1a'  # 选择一个底物进行分析
    
    # 测量不同pH条件下的活性
    ta_results = assay.measure_activity(
        enzyme_variants=variants,
        substrates=[substrate],
        ph_range=(6.0, 9.0, 0.2)
    )
    
    # 步骤2：计算多酶系统的整体活性
    print("\n2. 计算多酶系统的整体活性")
    
    # 生成pH值范围
    ph_values = np.linspace(6.0, 9.0, 100)
    
    # 计算LDH和GDH的活性曲线
    ldh_activities = ldh_activity_model(ph_values)
    gdh_activities = gdh_activity_model(ph_values)
    
    # 为每个转氨酶变体计算多酶系统的整体活性
    cascade_results = []
    
    for variant in variants:
        # 获取变体在不同pH下的活性数据
        variant_data = ta_results[ta_results['Variant'] == variant]
        
        # 对每个pH值插值计算转氨酶活性
        ta_activities = np.interp(
            ph_values,
            variant_data['pH'].values,
            variant_data['Activity_Mean'].values / variant_data['Activity_Mean'].max()  # 归一化活性
        )
        
        # 计算每个pH点的级联反应整体活性（取三个酶活性的几何平均值）
        overall_activities = np.power(ta_activities * ldh_activities * gdh_activities, 1/3)
        
        # 找出最佳pH
        best_ph_idx = np.argmax(overall_activities)
        best_ph = ph_values[best_ph_idx]
        best_activity = overall_activities[best_ph_idx]
        
        # 存储结果
        cascade_results.append({
            'Variant': variant,
            'Best_pH': best_ph,
            'Max_Activity': best_activity,
            'TA_Activity_at_Best_pH': ta_activities[best_ph_idx],
            'LDH_Activity_at_Best_pH': ldh_activities[best_ph_idx],
            'GDH_Activity_at_Best_pH': gdh_activities[best_ph_idx],
            'pH_Values': ph_values,
            'TA_Activities': ta_activities,
            'Overall_Activities': overall_activities
        })
    
    # 步骤3：可视化各酶活性曲线和整体活性
    print("\n3. 可视化各酶活性曲线和整体活性")
    
    for variant_result in cascade_results:
        variant = variant_result['Variant']
        
        plt.figure(figsize=(12, 8))
        
        # 绘制三种酶的活性曲线
        plt.plot(
            ph_values, 
            variant_result['TA_Activities'], 
            'b-', 
            linewidth=2, 
            label=f'转氨酶 ({variant})'
        )
        plt.plot(
            ph_values, 
            ldh_activities, 
            'g-', 
            linewidth=2, 
            label='乳酸脱氢酶 (LDH)'
        )
        plt.plot(
            ph_values, 
            gdh_activities, 
            'r-', 
            linewidth=2, 
            label='葡萄糖脱氢酶 (GDH)'
        )
        
        # 绘制整体活性曲线
        plt.plot(
            ph_values, 
            variant_result['Overall_Activities'], 
            'k--', 
            linewidth=3, 
            label='级联反应整体活性'
        )
        
        # 标记最佳pH点
        best_ph = variant_result['Best_pH']
        best_activity = variant_result['Max_Activity']
        plt.scatter(
            [best_ph], 
            [best_activity], 
            s=100, 
            c='purple', 
            marker='*', 
            zorder=10
        )
        plt.annotate(
            f'最佳pH: {best_ph:.2f}',
            xy=(best_ph, best_activity),
            xytext=(best_ph + 0.3, best_activity + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=10
        )
        
        plt.title(f'转氨酶变体 {variant} 与LDH和GDH的多酶级联反应')
        plt.xlabel('pH')
        plt.ylabel('相对活性')
        plt.xlim(6.0, 9.0)
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(f'{results_dir}/cascade_activity_{variant}.png', dpi=300)
    
    # 步骤4：比较不同变体的级联反应性能
    print("\n4. 比较不同变体的级联反应性能")
    
    # 创建性能比较数据框
    performance_data = {
        'Variant': [],
        'Best_pH': [],
        'Max_Activity': [],
        'TA_Activity': [],
        'LDH_Activity': [],
        'GDH_Activity': []
    }
    
    for result in cascade_results:
        performance_data['Variant'].append(result['Variant'])
        performance_data['Best_pH'].append(result['Best_pH'])
        performance_data['Max_Activity'].append(result['Max_Activity'])
        performance_data['TA_Activity'].append(result['TA_Activity_at_Best_pH'])
        performance_data['LDH_Activity'].append(result['LDH_Activity_at_Best_pH'])
        performance_data['GDH_Activity'].append(result['GDH_Activity_at_Best_pH'])
    
    performance_df = pd.DataFrame(performance_data)
    
    # 保存性能比较结果
    performance_df.to_csv(f'{results_dir}/cascade_performance.csv', index=False)
    
    # 绘制最佳pH柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(variants)), 
        performance_df['Best_pH'],
        color=[plt.cm.viridis(i/len(variants)) for i in range(len(variants))]
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.05,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.title('不同转氨酶变体在多酶级联反应中的最佳pH')
    plt.xlabel('变体')
    plt.ylabel('最佳pH')
    plt.xticks(range(len(variants)), variants, rotation=45)
    plt.ylim(6.5, 8.5)
    plt.axhline(7.5, color='red', linestyle='--', alpha=0.7, label='LDH最适pH')
    plt.axhline(7.75, color='green', linestyle='--', alpha=0.7, label='GDH最适pH')
    plt.legend()
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/best_ph_comparison.png', dpi=300)
    
    # 绘制最大活性柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(variants)), 
        performance_df['Max_Activity'],
        color=[plt.cm.plasma(i/len(variants)) for i in range(len(variants))]
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    plt.title('不同转氨酶变体在多酶级联反应中的最大整体活性')
    plt.xlabel('变体')
    plt.ylabel('最大整体活性')
    plt.xticks(range(len(variants)), variants, rotation=45)
    plt.ylim(0, max(performance_df['Max_Activity']) * 1.2)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/max_activity_comparison.png', dpi=300)
    
    # 绘制各组分活性堆叠柱状图
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(variants))
    width = 0.2
    
    plt.bar(
        x - width, 
        performance_df['TA_Activity'], 
        width, 
        label='转氨酶活性', 
        color='royalblue'
    )
    plt.bar(
        x, 
        performance_df['LDH_Activity'], 
        width, 
        label='LDH活性', 
        color='forestgreen'
    )
    plt.bar(
        x + width, 
        performance_df['GDH_Activity'], 
        width, 
        label='GDH活性', 
        color='firebrick'
    )
    
    plt.title('多酶级联反应中各酶在最佳pH下的活性')
    plt.xlabel('转氨酶变体')
    plt.ylabel('相对活性')
    plt.xticks(x, variants, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/enzyme_activities_comparison.png', dpi=300)
    
    # 输出分析结果
    print("\n各变体在多酶级联反应中的表现:")
    for _, row in performance_df.iterrows():
        print(f"变体: {row['Variant']}")
        print(f"  最佳pH: {row['Best_pH']:.2f}")
        print(f"  最大整体活性: {row['Max_Activity']:.3f}")
        print(f"  在最佳pH下的各酶活性:")
        print(f"    - 转氨酶: {row['TA_Activity']:.3f}")
        print(f"    - LDH: {row['LDH_Activity']:.3f}")
        print(f"    - GDH: {row['GDH_Activity']:.3f}")
    
    # 找出整体活性最高的变体
    best_idx = performance_df['Max_Activity'].idxmax()
    best_variant = performance_df.loc[best_idx]
    
    print(f"\n多酶级联反应中表现最佳的变体: {best_variant['Variant']}")
    print(f"  最佳pH: {best_variant['Best_pH']:.2f}")
    print(f"  最大整体活性: {best_variant['Max_Activity']:.3f}")
    print(f"  相对于野生型的提升: {best_variant['Max_Activity'] / performance_df[performance_df['Variant'] == '3FCR-3M']['Max_Activity'].values[0]:.2f} 倍")
    
    print("\n多酶级联反应优化工作流程完成！结果保存在 'examples_results/cascade_optimization' 目录中")
    
    return performance_df

if __name__ == "__main__":
    # 运行多酶级联反应优化工作流程
    optimize_cascade_reaction() 