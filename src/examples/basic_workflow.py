"""
基本工作流程示例脚本

这个脚本演示了如何使用本项目实现的工具来优化转氨酶的pH适应性。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.activity_assay import EnzymeActivityAssay
import matplotlib.pyplot as plt

def run_basic_workflow():
    """运行基本工作流程"""
    print("=== 转氨酶优化基本工作流程 ===")
    
    # 创建结果目录
    os.makedirs('examples_results', exist_ok=True)
    
    # 步骤1：模拟活性测定实验
    print("\n1. 模拟酶活性测定实验")
    assay = EnzymeActivityAssay(output_dir='examples_results')
    
    # 定义要测试的变体和底物
    variants = [
        '3FCR-3M',           # 基础模板
        '3FCR-3M-F168Y',     # 单突变体，高活性
        '3FCR-3M-L58M',      # 单突变体，平缓的pH曲线
        '3FCR-3M-F168Y-W59C' # 双突变体，在中性pH下活性更高
    ]
    
    substrates = ['1a', '1f']
    
    # 测量不同pH条件下的活性
    results = assay.measure_activity(
        enzyme_variants=variants,
        substrates=substrates,
        ph_range=(6.5, 9.0, 0.5)
    )
    
    # 步骤2：可视化结果
    print("\n2. 可视化实验结果")
    
    # 绘制pH-活性曲线
    ph_profile_fig = assay.plot_ph_activity_profile(
        results,
        output_file='examples_results/ph_activity_profile.png'
    )
    
    # 比较在pH 7.5下的活性
    comparison_fig = assay.compare_variants_at_ph(
        results,
        ph_value=7.5,
        output_file='examples_results/variant_comparison_pH7.5.png'
    )
    
    # 步骤3：分析最佳变体
    print("\n3. 分析最佳变体")
    
    # 筛选pH 7.5下的数据
    ph_diff = abs(results['pH'] - 7.5)
    min_diff = ph_diff.min()
    ph7_5_data = results[ph_diff <= min_diff + 1e-6]
    
    # 对每个底物找出最佳变体
    for substrate in substrates:
        substrate_data = ph7_5_data[ph7_5_data['Substrate'] == substrate]
        best_variant_idx = substrate_data['Activity_Mean'].idxmax()
        best_variant = substrate_data.loc[best_variant_idx]
        
        print(f"底物 {substrate} 在 pH {best_variant['pH']} 下的最佳变体:")
        print(f"  变体: {best_variant['Variant']}")
        print(f"  活性: {best_variant['Activity_Mean']:.2f} ± {best_variant['Activity_Std']:.2f} U/mg")
        print(f"  相对于野生型的提升: {best_variant['Activity_Mean'] / ph7_5_data[(ph7_5_data['Substrate'] == substrate) & (ph7_5_data['Variant'] == '3FCR-3M')]['Activity_Mean'].values[0]:.2f} 倍")
    
    print("\n工作流程完成！结果保存在 'examples_results' 目录中")
    
    return results

if __name__ == "__main__":
    run_basic_workflow() 