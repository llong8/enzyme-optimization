"""
高级变体设计示例脚本

本脚本演示如何使用机器学习模型设计和优化转氨酶变体，以提高其在中性pH下的催化活性。
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.activity_assay import EnzymeActivityAssay
from src.utils.feature_engineering import extract_features_from_variant

def run_advanced_variant_design():
    """运行高级变体设计工作流程"""
    print("=== 转氨酶高级变体设计工作流程 ===")
    
    # 创建结果目录
    results_dir = 'examples_results/advanced_design'
    os.makedirs(results_dir, exist_ok=True)
    
    # 步骤1：生成初始训练数据
    print("\n1. 生成初始训练数据")
    assay = EnzymeActivityAssay(output_dir=results_dir)
    
    # 定义初始变体库
    initial_variants = [
        '3FCR-3M',               # 基础模板
        '3FCR-3M-F168Y',         # 单点突变
        '3FCR-3M-L58M',          # 单点突变
        '3FCR-3M-W59C',          # 单点突变
        '3FCR-3M-F168Y-W59C',    # 双突变
        '3FCR-3M-L58M-F168Y',    # 双突变
    ]
    
    # 定义目标底物和pH
    target_substrate = '1a'
    target_ph = 7.5
    
    # 测量初始变体库的活性
    initial_results = assay.measure_activity(
        enzyme_variants=initial_variants,
        substrates=[target_substrate],
        ph_range=(target_ph, target_ph, 0.5)  # 只测量目标pH
    )
    
    # 可视化初始变体库的活性
    plt.figure(figsize=(10, 6))
    sns_palette = plt.cm.viridis(np.linspace(0, 1, len(initial_variants)))
    
    for i, variant in enumerate(initial_variants):
        variant_data = initial_results[initial_results['Variant'] == variant]
        plt.bar(
            i,
            variant_data['Activity_Mean'].values[0],
            yerr=variant_data['Activity_Std'].values[0],
            color=sns_palette[i],
            label=variant
        )
    
    plt.title(f'初始变体库在pH {target_ph}下对底物{target_substrate}的活性')
    plt.xlabel('变体')
    plt.ylabel('活性 (U/mg)')
    plt.xticks(range(len(initial_variants)), initial_variants, rotation=45)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/initial_variants_activity.png', dpi=300)
    
    # 步骤2：提取特征和构建模型
    print("\n2. 提取特征和构建预测模型")
    
    # 准备训练数据
    X_data = []
    y_data = []
    
    for variant in initial_variants:
        # 提取特征
        features = extract_features_from_variant(variant)
        
        # 获取活性数据
        activity = initial_results[
            (initial_results['Variant'] == variant) & 
            (initial_results['Substrate'] == target_substrate)
        ]['Activity_Mean'].values[0]
        
        X_data.append(features)
        y_data.append(activity)
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"模型评估:")
    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  决定系数 (R²): {r2:.4f}")
    
    # 步骤3：设计新变体
    print("\n3. 设计并筛选新变体")
    
    # 定义潜在的突变位点
    mutation_sites = [58, 59, 60, 167, 168, 169]
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # 生成所有可能的双突变体
    candidate_variants = []
    
    for i, site1 in enumerate(mutation_sites):
        for j, site2 in enumerate(mutation_sites[i+1:], i+1):
            for aa1 in amino_acids:
                for aa2 in amino_acids:
                    # 跳过已有的变体
                    variant_name = f'3FCR-3M-L{site1}{aa1}-F{site2}{aa2}'
                    if variant_name not in initial_variants:
                        # 提取特征
                        features = extract_features_from_variant(variant_name)
                        # 预测活性
                        predicted_activity = model.predict([features])[0]
                        
                        candidate_variants.append({
                            'Variant': variant_name,
                            'Predicted_Activity': predicted_activity
                        })
    
    # 将候选变体转换为DataFrame并排序
    candidates_df = pd.DataFrame(candidate_variants)
    candidates_df = candidates_df.sort_values('Predicted_Activity', ascending=False)
    
    # 输出前10个预测活性最高的变体
    print("预测活性最高的前10个新变体:")
    for i, (_, row) in enumerate(candidates_df.head(10).iterrows()):
        print(f"  {i+1}. {row['Variant']}: {row['Predicted_Activity']:.2f} U/mg")
    
    # 保存候选变体列表
    candidates_df.to_csv(f'{results_dir}/candidate_variants.csv', index=False)
    
    # 步骤4：验证预测结果
    print("\n4. 验证预测结果")
    
    # 选择前3个预测活性最高的变体进行验证
    top_variants = list(candidates_df.head(3)['Variant'])
    validation_variants = initial_variants + top_variants
    
    # 测量验证变体的活性
    validation_results = assay.measure_activity(
        enzyme_variants=validation_variants,
        substrates=[target_substrate],
        ph_range=(target_ph, target_ph, 0.5)
    )
    
    # 可视化验证结果
    plt.figure(figsize=(12, 6))
    
    for i, variant in enumerate(validation_variants):
        variant_data = validation_results[validation_results['Variant'] == variant]
        color = 'royalblue' if variant in initial_variants else 'crimson'
        plt.bar(
            i,
            variant_data['Activity_Mean'].values[0],
            yerr=variant_data['Activity_Std'].values[0],
            color=color,
            label='初始变体' if i == 0 and variant in initial_variants else ('预测变体' if i == len(initial_variants) else '')
        )
    
    plt.title(f'初始变体和预测变体在pH {target_ph}下对底物{target_substrate}的活性比较')
    plt.xlabel('变体')
    plt.ylabel('活性 (U/mg)')
    plt.xticks(range(len(validation_variants)), validation_variants, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/validation_results.png', dpi=300)
    
    # 计算最佳变体相对于野生型的活性提升
    wt_activity = validation_results[validation_results['Variant'] == '3FCR-3M']['Activity_Mean'].values[0]
    best_variant_idx = validation_results['Activity_Mean'].idxmax()
    best_variant = validation_results.loc[best_variant_idx]
    
    print(f"验证结果:")
    print(f"  最佳变体: {best_variant['Variant']}")
    print(f"  活性: {best_variant['Activity_Mean']:.2f} ± {best_variant['Activity_Std']:.2f} U/mg")
    print(f"  相对于野生型的提升: {best_variant['Activity_Mean'] / wt_activity:.2f} 倍")
    
    print("\n高级变体设计工作流程完成！结果保存在 'examples_results/advanced_design' 目录中")
    
    return validation_results

if __name__ == "__main__":
    # 导入可视化库（只在主函数中导入以防止循环导入）
    import seaborn as sns
    # 运行高级变体设计工作流程
    run_advanced_variant_design() 