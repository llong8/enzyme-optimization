#!/usr/bin/env python
# coding: utf-8

# # 转氨酶在中性pH下的活性优化分析
# 
# 本笔记本展示了如何使用机器学习方法优化转氨酶在中性pH条件下的催化活性。我们将分析不同变体的活性数据，训练模型，并设计新的高活性变体。

# 导入必要的库
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置样式
plt.style.use('seaborn')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 导入自定义模块
sys.path.append(str(Path.cwd().parent))
from src.models.ml_model import EnzymeActivityPredictor
from src.utils.feature_engineering import EnzymeFeatureExtractor
from src.experiments.activity_assay import EnzymeActivityAssay

# ## 1. 模拟酶活性实验数据
# 
# 首先，我们将模拟一系列转氨酶变体在不同pH条件下的催化活性数据。

# 创建结果目录
os.makedirs('results/data', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# 创建实验对象
assay = EnzymeActivityAssay(output_dir='results/data')

# 定义要测试的变体和底物
variants = [
    '3FCR-3M',           # 基础模板
    '3FCR-3M-F168Y',     # 单突变体，高活性
    '3FCR-3M-L58M',      # 单突变体，平缓的pH曲线
    '3FCR-3M-W59C',      # 单突变体，pH曲线平缓
    '3FCR-3M-F168Y-W59C', # 双突变体，在中性pH下活性更高
    '3FCR-3M-L58M-F168Y'  # 双突变体，高活性
]

substrates = ['1a', '1f']

# 测量不同pH条件下的活性
results = assay.measure_activity(
    enzyme_variants=variants,
    substrates=substrates,
    ph_range=(6.5, 9.0, 0.5)
)

# 显示结果数据框
print(results.head())

# ## 2. 可视化pH-活性曲线
# 
# 让我们可视化不同变体的pH-活性曲线，以了解它们的pH依赖性。

# 绘制pH-活性曲线
ph_profile_fig = assay.plot_ph_activity_profile(results, output_file='results/plots/ph_activity_profile.png')
plt.show()

# 比较在pH 7.5下的活性
comparison_fig = assay.compare_variants_at_ph(results, ph_value=7.5, output_file='results/plots/variant_comparison_pH7.5.png')
plt.show()

# ## 3. 特征工程
# 
# 接下来，我们将从变体序列中提取特征，以训练机器学习模型。

# 定义关键位点
key_positions = [19, 58, 59, 87, 152, 165, 167, 168, 231, 261, 420]

# 定义野生型序列（仅关键残基）
wild_type_key_residues = {
    19: 'S', 58: 'L', 59: 'W', 87: 'F', 152: 'Y',
    165: 'L', 167: 'L', 168: 'F', 231: 'A', 261: 'V', 420: 'R'
}

# 解析变体名称并提取突变信息
variants_data = []
variant_names = []

for variant in results['Variant'].unique():
    mutations = []
    
    # 跳过基础模板
    if variant == '3FCR-3M':
        pass
    # 解析单突变体名称
    elif variant.startswith('3FCR-3M-'):
        mutation_code = variant.split('3FCR-3M-')[1]
        if len(mutation_code) == 4:  # 例如 L58M
            wt_aa = mutation_code[0]
            pos = int(mutation_code[1:3])
            mut_aa = mutation_code[3]
            mutations.append((pos, wt_aa, mut_aa))
        elif '-' in mutation_code:  # 例如 L58M-F168Y
            for single_mutation in mutation_code.split('-'):
                wt_aa = single_mutation[0]
                pos = int(single_mutation[1:3])
                mut_aa = single_mutation[3]
                mutations.append((pos, wt_aa, mut_aa))
    
    # 创建一个包含所有氨基酸的假序列
    dummy_sequence = ''.join([wild_type_key_residues[pos] for pos in sorted(wild_type_key_residues.keys())])
    
    # 提取特征
    if not mutations:  # 基础模板
        features = []
        for pos in key_positions:
            aa = wild_type_key_residues[pos]
            features.extend(EnzymeFeatureExtractor.get_aa_descriptors(aa))
    else:
        features = EnzymeFeatureExtractor.extract_mutation_features(dummy_sequence, mutations, key_positions)
    
    variants_data.append(features)
    variant_names.append(variant)

# 底物特征（这里使用简化的描述符）
substrate_features = {
    '1a': [100, 120, 0],  # 三个描述符代表底物的立体属性
    '1f': [95, 110, 10]   # 不同底物的特征
}

# 构建特征矩阵
X_list = []
y_list = []
metadata = []

for i, variant_feature in enumerate(variants_data):
    variant_name = variant_names[i]
    
    for substrate in results['Substrate'].unique():
        substrate_feature = substrate_features[substrate]
        
        for _, row in results[(results['Variant'] == variant_name) & 
                           (results['Substrate'] == substrate)].iterrows():
            ph = row['pH']
            activity = row['Activity_Mean']
            
            # 特征向量：变体特征 + 底物特征 + pH
            X_list.append(variant_feature + substrate_feature + [ph])
            
            # 目标值：活性的自然对数
            y_list.append(np.log(max(0.01, activity)))
            
            # 元数据
            metadata.append({
                'Variant': variant_name,
                'Substrate': substrate,
                'pH': ph,
                'Activity': activity
            })

X = np.array(X_list)
y = np.array(y_list)
metadata_df = pd.DataFrame(metadata)

# 生成特征名称
feature_names = EnzymeFeatureExtractor.generate_feature_names(key_positions, 3)

print(f"特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# ## 4. 训练机器学习模型
# 
# 现在，我们将训练机器学习模型来预测酶变体的活性。

# 创建并训练GBRT模型
model = EnzymeActivityPredictor(model_type='GBRT')
performance = model.train(X, y, feature_names=feature_names, optimize_hyperparams=True)

# 显示模型性能
print("GBRT模型性能:")
for metric, value in performance.items():
    print(f"  {metric}: {value:.4f}")

# 可视化特征重要性
importance_df = model.feature_importance()
plt.figure(figsize=(10, 8))
sns.barplot(x='重要性', y='特征', data=importance_df.head(15))
plt.title('GBRT模型的特征重要性')
plt.tight_layout()
plt.savefig('results/plots/feature_importance.png', dpi=300)
plt.show()

# 可视化预测值与真实值的对比
y_pred = model.predict(X)
plt.figure(figsize=(10, 8))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('真实活性(log)')
plt.ylabel('预测活性(log)')
plt.title('模型预测结果对比')
plt.grid(True)
plt.savefig('results/plots/model_prediction.png', dpi=300)
plt.show()

# ## 5. 使用模型设计新变体
# 
# 最后，我们将使用训练好的模型设计在中性pH下活性更高的新变体。

# 定义可能的突变
allowed_mutations = {
    19: ['S', 'A', 'T', 'C'],
    58: ['L', 'M', 'I', 'V'],
    59: ['W', 'F', 'Y', 'C'],
    87: ['F', 'Y', 'W', 'L'],
    152: ['Y', 'F', 'W', 'H'],
    165: ['L', 'I', 'V', 'M'],
    167: ['L', 'I', 'V', 'M'],
    168: ['F', 'Y', 'W', 'L'],
    231: ['A', 'G', 'S', 'T'],
    261: ['V', 'I', 'L', 'M'],
    420: ['R', 'K', 'H', 'Q']
}

# 定义模板变体
template_variants = ['3FCR-3M-F168Y', '3FCR-3M-L58M']

# 定义目标pH和底物
target_ph = 7.5
target_substrate = '1a'
substrate_feature = substrate_features[target_substrate]

# 预测结果
predictions = []

# 对每个模板变体生成可能的新变体
for template_name in template_variants:
    # 获取模板变体的索引
    template_idx = variant_names.index(template_name)
    template_feature = variants_data[template_idx]
    
    # 从模板名称中提取已有的突变
    existing_mutations = []
    mutation_code = template_name.split('3FCR-3M-')[1]
    if '-' in mutation_code:
        for single_mutation in mutation_code.split('-'):
            pos = int(single_mutation[1:3])
            existing_mutations.append(pos)
    else:
        pos = int(mutation_code[1:3])
        existing_mutations.append(pos)
    
    # 对每个未突变的位点尝试所有可能的突变
    for pos in key_positions:
        if pos in existing_mutations:
            continue
            
        # 计算位点在特征向量中的索引
        pos_idx = key_positions.index(pos) * 3
        
        # 尝试每种可能的突变
        for mut_aa in allowed_mutations[pos]:
            # 跳过野生型残基
            if mut_aa == wild_type_key_residues[pos]:
                continue
            
            # 复制模板特征
            new_feature = template_feature.copy()
            
            # 替换该位点的描述符
            new_feature[pos_idx:pos_idx+3] = EnzymeFeatureExtractor.get_aa_descriptors(mut_aa)
            
            # 构建完整特征向量
            full_feature = new_feature + substrate_feature + [target_ph]
            
            # 预测活性
            predicted_activity = np.exp(model.predict([full_feature])[0])
            
            # 生成变体名称
            new_variant = f'{template_name}-{wild_type_key_residues[pos]}{pos}{mut_aa}'
            
            # 记录预测结果
            predictions.append({
                'Variant': new_variant,
                'Template': template_name,
                'Mutation_Position': pos,
                'Wild_AA': wild_type_key_residues[pos],
                'Mutant_AA': mut_aa,
                'Predicted_Activity': predicted_activity,
            })

# 转换为DataFrame并排序
predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.sort_values('Predicted_Activity', ascending=False)

# 显示预测活性最高的10个变体
print(predictions_df.head(10))

# 可视化预测的顶级变体
plt.figure(figsize=(14, 8))
top_predictions = predictions_df.head(15)
sns.barplot(x='Variant', y='Predicted_Activity', hue='Template', data=top_predictions)
plt.title(f'预测活性最高的15个变体 (pH {target_ph}, 底物 {target_substrate})')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('results/plots/top_variant_predictions.png', dpi=300)
plt.show()

# ## 6. 保存模型
# 
# 将训练好的模型保存到磁盘，以便后续使用。

# 保存模型
model.save_model('results/models/GBRT_model.pkl')
print("模型已保存到 'results/models/GBRT_model.pkl'")

# ## 总结
# 
# 在本笔记本中，我们：
# 
# 1. 模拟了不同转氨酶变体在各种pH条件下的催化活性
# 2. 可视化了pH-活性曲线，了解了变体的pH依赖性
# 3. 从变体序列和突变中提取了特征
# 4. 训练了GBRT模型来预测活性
# 5. 使用模型设计了在中性pH下活性更高的新变体
# 
# 这些结果显示，机器学习可以有效地帮助优化酶的性能，特别是在调整pH依赖性方面。 