import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# 导入自定义模块
from models.ml_model import EnzymeActivityPredictor
from utils.feature_engineering import EnzymeFeatureExtractor
from experiments.activity_assay import EnzymeActivityAssay

def create_results_dirs():
    """创建结果目录"""
    results_dir = Path('results')
    model_dir = results_dir / 'models'
    data_dir = results_dir / 'data'
    plots_dir = results_dir / 'plots'
    
    for dir_path in [results_dir, model_dir, data_dir, plots_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    return results_dir, model_dir, data_dir, plots_dir

def simulate_initial_experiment():
    """模拟初始实验：测试单突变体在不同pH下的活性"""
    print("开始初始实验：测试单突变体...")
    
    # 创建实验对象
    assay = EnzymeActivityAssay(output_dir='results/data')
    
    # 定义变体和底物
    variants = ['3FCR-3M', '3FCR-3M-F168Y', '3FCR-3M-L58M', '3FCR-3M-W59C']
    substrates = ['1a']
    
    # 测量活性
    results = assay.measure_activity(variants, substrates, ph_range=(6.5, 9.0, 0.5))
    
    # 绘制pH-活性曲线
    assay.plot_ph_activity_profile(results, output_file='results/plots/initial_ph_activity.png')
    
    # 比较pH 7.5下的活性
    assay.compare_variants_at_ph(results, 7.5, output_file='results/plots/initial_comparison_pH7.5.png')
    
    return results

def prepare_training_data(results_df):
    """从实验结果准备机器学习训练数据"""
    print("准备机器学习训练数据...")
    
    # 定义关键位点
    key_positions = [19, 58, 59, 87, 152, 165, 167, 168, 231, 261, 420]
    
    # 定义野生型序列（仅关键残基）
    # 在实际应用中，这应该从PDB或其他来源获取
    wild_type_key_residues = {
        19: 'S', 58: 'L', 59: 'W', 87: 'F', 152: 'Y',
        165: 'L', 167: 'L', 168: 'F', 231: 'A', 261: 'V', 420: 'R'
    }
    
    # 解析变体名称并提取突变信息
    variants_data = []
    variant_names = []
    
    for variant in results_df['Variant'].unique():
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
        
        # 提取特征
        features = []
        if not mutations:  # 基础模板
            # 使用野生型残基的描述符
            for pos in key_positions:
                aa = wild_type_key_residues[pos]
                features.extend(EnzymeFeatureExtractor.get_aa_descriptors(aa))
        else:
            # 创建一个包含所有氨基酸的假序列（实际应用中应使用真实序列）
            dummy_sequence = ''.join([wild_type_key_residues[pos] for pos in sorted(wild_type_key_residues.keys())])
            features = EnzymeFeatureExtractor.extract_mutation_features(dummy_sequence, mutations, key_positions)
        
        variants_data.append(features)
        variant_names.append(variant)
    
    # 底物特征（这里使用简化的描述符）
    substrate_features = {
        '1a': [100, 120, 0]  # 三个描述符代表底物的立体属性
    }
    
    # 构建特征矩阵
    X_list = []
    y_list = []
    
    for i, variant_feature in enumerate(variants_data):
        variant_name = variant_names[i]
        
        for substrate in results_df['Substrate'].unique():
            substrate_feature = substrate_features[substrate]
            
            for _, row in results_df[(results_df['Variant'] == variant_name) & 
                                   (results_df['Substrate'] == substrate)].iterrows():
                ph = row['pH']
                activity = row['Activity_Mean']
                
                # 特征向量：变体特征 + 底物特征 + pH
                X_list.append(variant_feature + substrate_feature + [ph])
                
                # 目标值：活性的自然对数
                y_list.append(np.log(max(0.01, activity)))
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # 生成特征名称
    feature_names = EnzymeFeatureExtractor.generate_feature_names(key_positions, len(substrate_features['1a']))
    
    return X, y, feature_names

def train_and_evaluate_models(X, y, feature_names):
    """训练和评估不同的机器学习模型"""
    print("训练和评估机器学习模型...")
    
    # 定义要测试的模型类型
    model_types = ['GBRT', 'RF', 'SVR', 'KRR']
    
    best_model = None
    best_performance = {'test_r2': -float('inf')}
    model_results = {}
    
    for model_type in model_types:
        print(f"训练 {model_type} 模型...")
        
        # 创建模型
        model = EnzymeActivityPredictor(model_type=model_type)
        
        # 训练模型
        performance = model.train(X, y, feature_names=feature_names)
        
        print(f"{model_type} 模型性能:")
        print(f"  训练集 R²: {performance['train_r2']:.4f}")
        print(f"  测试集 R²: {performance['test_r2']:.4f}")
        print(f"  训练集 RMSD: {performance['train_rmsd']:.4f}")
        print(f"  测试集 RMSD: {performance['test_rmsd']:.4f}")
        print(f"  交叉验证 R²: {performance['cv_r2_mean']:.4f} ± {performance['cv_r2_std']:.4f}")
        
        # 保存模型
        model.save_model(f'results/models/{model_type}_model.pkl')
        
        # 记录结果
        model_results[model_type] = performance
        
        # 更新最佳模型
        if performance['test_r2'] > best_performance['test_r2']:
            best_model = model
            best_performance = performance
    
    # 保存模型比较结果
    with open('results/models/model_comparison.json', 'w') as f:
        json.dump(model_results, f, indent=2)
    
    # 为最佳模型绘制特征重要性
    if best_model.model_type in ['GBRT', 'RF']:
        importance_df = best_model.feature_importance()
        plt.figure(figsize=(10, 8))
        sns.barplot(x='重要性', y='特征', data=importance_df.head(15))
        plt.title(f'最佳模型({best_model.model_type})的特征重要性')
        plt.tight_layout()
        plt.savefig('results/plots/feature_importance.png', dpi=300)
    
    return best_model

def design_new_variants(model, X, y, feature_names):
    """使用机器学习模型设计新的变体"""
    print("使用机器学习模型设计新变体...")
    
    # 定义关键位点
    key_positions = [19, 58, 59, 87, 152, 165, 167, 168, 231, 261, 420]
    
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
    
    # 解析基础变体并提取特征
    base_features = {}
    
    for variant in template_variants:
        # 提取基础变体的特征向量
        variant_idx = np.where(np.array([row[0] for row in X]) == template_variants.index(variant))[0][0]
        base_feature = X[variant_idx][:-4]  # 移除底物特征和pH
        base_features[variant] = base_feature
    
    # 底物特征
    substrate_feature = [100, 120, 0]  # 为底物1a简化的描述符
    
    # 预测结果
    predictions = []
    
    # 对每个模板变体生成可能的新变体
    for template in template_variants:
        base_feature = base_features[template]
        
        # 从模板名称中提取已有的突变
        existing_mutations = []
        if template != '3FCR-3M':
            mutation_code = template.split('3FCR-3M-')[1]
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
                # 复制基础特征
                new_feature = base_feature.copy()
                
                # 替换该位点的描述符
                new_feature[pos_idx:pos_idx+3] = EnzymeFeatureExtractor.get_aa_descriptors(mut_aa)
                
                # 构建完整特征向量
                full_feature = list(new_feature) + substrate_feature + [target_ph]
                
                # 预测活性
                predicted_activity = np.exp(model.predict([full_feature])[0])
                
                # 生成变体名称
                if template == '3FCR-3M':
                    new_variant = f'3FCR-3M-{wild_type_key_residues[pos]}{pos}{mut_aa}'
                else:
                    new_variant = f'{template}-{wild_type_key_residues[pos]}{pos}{mut_aa}'
                
                # 记录预测结果
                predictions.append({
                    'Variant': new_variant,
                    'Predicted_Activity': predicted_activity,
                    'Template': template
                })
    
    # 转换为DataFrame并排序
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('Predicted_Activity', ascending=False)
    
    # 保存预测结果
    predictions_df.to_csv('results/data/variant_predictions.csv', index=False)
    
    # 绘制前20个预测结果
    plt.figure(figsize=(12, 8))
    top_predictions = predictions_df.head(20)
    sns.barplot(x='Variant', y='Predicted_Activity', hue='Template', data=top_predictions)
    plt.title(f'预测活性最高的20个变体 (pH {target_ph}, 底物 {target_substrate})')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('results/plots/top_variant_predictions.png', dpi=300)
    
    # 返回预测活性最高的前5个变体
    return predictions_df.head(5)

def validate_predictions(top_variants):
    """模拟验证预测的顶级变体"""
    print("模拟验证预测的顶级变体...")
    
    # 创建实验对象
    assay = EnzymeActivityAssay(output_dir='results/data')
    
    # 定义变体和底物
    variants = ['3FCR-3M'] + list(top_variants['Variant'])
    substrates = ['1a', '1f']
    
    # 测量活性
    results = assay.measure_activity(variants, substrates, ph_range=(7.0, 8.0, 0.5))
    
    # 绘制pH-活性曲线
    assay.plot_ph_activity_profile(results, output_file='results/plots/validation_ph_activity.png')
    
    # 比较pH 7.5下的活性
    assay.compare_variants_at_ph(results, 7.5, output_file='results/plots/validation_comparison_pH7.5.png')
    
    return results

def main():
    """主函数：运行完整工作流程"""
    print("=== 转氨酶优化工作流程 ===")
    
    # 创建结果目录
    results_dir, model_dir, data_dir, plots_dir = create_results_dirs()
    
    # 步骤1：模拟初始实验
    initial_results = simulate_initial_experiment()
    
    # 步骤2：准备训练数据
    X, y, feature_names = prepare_training_data(initial_results)
    
    # 步骤3：训练和评估模型
    best_model = train_and_evaluate_models(X, y, feature_names)
    
    # 步骤4：设计新变体
    top_variants = design_new_variants(best_model, X, y, feature_names)
    print("预测活性最高的变体:")
    print(top_variants)
    
    # 步骤5：验证预测
    validation_results = validate_predictions(top_variants)
    
    print("工作流程完成！结果保存在 'results' 目录中")

if __name__ == "__main__":
    # 记录起始时间
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    # 记录结束时间
    end_time = datetime.now()
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总运行时间: {end_time - start_time}") 