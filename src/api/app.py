import os
import sys
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
import pickle

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.ml_model import EnzymeActivityPredictor
from src.utils.feature_engineering import EnzymeFeatureExtractor

app = Flask(__name__)

# 定义关键位点
KEY_POSITIONS = [19, 58, 59, 87, 152, 165, 167, 168, 231, 261, 420]

# 定义野生型序列（仅关键残基）
WILD_TYPE_KEY_RESIDUES = {
    19: 'S', 58: 'L', 59: 'W', 87: 'F', 152: 'Y',
    165: 'L', 167: 'L', 168: 'F', 231: 'A', 261: 'V', 420: 'R'
}

# 可能的突变
ALLOWED_MUTATIONS = {
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

# 底物特征
SUBSTRATE_FEATURES = {
    '1a': [100, 120, 0],
    '1f': [95, 110, 10]
}

def load_model():
    """加载预训练模型"""
    model_path = 'results/models/GBRT_model.pkl'
    
    # 如果模型文件不存在，返回None
    if not os.path.exists(model_path):
        return None
    
    try:
        model = EnzymeActivityPredictor.load_model(model_path)
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'ok', 'message': '酶工程API服务正常运行中'})

@app.route('/api/predict', methods=['POST'])
def predict_activity():
    """预测酶变体的活性"""
    data = request.json
    
    if not data:
        return jsonify({'error': '无效的请求数据'}), 400
    
    # 获取请求参数
    variant = data.get('variant')
    substrate = data.get('substrate', '1a')
    ph = data.get('ph', 7.5)
    
    if not variant:
        return jsonify({'error': '缺少变体信息'}), 400
    
    if substrate not in SUBSTRATE_FEATURES:
        return jsonify({'error': f'不支持的底物: {substrate}'}), 400
    
    if not 5.0 <= ph <= 10.0:
        return jsonify({'error': 'pH值必须在5.0-10.0范围内'}), 400
    
    # 加载模型
    model = load_model()
    if model is None:
        return jsonify({'error': '模型加载失败'}), 500
    
    try:
        # 解析变体名称并提取特征
        features = extract_variant_features(variant)
        
        # 构建完整特征向量
        substrate_feature = SUBSTRATE_FEATURES[substrate]
        full_feature = features + substrate_feature + [ph]
        
        # 预测活性
        predicted_activity = np.exp(model.predict([full_feature])[0])
        
        # 返回结果
        return jsonify({
            'variant': variant,
            'substrate': substrate,
            'ph': ph,
            'predicted_activity': float(predicted_activity)
        })
    
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/variants/suggest', methods=['POST'])
def suggest_variants():
    """根据底物和pH条件推荐高活性变体"""
    data = request.json
    
    if not data:
        return jsonify({'error': '无效的请求数据'}), 400
    
    # 获取请求参数
    template = data.get('template', '3FCR-3M')
    substrate = data.get('substrate', '1a')
    ph = data.get('ph', 7.5)
    top_n = data.get('top_n', 5)
    
    if substrate not in SUBSTRATE_FEATURES:
        return jsonify({'error': f'不支持的底物: {substrate}'}), 400
    
    if not 5.0 <= ph <= 10.0:
        return jsonify({'error': 'pH值必须在5.0-10.0范围内'}), 400
    
    # 加载模型
    model = load_model()
    if model is None:
        return jsonify({'error': '模型加载失败'}), 500
    
    try:
        # 生成候选变体并预测活性
        candidates = []
        template_features = extract_variant_features(template)
        substrate_feature = SUBSTRATE_FEATURES[substrate]
        
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
        
        # 对每个关键位点生成候选变体
        for pos in KEY_POSITIONS:
            if pos in existing_mutations:
                continue
                
            # 计算位点在特征向量中的索引
            pos_idx = KEY_POSITIONS.index(pos) * 3
            
            # 尝试每种可能的突变
            for mut_aa in ALLOWED_MUTATIONS[pos]:
                # 跳过野生型残基
                if mut_aa == WILD_TYPE_KEY_RESIDUES[pos]:
                    continue
                
                # 复制模板特征
                new_feature = template_features.copy()
                
                # 替换该位点的描述符
                new_feature[pos_idx:pos_idx+3] = EnzymeFeatureExtractor.get_aa_descriptors(mut_aa)
                
                # 构建完整特征向量
                full_feature = new_feature + substrate_feature + [ph]
                
                # 预测活性
                predicted_activity = np.exp(model.predict([full_feature])[0])
                
                # 生成变体名称
                if template == '3FCR-3M':
                    new_variant = f'3FCR-3M-{WILD_TYPE_KEY_RESIDUES[pos]}{pos}{mut_aa}'
                else:
                    new_variant = f'{template}-{WILD_TYPE_KEY_RESIDUES[pos]}{pos}{mut_aa}'
                
                # 记录预测结果
                candidates.append({
                    'variant': new_variant,
                    'predicted_activity': float(predicted_activity)
                })
        
        # 按预测活性排序并返回前N个
        candidates.sort(key=lambda x: x['predicted_activity'], reverse=True)
        top_candidates = candidates[:top_n]
        
        return jsonify({
            'template': template,
            'substrate': substrate,
            'ph': ph,
            'suggested_variants': top_candidates
        })
    
    except Exception as e:
        return jsonify({'error': f'推荐失败: {str(e)}'}), 500

def extract_variant_features(variant):
    """提取变体的特征向量"""
    mutations = []
    
    # 解析变体名称
    if variant == '3FCR-3M':
        pass  # 基础模板，没有突变
    elif variant.startswith('3FCR-3M-'):
        mutation_code = variant.split('3FCR-3M-')[1]
        if '-' in mutation_code:  # 双重或多重突变
            for single_mutation in mutation_code.split('-'):
                wt_aa = single_mutation[0]
                pos = int(single_mutation[1:3])
                mut_aa = single_mutation[3]
                mutations.append((pos, wt_aa, mut_aa))
        else:  # 单突变
            wt_aa = mutation_code[0]
            pos = int(mutation_code[1:3])
            mut_aa = mutation_code[3]
            mutations.append((pos, wt_aa, mut_aa))
    else:
        raise ValueError(f"无效的变体名称: {variant}")
    
    # 创建一个包含所有氨基酸的假序列
    dummy_sequence = ''.join([WILD_TYPE_KEY_RESIDUES[pos] for pos in sorted(WILD_TYPE_KEY_RESIDUES.keys())])
    
    # 提取特征
    if not mutations:  # 基础模板
        features = []
        for pos in KEY_POSITIONS:
            aa = WILD_TYPE_KEY_RESIDUES[pos]
            features.extend(EnzymeFeatureExtractor.get_aa_descriptors(aa))
        return features
    else:
        return EnzymeFeatureExtractor.extract_mutation_features(dummy_sequence, mutations, KEY_POSITIONS)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 