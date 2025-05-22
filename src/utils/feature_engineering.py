import numpy as np
import pandas as pd

# 氨基酸物理化学描述符（范德华体积、疏水性、等电点）
AA_DESCRIPTORS = {
    'A': [67, 1.8, 6.0],  # 丙氨酸
    'C': [86, 2.5, 5.07], # 半胱氨酸
    'D': [91, -3.5, 2.77], # 天冬氨酸
    'E': [109, -3.5, 3.22], # 谷氨酸
    'F': [135, 2.8, 5.48], # 苯丙氨酸
    'G': [48, -0.4, 5.97], # 甘氨酸
    'H': [118, -3.2, 7.59], # 组氨酸
    'I': [124, 4.5, 6.02], # 异亮氨酸
    'K': [135, -3.9, 9.74], # 赖氨酸
    'L': [124, 3.8, 5.98], # 亮氨酸
    'M': [124, 1.9, 5.74], # 甲硫氨酸
    'N': [96, -3.5, 5.41], # 天冬酰胺
    'P': [90, -1.6, 6.30], # 脯氨酸
    'Q': [114, -3.5, 5.65], # 谷氨酰胺
    'R': [148, -4.5, 10.76], # 精氨酸
    'S': [73, -0.8, 5.68], # 丝氨酸
    'T': [93, -0.7, 5.87], # 苏氨酸
    'V': [105, 4.2, 5.96], # 缬氨酸
    'W': [163, -0.9, 5.89], # 色氨酸
    'Y': [141, -1.3, 5.66], # 酪氨酸
}

class EnzymeFeatureExtractor:
    """
    提取酶变体特征的工具类
    """
    
    @staticmethod
    def get_aa_descriptors(aa):
        """
        获取氨基酸的描述符
        
        Parameters:
        -----------
        aa : str
            单字母氨基酸代码
        
        Returns:
        --------
        list
            氨基酸的物理化学描述符 [范德华体积, 疏水性, 等电点]
        """
        if aa not in AA_DESCRIPTORS:
            raise ValueError(f"未知的氨基酸: {aa}")
        
        return AA_DESCRIPTORS[aa]
    
    @staticmethod
    def extract_mutation_features(wild_type, mutations, key_positions):
        """
        从突变中提取特征
        
        Parameters:
        -----------
        wild_type : str
            野生型酶的序列
        mutations : list of tuple
            突变列表，每个突变为 (position, wild_aa, mutant_aa)
            例如：[(58, 'L', 'M'), (168, 'F', 'Y')]
        key_positions : list
            关键位点列表
        
        Returns:
        --------
        list
            酶变体的特征向量
        """
        features = []
        
        # 为每个关键位点提取特征
        for pos in key_positions:
            # 查找该位点是否有突变
            mutated = False
            for mut_pos, _, mut_aa in mutations:
                if mut_pos == pos:
                    features.extend(EnzymeFeatureExtractor.get_aa_descriptors(mut_aa))
                    mutated = True
                    break
            
            # 如果没有突变，使用野生型氨基酸的描述符
            if not mutated:
                wild_aa = wild_type[pos-1]  # 位置从1开始，但索引从0开始
                features.extend(EnzymeFeatureExtractor.get_aa_descriptors(wild_aa))
        
        return features
    
    @staticmethod
    def build_feature_matrix(variants, substrate_features, ph_values):
        """
        构建特征矩阵
        
        Parameters:
        -----------
        variants : list of list
            变体特征列表
        substrate_features : list
            底物特征
        ph_values : list
            pH值列表
        
        Returns:
        --------
        numpy.ndarray
            特征矩阵，每行包含一个变体-底物-pH组合的特征
        """
        feature_matrix = []
        
        for variant_features in variants:
            for substrate_feature in substrate_features:
                for ph in ph_values:
                    # 组合特征：变体特征 + 底物特征 + pH
                    combined_features = variant_features + substrate_feature + [ph]
                    feature_matrix.append(combined_features)
        
        return np.array(feature_matrix)
    
    @staticmethod
    def generate_feature_names(key_positions, num_substrate_features):
        """
        生成特征名称
        
        Parameters:
        -----------
        key_positions : list
            关键位点列表
        num_substrate_features : int
            底物特征数量
        
        Returns:
        --------
        list
            特征名称列表
        """
        feature_names = []
        
        # 酶特征名称
        descriptor_names = ['VDW', 'Hydrophobicity', 'pI']
        for pos in key_positions:
            for desc in descriptor_names:
                feature_names.append(f"Pos{pos}_{desc}")
        
        # 底物特征名称
        for i in range(num_substrate_features):
            feature_names.append(f"Substrate_feature{i+1}")
        
        # pH特征
        feature_names.append("pH")
        
        return feature_names 