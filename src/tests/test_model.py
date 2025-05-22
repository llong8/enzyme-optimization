import sys
import os
import unittest
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.ml_model import EnzymeActivityPredictor
from src.utils.feature_engineering import EnzymeFeatureExtractor

class TestEnzymeActivityPredictor(unittest.TestCase):
    """测试机器学习模型的单元测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个简单的测试数据集
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 37  # 11个关键位点 * 3个描述符 + 3个底物描述符 + 1个pH值
        
        # 创建随机特征矩阵
        self.X = np.random.rand(self.n_samples, self.n_features)
        
        # 创建目标变量 (活性的对数值)
        self.y = 2.0 + 0.5 * np.sum(self.X[:, :5], axis=1) - 0.3 * np.sum(self.X[:, 5:10], axis=1) + 0.1 * np.random.randn(self.n_samples)
        
        # 特征名称
        self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]
    
    def test_model_initialization(self):
        """测试模型初始化"""
        for model_type in ['GBRT', 'RF', 'SVR', 'KRR']:
            model = EnzymeActivityPredictor(model_type=model_type)
            self.assertEqual(model.model_type, model_type)
            self.assertIsNone(model.feature_names)
    
    def test_model_training(self):
        """测试模型训练"""
        model = EnzymeActivityPredictor(model_type='GBRT')
        performance = model.train(
            self.X, self.y, 
            feature_names=self.feature_names,
            test_size=0.2,
            cv=3,
            optimize_hyperparams=False
        )
        
        # 检查性能指标
        self.assertIn('train_r2', performance)
        self.assertIn('test_r2', performance)
        self.assertIn('train_rmsd', performance)
        self.assertIn('test_rmsd', performance)
        self.assertIn('cv_r2_mean', performance)
        self.assertIn('cv_r2_std', performance)
        
        # 检查R²值是否在合理范围内
        self.assertGreater(performance['train_r2'], 0.5)
        self.assertLess(performance['train_r2'], 1.0)
    
    def test_model_prediction(self):
        """测试模型预测"""
        model = EnzymeActivityPredictor(model_type='GBRT')
        model.train(self.X, self.y, optimize_hyperparams=False)
        
        # 预测训练集
        predictions = model.predict(self.X)
        
        # 检查预测结果的形状
        self.assertEqual(len(predictions), self.n_samples)
        
        # 检查新样本的预测
        new_sample = np.random.rand(1, self.n_features)
        new_prediction = model.predict(new_sample)
        self.assertEqual(len(new_prediction), 1)
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        model = EnzymeActivityPredictor(model_type='RF')
        model.train(self.X, self.y, feature_names=self.feature_names, optimize_hyperparams=False)
        
        # 创建临时目录
        os.makedirs('temp', exist_ok=True)
        model_path = 'temp/test_model.pkl'
        
        # 保存模型
        model.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # 加载模型
        loaded_model = EnzymeActivityPredictor.load_model(model_path)
        self.assertEqual(loaded_model.model_type, 'RF')
        self.assertEqual(loaded_model.feature_names, self.feature_names)
        
        # 检查预测结果是否一致
        original_predictions = model.predict(self.X)
        loaded_predictions = loaded_model.predict(self.X)
        np.testing.assert_almost_equal(original_predictions, loaded_predictions)
        
        # 清理
        os.remove(model_path)
    
    def test_feature_importance(self):
        """测试特征重要性计算"""
        for model_type in ['GBRT', 'RF']:
            model = EnzymeActivityPredictor(model_type=model_type)
            model.train(self.X, self.y, feature_names=self.feature_names, optimize_hyperparams=False)
            
            # 获取特征重要性
            importance_df = model.feature_importance()
            
            # 检查数据框结构
            self.assertEqual(len(importance_df), self.n_features)
            self.assertIn('特征', importance_df.columns)
            self.assertIn('重要性', importance_df.columns)
            
            # 检查重要性值之和是否接近1
            self.assertAlmostEqual(importance_df['重要性'].sum(), 1.0, places=5)

class TestEnzymeFeatureExtractor(unittest.TestCase):
    """测试特征提取工具的单元测试类"""
    
    def test_aa_descriptors(self):
        """测试氨基酸描述符获取"""
        # 测试几个氨基酸
        for aa in ['A', 'C', 'D', 'L', 'Y']:
            descriptors = EnzymeFeatureExtractor.get_aa_descriptors(aa)
            self.assertEqual(len(descriptors), 3)  # 3个描述符
        
        # 测试未知氨基酸
        with self.assertRaises(ValueError):
            EnzymeFeatureExtractor.get_aa_descriptors('B')  # B不是标准氨基酸
    
    def test_extract_mutation_features(self):
        """测试突变特征提取"""
        # 模拟野生型序列
        wild_type = "ACDEFGHIKLMNPQRSTVWY"
        
        # 定义关键位点 (1-indexed)
        key_positions = [1, 5, 10, 15, 20]
        
        # 定义突变
        mutations = [(1, 'A', 'G'), (10, 'K', 'R')]
        
        # 提取特征
        features = EnzymeFeatureExtractor.extract_mutation_features(wild_type, mutations, key_positions)
        
        # 检查特征长度
        self.assertEqual(len(features), 15)  # 5个位点 * 3个描述符
    
    def test_build_feature_matrix(self):
        """测试特征矩阵构建"""
        # 模拟变体特征
        variants = [
            [1, 2, 3, 4, 5, 6],  # 变体1的特征
            [7, 8, 9, 10, 11, 12]  # 变体2的特征
        ]
        
        # 模拟底物特征
        substrate_features = [
            [100, 101, 102],  # 底物1的特征
            [200, 201, 202]   # 底物2的特征
        ]
        
        # pH值
        ph_values = [6.5, 7.0]
        
        # 构建特征矩阵
        feature_matrix = EnzymeFeatureExtractor.build_feature_matrix(variants, substrate_features, ph_values)
        
        # 检查矩阵维度
        self.assertEqual(feature_matrix.shape, (8, 10))  # 2个变体 * 2个底物 * 2个pH = 8行，6+3+1=10列

if __name__ == "__main__":
    unittest.main() 