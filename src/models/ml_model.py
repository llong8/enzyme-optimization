import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os

class EnzymeActivityPredictor:
    """
    机器学习模型类，用于预测转氨酶在不同pH条件下的催化活性
    """
    
    def __init__(self, model_type='GBRT'):
        """
        初始化模型
        
        Parameters:
        -----------
        model_type : str
            模型类型，可选 'GBRT', 'RF', 'SVR', 'KRR'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
        # 初始化模型
        if model_type == 'GBRT':
            self.model = GradientBoostingRegressor(random_state=42)
        elif model_type == 'RF':
            self.model = RandomForestRegressor(random_state=42)
        elif model_type == 'SVR':
            self.model = SVR()
        elif model_type == 'KRR':
            self.model = KernelRidge()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def train(self, X, y, feature_names=None, test_size=0.2, cv=10, optimize_hyperparams=True):
        """
        训练模型
        
        Parameters:
        -----------
        X : array-like
            特征矩阵，包含酶描述符和pH值
        y : array-like
            目标值，催化活性的对数值
        feature_names : list, optional
            特征名称列表
        test_size : float, optional
            测试集比例
        cv : int, optional
            交叉验证折数
        optimize_hyperparams : bool, optional
            是否进行超参数优化
        
        Returns:
        --------
        dict
            模型性能指标
        """
        # 保存特征名称
        self.feature_names = feature_names
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 超参数优化
        if optimize_hyperparams:
            self._optimize_hyperparams(X_train, y_train, cv)
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 计算性能指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmsd = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmsd = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmsd': train_rmsd,
            'test_rmsd': test_rmsd,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    
    def _optimize_hyperparams(self, X, y, cv):
        """
        超参数优化
        
        Parameters:
        -----------
        X : array-like
            特征矩阵
        y : array-like
            目标值
        cv : int
            交叉验证折数
        """
        param_grid = None
        
        # 为不同模型定义超参数网格
        if self.model_type == 'GBRT':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'RF':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'SVR':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        elif self.model_type == 'KRR':
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': [0.01, 0.1, 1, 10]
            }
        
        if param_grid:
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            print(f"最佳超参数: {grid_search.best_params_}")
    
    def predict(self, X):
        """
        预测催化活性
        
        Parameters:
        -----------
        X : array-like
            特征矩阵
        
        Returns:
        --------
        array-like
            预测的催化活性对数值
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """
        保存模型
        
        Parameters:
        -----------
        filepath : str
            模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        加载模型
        
        Parameters:
        -----------
        filepath : str
            模型文件路径
        
        Returns:
        --------
        EnzymeActivityPredictor
            加载的模型实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model_instance = cls(model_type=data['model_type'])
        model_instance.model = data['model']
        model_instance.feature_names = data['feature_names']
        
        return model_instance
    
    def feature_importance(self):
        """
        获取特征重要性
        
        Returns:
        --------
        pd.DataFrame
            特征重要性数据框
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if self.model_type not in ['GBRT', 'RF']:
            raise ValueError(f"模型类型 {self.model_type} 不支持特征重要性分析")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_names = self.feature_names if self.feature_names else [f"特征 {i}" for i in range(len(importances))]
        
        return pd.DataFrame({
            '特征': [feature_names[i] for i in indices],
            '重要性': importances[indices]
        }) 