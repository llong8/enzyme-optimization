# 转氨酶pH依赖性分析说明

本文档说明如何使用`pH依赖性分析.py`脚本分析转氨酶变体的pH依赖性特征。

## 功能概述

`pH依赖性分析.py`实现了转氨酶变体pH依赖性的深入分析流程，包括以下步骤：
1. 测量不同变体在广泛pH范围内的活性数据
2. 拟合pH-活性曲线模型并提取关键参数
3. 分析不同变体的最适pH值和pH适应范围宽度
4. 计算中性pH区域的相对活性

## 使用方法

### 直接运行脚本

```bash
# 在本地环境中运行
python src/examples/pH依赖性分析.py

# 或在Docker容器中运行
docker exec -it enzyme-ml python src/examples/pH依赖性分析.py
```

### 在代码中调用

```python
from src.examples.pH依赖性分析 import analyze_ph_dependence

# 运行pH依赖性分析工作流程并获取结果
params_df = analyze_ph_dependence()

# 使用返回的参数进行进一步分析
print(params_df)
```

## 工作流程详解

### 1. 测量活性数据

脚本使用`EnzymeActivityAssay`类测量以下变体在不同pH条件下的活性：
- `3FCR-3M`：基础模板
- `3FCR-3M-F168Y`：单突变体，高活性
- `3FCR-3M-L58M`：单突变体，平缓的pH曲线
- `3FCR-3M-F168Y-W59C`：双突变体，在中性pH下活性更高

测量范围为pH 6.0到9.5，步长为0.2，以获得更精细的pH-活性曲线。

### 2. 拟合pH-活性曲线

脚本使用高斯模型拟合每个变体的pH-活性曲线：

```
活性 = a * exp(-(pH - pH_opt)² / (2 * b²)) + c
```

其中：
- `a`: 曲线高度参数
- `b`: 曲线宽度参数
- `c`: 基线活性参数
- `pH_opt`: 最适pH值

拟合结果以图表形式保存在`examples_results/ph_analysis/ph_activity_fitting.png`。

### 3. 分析pH依赖性参数

从拟合曲线中提取以下关键参数：
- 最适pH值：活性达到最大值的pH
- pH适应范围宽度：活性超过最大值一半的pH范围

这些参数以柱状图形式可视化，保存在：
- `examples_results/ph_analysis/optimal_ph_comparison.png`
- `examples_results/ph_analysis/ph_range_width_comparison.png`

### 4. 计算中性pH区域的相对活性

脚本定义中性pH区域（默认为7.0-7.8），并计算每个变体在此区域内的积分活性，以及相对于全pH范围的活性占比。结果以柱状图形式保存在`examples_results/ph_analysis/neutral_ph_activity_ratio.png`。

## 输出结果

脚本执行后会输出以下信息：
- 各变体的pH依赖性参数（最适pH、pH适应范围宽度、中性pH活性占比、最大活性）
- 中性pH区域活性最高的变体及其相对于野生型的提升倍数

所有参数数据保存在`examples_results/ph_analysis/ph_dependence_parameters.csv`文件中。

## 自定义工作流程

如果需要自定义工作流程，可以修改以下参数：

```python
# 自定义变体列表
variants = [
    '3FCR-3M',
    '3FCR-3M-F168Y',
    # 添加其他变体...
]

# 自定义底物
substrate = '1f'  # 改为底物1f

# 自定义pH测量范围和步长
results = assay.measure_activity(
    enzyme_variants=variants,
    substrates=[substrate],
    ph_range=(5.5, 10.0, 0.1)  # 更宽的范围，更小的步长
)

# 自定义中性pH区域定义
neutral_ph_min = 6.8
neutral_ph_max = 7.6
```

## 技术说明

### pH-活性模型

本脚本使用高斯模型拟合pH-活性关系，这是酶学研究中常用的模型。该模型能够很好地描述酶活性随pH变化的钟形曲线特征。拟合过程使用`scipy.optimize.curve_fit`函数实现，并设置了合理的参数范围约束。

### 参数解释

- **最适pH值(pH_opt)**：酶活性达到最大值的pH，反映了酶活性中心的质子化状态最适合催化反应的条件
- **pH适应范围宽度(ph_range_width)**：定义为2*b，其中b是高斯模型的宽度参数，反映了酶对pH变化的耐受性
- **中性pH活性占比(neutral_ratio)**：中性pH区域内的积分活性与全pH范围积分活性之比，反映了酶在中性条件下的表现

## 注意事项

- 该脚本使用模拟数据，实际应用中应替换为真实的实验测量数据
- 曲线拟合的初始参数和边界条件可能需要根据实际数据调整
- 中性pH区域的定义应根据具体应用场景来确定
- 为获得可靠的拟合结果，建议使用足够密集的pH梯度进行测量 