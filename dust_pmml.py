"""
@author:duke.du
@time:2018/11/29 19:19
@file:dust_pmml.py
@contact: duke_du123@163.com
@function:训练模型，生成PMML文件
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import PMMLPipeline, sklearn2pmml  # No problem

iris = load_iris()

# 创建带有特征名称的 DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("iris' size :", iris_df.shape)

# 创建模型管道
iris_pipeline = PMMLPipeline([
    ("classifier", RandomForestClassifier())
])

# 训练模型
iris_pipeline.fit(iris_df, iris.target)

# 导出模型到 RandomForestClassifier_Iris.pmml 文件
sklearn2pmml(iris_pipeline, "pmml/RandomForestClassifier_Iris.pmml")
