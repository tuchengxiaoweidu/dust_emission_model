"""
author: duke.du
datetime:2018/11/28 11:22
function:TPOT自动选择机器学习模型和参数--回归示例,为数据选择最合适的回归模型，为ElasticNetCV
"""

from tpot import TPOTRegressor
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

df = pd.DataFrame(pd.read_excel('data/point1234.xlsx', header=0))

training_data_input = df.values[:, 8:13].astype(float)
training_data_output = df.values[:, 7:8].astype(float)  # .dtype打印出数据类型，.ravel() 返回一维的数组

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(training_data_input, training_data_output.ravel())
# ,train_size=0.75, test_size=0.25)默认值

tpot = TPOTRegressor(generations=20, verbosity=2)  # 遗传算法迭代20次
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('pipeline.py')

# s = df.values[:, 9:10].astype(float)
# print(s)
# print(np.around(s, decimals=1))
# data_df = pd.DataFrame(np.around(s, decimals=1))
# data_df.columns = ["施工强度s"]
# writer = pd.ExcelWriter('data/dust_flux1.xlsx')
# data_df.to_excel(writer, startcol=10, index=False, merge_cells=False)  # float_format 控制精度
# writer.close()
# exit()
