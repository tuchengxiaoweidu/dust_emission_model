"""
@author:duke.du
@time:2018/11/28 16:48
@file:gbr_vs_xgb.py
@contact: duke_du123@163.com
@function:function: 读取保存好的pickle模型并应用
"""
import pickle
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import preprocessing

# pickle写入模型
# file = open("model/gbr_model.pickle", "wb")
# pickle.dump(model_gbr_best, file)
# file.close()
df = pd.read_excel('point_validation.xlsx', header=0)
X = df.iloc[:, 8:13].values
y_true = df.iloc[:, 7:8].values

model = joblib.load("model/joblib_gbr_model.joblib")
y_predict = model.predict(X)
mse_test = mean_absolute_error(y_true, y_predict)  # 跟数学公式一样的
rmse_test = round(mse_test ** 0.5, 2)
r2 = round(r2_score(y_true, y_predict),2)
# 画图
plt.figure()
n = np.arange(X.shape[0])
plt.plot(n, y_true, 'r*-', label='origin weight')

plt.plot(n, y_predict, 'go-', label='predict weight')  # 画出每条预测结果线
plt.title('Comparison and RMSE= ' + str(rmse_test) + ',R2= ' + str(r2))  # 标题
plt.legend(loc='upper right')
plt.xlabel('Test data number')
plt.ylabel('Filter weight gain(g/h)')
plt.savefig("picture/application.jpg", dpi=500)
plt.show()
