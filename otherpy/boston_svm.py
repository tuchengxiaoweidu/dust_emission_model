import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_excel('xisha.xls', header=0)
X = df.iloc[:, 2:8].values
y = df.iloc[:, 8].values
train_size = 350
x_train_data = X[:train_size]
y_train_data = y[:train_size]
x_test_data = X[train_size:]
y_test_data = y[train_size:]

# 没有归一化
# train
# 寻找合适的参数，gamma核系数
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})   # 岭回归

svr.fit(x_train_data, y_train_data)
kr.fit(x_train_data, y_train_data)

y_svr = svr.predict(x_test_data)  # test data,
y_kr = kr.predict(x_test_data)
plt.plot(range(len(x_test_data)), y_test_data, c='b', label="origin")
plt.plot(range(len(x_test_data)), y_svr, c='r', label="svr")
plt.plot(range(len(x_test_data)), y_kr, c='g', label="kr")
plt.xlabel('test size')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
plt.show()

