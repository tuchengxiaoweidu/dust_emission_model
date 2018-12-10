import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 数据
df = pd.read_excel('xisha.xls', header=0)
x = df.iloc[:, 2:8].values
y = df.iloc[:, 8].values
# print('input：', x.shape)
# print(x)
# print('output：', y.shape)
# print(y)
# exit()

"""
随机挑选 train_size:样本占比
"""
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x, y, test_size=0.2,
                                                                                        random_state=33)

# 数据标准化#:为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
ss_x = preprocessing.StandardScaler()
train_x_disorder = ss_x.fit_transform(train_x_disorder)  # 估算每个特征的平均值和标准差
test_x_disorder = ss_x.transform(test_x_disorder)  # 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性

ss_y = preprocessing.StandardScaler()
train_y_disorder = ss_y.fit_transform(train_y_disorder.reshape(-1, 1))
test_y_disorder = ss_y.transform(test_y_disorder.reshape(-1, 1))  # 使得test_y_disorder变成一列

# 多层感知器-回归模型alpha : default 0.0001
model_mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=1)  # default ‘relu’
model_mlp.fit(train_x_disorder, train_y_disorder.ravel())  # 将多维数组降位一维
mlp_score = model_mlp.score(test_x_disorder, test_y_disorder.ravel())  # # 将多维数组降位一维
print("sklearn多层感知器-回归模型得分", mlp_score)
# labels回归
model_gbr_disorder = GradientBoostingRegressor()
model_gbr_disorder.fit(train_x_disorder, train_y_disorder.ravel())
gbr_score_disorder = model_gbr_disorder.score(test_x_disorder, test_y_disorder.ravel())
print('sklearn集成-回归模型得分', gbr_score_disorder)
"""
# 参数网格优选#############
model_gbr_GridSearch = GradientBoostingRegressor()
# 设置参数池
param_grid = {'n_estimators': range(20, 100, 10),
              'learning_rate': [0.3, 0.2,0.1, 0.05, 0.02, 0.01],
              'max_depth': [3,4, 6, 8], 'min_samples_leaf': [3, 5, 9, 14],
              'max_features': [0.8, 0.5, 0.3, 0.1]}
# 网格调参
estimator = GridSearchCV(model_gbr_GridSearch, param_grid)  # 所使用的分类器,最优化的参数的取值:字典或列表
estimator.fit(train_x_disorder, train_y_disorder.ravel())
print('最优调参：', estimator.best_params_)
print('调参后得分', estimator.score(test_x_disorder, test_y_disorder.ravel()))
exit()
"""
# 画图
model_gbr_best = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features=0.5, min_samples_leaf=3,
                                           n_estimators=70)
# 使用默认参数的模型进行预测
model_gbr_best.fit(train_x_disorder, train_y_disorder.ravel())
gbr_score_best = model_gbr_best.score(test_x_disorder, test_y_disorder.ravel())
gbr_predict_disorder = model_gbr_best.predict(test_x_disorder)
print('优化后的sklearn集成-回归模型得分', gbr_score_best)
# 多层感知器
mlp_predict_disorder = model_mlp.predict(test_x_disorder)

fig = plt.figure(figsize=(20, 3))  # dpi参数制定绘图对象的分辨率
axes = fig.add_subplot(1, 1, 1)
line3, = axes.plot(range(len(test_y_disorder)), ss_y.inverse_transform(test_y_disorder), 'g', label='实际')
line2, = axes.plot(range(len(gbr_predict_disorder)), ss_y.inverse_transform(gbr_predict_disorder), 'b--',
                   label='集成模型', linewidth=2)
line1, = axes.plot(range(len(mlp_predict_disorder)), ss_y.inverse_transform(mlp_predict_disorder), 'r--',
                   label='多层感知器', linewidth=2)
axes.grid()
plt.legend(handles=[line1, line2, line3])
# plt.legend(handels=[line1,line3])
plt.title('sklearn 回归模型')
plt.show()
