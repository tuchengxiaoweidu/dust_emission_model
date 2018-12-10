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

# 随机挑选 train_size:样本占比
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

# 多层感知器
mlp_predict_disorder = model_mlp.predict(test_x_disorder)

fig = plt.figure(figsize=(20, 3))  # dpi参数制定绘图对象的分辨率
axes = fig.add_subplot(1, 1, 1)
line3, = axes.plot(range(len(test_y_disorder)), ss_y.inverse_transform(test_y_disorder), 'g', label='实际')
line1, = axes.plot(range(len(mlp_predict_disorder)), ss_y.inverse_transform(mlp_predict_disorder), 'r--',
                   label='多层感知器', linewidth=2)
axes.grid()
plt.legend(handles=[line1, line3])
# plt.legend(handels=[line1,line3])
plt.title('sklearn 回归模型')
plt.show()
