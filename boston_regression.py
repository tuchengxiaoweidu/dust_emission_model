"""
CRIM：城镇人均犯罪率。
ZN：住宅用地超过 25000 sq.ft. 的比例。
INDUS：城镇非零售商用土地的比例。
CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
NOX：一氧化氮浓度。
RM：住宅平均房间数。
AGE：1940 年之前建成的自用房屋比例。
DIS：到波士顿五个中心区域的加权距离。
RAD：辐射性公路的接近指数。
TAX：每 10000 美元的全值财产税率。
PTRATIO：城镇师生比例。
B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
LSTAT：人口中地位低下者的比例。
MEDV：自住房的平均房价，以千美元计。
预测平均值的基准性能的均方根误差（RMSE）是约 9.21 千美元。
"""
# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
# 批量导入要实现的回归算法:贝叶斯、普通线性回归、弹性回归(岭回归和Lasso回归的组合)
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
# 批量导入指标算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score  # 交叉检验
# 批量导入指标算法
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 对稀疏数据做标准化，不能采用中心化的方式，否则会破坏稀疏数据的结构.比如此文件的boston数据集

size = 500  # train size
df = pd.DataFrame(pd.read_excel('boston_data.xlsx', header=0))
# np.random.shuffle()
training_data_input = df.values[:, 0:13][:size]  # 500*13
training_data_output = df.values[:, 13:14][:size].ravel()  # .dtype打印出数据类型，.ravel() 返回一维的数组
test_data_input = df.values[:, 0:13][size:]
test_data_output = df.values[:, 13:14][size:].ravel()  # float64

n_folds = 6  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强算法回归模型对象
model_mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=1)
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR', 'MLP']  # 不同模型的名称列表
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr, model_mlp]  # 不同回归模型对象的集合

cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    # 将每个回归模型导入交叉检验模型中做训练检验
    scores = cross_val_score(model, training_data_input, training_data_output, cv=n_folds)   # 模型与数据间的距离，并非越大越好
    cv_score_list.append(scores)
    pre_y_list.append(
        model.fit(training_data_input, training_data_output).predict(training_data_input))  # 将回归训练中得到的预测y存入列表

# 模型效果指标评估
"""
explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度，其其值越小说明拟合效果越好。
mean_squared_error：均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好。
r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
"""
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(len(model_dic)):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(training_data_output, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
# 打印输出交叉检验的数据框
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
# 建立回归指标的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])
n_samples, n_features = training_data_input.shape  # 总样本量,总特征数
print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
print('cross validation result:')  # 打印输出标题
print(df1)  # 打印输出交叉检验的数据框
print(70 * '-')
print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(70 * '-')
print("Remarks:")
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance')
print('mae \t mean_absolute_error')
print('mse \t mean_squared_error')
print('r2 \t coefficient of determination')
print(70 * '-')
exit()
#  模型效果可视化
plt.figure()
x = np.arange(training_data_input.shape[0])
plt.plot(x, training_data_output, color='r', label='origin y')
color_list = ['k.', 'b.', 'go', 'yv', 'c*', 'm^']  # 颜色列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(x, pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')
plt.xlabel('test data number')
plt.ylabel('real and predicted values')
# plt.savefig("regression compare.jpg", dpi=500)
plt.show()

# 模型应用
print('regression prediction:')
print('predict data \t real data')
new_pre_y = model_gbr.predict(test_data_input)  # 使用GBR进行预测
model_gbr_score = model_gbr.score(test_data_input, test_data_output)
print("The score of model_gbr is : %f" % model_gbr_score)
for i in range(len(test_data_input)):
    print('  %.2f \t %0.2f' % (new_pre_y[i], test_data_output[i]))  # 打印输出每个数据点的预测信息

# if __name__ == "__main__":
#     svm_baseline()
