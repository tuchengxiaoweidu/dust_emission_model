"""
function：不同模型的评估验证和评价指标
"""
# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
# 批量导入要实现的回归算法:贝叶斯、普通线性回归、弹性回归(岭回归和Lasso回归的组合)
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNetCV  # 批量导入指标算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 验证特征因子与排放量的关系
def draw_scatter(x, y, xlabel):
    plt.scatter(x, y)
    plt.title('%s与施工扬尘排放量散点图' % xlabel)
    plt.xlabel(xlabel)
    plt.ylabel('施工扬尘排放量')
    plt.grid()
    plt.savefig("scatter.jpg", dpi=500)
    plt.show()


df = pd.DataFrame(pd.read_excel('point13_data.xlsx', header=0))

training_data_input = df.values[:, 8:13]  # 500*13
training_data_output = df.values[:, 7:8]  # .dtype打印出数据类型，.ravel() 返回一维的数组
test_data_input = df.values[:, 8:13]
test_data_output = df.values[:, 7:8]  # float64
# draw_scatter(df['平均检测浓度mg/m3'], training_data_output, '平均检测浓度(mg/m3)')
# print(df.info())
# exit()
# size = 500  # train size
# df = pd.DataFrame(pd.read_excel('boston_data.xlsx', header=0))
# training_data_input = df.values[:, 0:13][:size]  # 500*13
# training_data_output = df.values[:, 13:14][:size]  # .dtype打印出数据类型，.ravel() 返回一维的数组
# test_data_input = df.values[:, 0:13][size:]
# test_data_output = df.values[:, 13:14][size:]  # float64

ss_x = preprocessing.MinMaxScaler()
training_data_input = ss_x.fit_transform(training_data_input)  # 估算每个特征的平均值和标准差
test_data_input = ss_x.transform(test_data_input)  # 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性

ss_y = preprocessing.MinMaxScaler()   # 多用于分类问题
training_data_output = ss_y.fit_transform(training_data_output)
test_data_output = ss_y.transform(test_data_output)  # 使得test_y_disorder变成一列

n_folds = 6  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNetCV()  # 建立弹性网络回归模型对象
model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强算法回归模型对象
model_mlp = MLPRegressor()  # 多元感知器网络
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNetCV', 'SVR', 'GBR', 'MLP']  # 不同模型的名称列表
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr, model_mlp]  # 不同回归模型对象的集合

cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    # 将每个回归模型导入交叉检验模型中做训练检验
    scores = cross_val_score(model, training_data_input, training_data_output.ravel(), cv=n_folds,
                             scoring='r2')
    cv_score_list.append(scores)
    pre_y_list.append(
        model.fit(training_data_input, training_data_output.ravel()).predict(training_data_input))  # 将回归训练中得到的预测y存入列表

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
        tmp_score = m(training_data_output.ravel(), pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

# 打印输出交叉检验的数据框
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
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
plt.plot(x, ss_y.inverse_transform(training_data_output), color='r', label='origin y')
color_list = ['k+', 'b.', 'go', 'cv', 'y*', 'm^']  # 颜色列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(x, ss_y.inverse_transform(pre_y_list[i].reshape(-1, 1)), color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('Comparison of results by six regression model')  # 标题
plt.legend(loc='upper right')
plt.xlabel('Test data number')
plt.ylabel('Filter weight gain(g/h)')
plt.savefig("picture/all regression compare.jpg", dpi=500)
plt.show()

# 模型应用
print('regression prediction:')
new_pre_y = ss_y.inverse_transform(model_gbr.predict(test_data_input).reshape(-1, 1))  # 使用GBR进行预测

print('predict data \t real data')
test_y_output = ss_y.inverse_transform(test_data_output)
for i in range(len(test_data_input)):
    print('  %.2f \t %0.2f' % (new_pre_y[i], test_y_output[i]))  # 打印输出每个数据点的预测信息
mse = mean_squared_error(test_y_output, new_pre_y)
print("The mse of model_gbr is : %f" % mse)
# if __name__ == "__main__":
#     svm_baseline()
