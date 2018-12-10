import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle  # 用pickle读写模型
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 获取模型最优参数
def get_best_para(train_x_data, test_x_data, train_y_data, test_y_data):
    # 数据标准化#:为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
    # ss_x = preprocessing.StandardScaler()
    # train_x_disorder = ss_x.fit_transform(train_x_disorder)  # 估算每个特征的平均值和标准差
    # test_x_disorder = ss_x.transform(test_x_disorder)  # 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
    #
    # ss_y = preprocessing.StandardScaler()
    # train_y_disorder = ss_y.fit_transform(train_y_disorder.reshape(-1, 1))
    # test_y_disorder = ss_y.transform(test_y_disorder.reshape(-1, 1))  # 使得test_y_disorder变成一列

    # 参数网格优选
    model_gbr_GridSearch = GradientBoostingRegressor(random_state=1)
    # 设置参数池：一般情况下learning_rate越低，n_estimators越大；GBDT中的决策树是个弱模型，深度较小一般不会超过5，
    # 叶子节点的数量也不会超过10，对于生成的每棵决策树乘上比较小的缩减系数（学习率<0.1）
    param_grid = {'n_estimators': range(50, 150, 20),
                  'learning_rate': [0.2, 0.15, 0.1, 0.05, 0.02, 0.01],
                  'max_depth': [3, 4, 6, 8], 'min_samples_leaf': [1, 3, 5, 9, 14],
                  'max_features': [0.8, 0.5, 0.3, 0.1]}  # 'loss': ['ls', 'lad', 'huber','quantile']
    # 网格调参
    clf = GridSearchCV(model_gbr_GridSearch, param_grid)  # 所使用的分类器,最优化的参数的取值:字典或列表
    clf.fit(train_x_data, train_y_data.ravel())
    # cv_result = pd.DataFrame.from_dict(clf.cv_results_)  # 将所有结果读出
    # with open('cv_result.csv', 'w') as f:
    #     cv_result.to_csv(f)
    print('最优调参：', clf.best_params_)
    print('调参后测试模型得分', clf.score(test_x_data, test_y_data.ravel()))
    exit()
    return clf.best_params_


def train_model():
    df2 = pd.read_excel('data/point1234.xlsx', header=0)
    x = df2.iloc[:, 8:13].values
    y = df2.iloc[:, 7:8].values
    """# 零值的处理，用特征列的均值代替
    x = x.replace("0", np.NAN)  # NaN却是用一个特殊的float，用来标示空缺数据
    # 使用Imputer给定缺省值，默认的是以平均值
    imputer = Imputer(missing_values="NaN", strategy='mean')
    x = imputer.fit_transform(x)
    """
    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x, y,
                                                                                            random_state=1)
    # search the best parameters
    best_params_list = get_best_para(train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder)

    # 建模
    # model_gbr_best = GradientBoostingRegressor()
    model_gbr_best = GradientBoostingRegressor(learning_rate=best_params_list['learning_rate'],
                                               max_depth=best_params_list['max_depth'],
                                               min_samples_leaf=best_params_list['min_samples_leaf'],
                                               n_estimators=best_params_list['n_estimators'])
    # train model by optimized parameters
    model_gbr_best.fit(train_x_disorder, train_y_disorder.ravel())
    # save model by joblib
    joblib.dump(model_gbr_best, "model/joblib_gbr_model.joblib")
    gbr_predict_disorder = model_gbr_best.predict(test_x_disorder)  # 预测数据

    gbr_score_best = model_gbr_best.score(test_x_disorder, test_y_disorder.ravel())
    gbr_score_best = round(gbr_score_best, 2)
    print('优化后的sklearn集成-回归模型得分,即拟合度R2', gbr_score_best)
    """
    # compute test set deviance
    test_score = np.zeros((best_params_list['n_estimators'],), dtype=np.float64)
    
    for i, y_pred in enumerate(model_gbr_best.staged_predict(test_x_disorder)):
        test_score[i] = model_gbr_best.loss_(test_y_disorder, y_pred)   # 越小越好

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(best_params_list['n_estimators']) + 1, model_gbr_best.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(best_params_list['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='best')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # #############################################################################
    # Plot feature importance
    feature_names = ['TSP浓度', '施工强度', '温度', '湿度', '风速']
    feature_importance = model_gbr_best.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    pos = np.arange(feature_importance.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.savefig("picture/variable.jpg", dpi=500)
    plt.show()
    exit()"""
    # 画图
    fig = plt.figure()  # dpi参数制定绘图对象的分辨率
    axes = fig.add_subplot(1, 1, 1)
    line3, = axes.plot(range(len(test_y_disorder)), test_y_disorder, 'r*-', label='实际')  # ss_y.inverse_transform(
    line2, = axes.plot(range(len(gbr_predict_disorder)), gbr_predict_disorder, 'g^-',
                       label='集成模型', linewidth=2)
    axes.grid()
    plt.legend(handles=[line2, line3])
    plt.title('GBR回归模型,r2= ' + str(gbr_score_best))
    plt.show()


if __name__ == '__main__':
    # 1:train model
    train_model()

    # 2: read validation data
    file_path = 'point_validation.xlsx'
    df = pd.read_excel(file_path, header=0)
    # print(df.columns.values.tolist())  # 获取列名
    X = df.iloc[:, 8:13].values
    y_true = df.iloc[:, 7:8].values

    # 3:load model and predict
    model = joblib.load("model/joblib_gbr_model.joblib")
    y_predict = model.predict(X)
    mse_test = mean_absolute_error(y_true, y_predict)  # 跟数学公式一样的
    rmse_test = round(mse_test ** 0.5, 2)
    r2 = round(r2_score(y_true, y_predict), 2)
    # plot
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
