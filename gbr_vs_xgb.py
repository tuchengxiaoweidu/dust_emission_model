"""
@author:duke.du
@time:2018/11/28 16:48
@file:gbr_vs_xgb.py
@contact: duke_du123@163.com
@function:function: 比较GradientBoostingRegressor与xgboost两种不同实现梯度提升算法的表现形式的优劣
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston  # 波士顿房价集
import xgboost.sklearn as xgb
import time
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, \
    AdaBoostRegressor, RandomForestRegressor

boston = load_boston()
X = boston.data
y = boston.target

# Make a validation set
X_train, X_validation, y_train, y_validation = train_test_split(X,
                                                                y,
                                                                random_state=1848)
# Sci-Kit Learn's Out of the Box Gradient Tree Implementation
sklearn_boost = GradientBoostingRegressor(random_state=1849)
t1 = time.time()
sklearn_boost.fit(X_train, y_train.ravel())
print('Training Error: {:.3f}'.format(1 - sklearn_boost.score(X_train,
                                                              y_train)))
print('Validation Error: {:.3f}'.format(1 - sklearn_boost.score(X_validation,
                                                                y_validation)))
# %timeit sklearn_boost.fit(X_train, y_train.ravel()) # ipython语句，用于测试该语句运行的时间
# XGBoost
xgb_boost = xgb.XGBRegressor(seed=1850)
xgb_boost.fit(X_train, y_train.ravel())
print('Training Error: {:.3f}'.format(1 - xgb_boost.score(X_train,
                                                          y_train)))
print('Validation Error: {:.3f}'.format(1 - xgb_boost.score(X_validation,
                                                            y_validation)))
# %timeit xgb_boost.fit(X_train, y_train.ravel())
