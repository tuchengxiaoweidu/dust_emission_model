"""
author: duke.du
datetime:2018/11/28 11:22
function:使用TensorFlow实现BP神经网络
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def standard_deviation(x):
    x_std = np.copy(x)
    for i in range(0, x.shape[1]):
        x_std[:, i] = (x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i]))  # 均值与标准差
    return x_std


def de_standard_deviation(d, m, n):
    d_std = d * (m - n) + n
    return d_std


# 读取输入输出数据
def load_dada(path):
    df = pd.DataFrame(pd.read_excel(path, header=0))
    x_data = standard_deviation(df.values[:, 0:13])
    y_data = standard_deviation(df.values[:, 13:14])
    max_y = np.max(df.values[:, 13:14])
    min_y = np.min(df.values[:, 13:14])

    return x_data, y_data, max_y, min_y


# 权重参数初始化
def init_weight(in_node, out_node, init_method):
    np.random.seed(1)  # 指定生成“特定”的随机数-与seed 1 相关,保证不同的初始化函数生成的随机数一致
    w = 0
    # 均值为0，方差为1的正态分布初始化
    if init_method == "random_normal":
        w = np.random.randn(in_node, out_node)
    # Xavier初始化
    elif init_method == "Xavier":
        w = np.random.randn(in_node, out_node) / np.sqrt(in_node)
    # He 初始化
    elif init_method == "he_init":
        w = np.random.randn(in_node, out_node) / np.sqrt(in_node / 2)
    return w


# 定义层
def fc_layer(input_data, in_node, out_node, activation_fun=None):
    # 参数初始化  权重参数使用he 初始化
    weight = init_weight(in_node, out_node, "he_init")  #
    print("weight: ", weight.shape)
    weight = tf.Variable(weight, dtype=tf.float32)
    bias = tf.Variable(tf.add(tf.zeros(shape=[1, out_node], dtype=tf.float32), 0.1), dtype=tf.float32)
    print("bias: ", bias.shape)
    # y=x*w+b
    y_cal = tf.add(tf.matmul(input_data, weight), bias)
    if activation_fun:
        y_cal = activation_fun(y_cal)
    return y_cal


if __name__ == "__main__":
    file_path = 'boston_data.xlsx'
    training_data_x, training_data_y, m, n = load_dada(file_path)
    size = 500
    x_train = training_data_x[:size]  # 500*13
    y_train = training_data_y[:size]  # .dtype打印出数据类型，.ravel() 返回一维的数组
    x_test = training_data_x[size:]
    y_test = training_data_y[size:]  # float64
    input_size = x_train.shape[1]  # 输入变量个数
    output_size = y_train.shape[1]  # 输出变量个数
    num_sample = len(x_train)  # 样本个数

    # 神经网络结构
    batch_size = 50  # 训练batch的大小
    hide_size = 4  # 隐藏神经元个数

    # 预留每个batch中输入与输出的空间
    x = tf.placeholder(dtype=tf.float32, shape=(None, input_size))  # None指行数不定
    y = tf.placeholder(dtype=tf.float32, shape=(None, output_size))

    # 定义前向传播过程
    layer1 = fc_layer(x, input_size, hide_size, tf.nn.relu)
    y_pred = fc_layer(layer1, hide_size, output_size)

    # 反向传播Tensorflow可以自动进行,只需要定义损失函数和反向计算算法就可以
    learning_rate = 2e-3  # 学习速率

    cross_entropy = tf.reduce_mean(tf.square(y - y_pred))  # 定义损失函数MSE
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)  # 定义反向传播优化方法

    # 创建会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)  # 初始化变量

        # 设定训练次数
        STEPS = 10000  # 训练次数
        for i in range(STEPS):
            sess.run(train_step, feed_dict={x: x_train, y: y_train})
            if STEPS % 100 == 0:
                loss_period = sess.run(cross_entropy, feed_dict={x: x_train, y: y_train})
                print('训练%d次后，误差为%f' % (i, loss_period))
                if loss_period <= 1e-3:
                    break
            elif i == STEPS-1:
                print('未达到训练目标')
        # 保存结果
        saver = tf.train.Saver()
        file_path = 'G:/bptest/'
        save_path = saver.save(sess, file_path)
        predict = sess.run(y_pred, feed_dict={x: x_train})
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
    columns=['ev', 'mae', 'mse', 'r2']
    predict = de_standard_deviation(predict, m, n)  # 转换为向量
    origin = de_standard_deviation(y_train, m, n)
    model_metrics_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(origin, predict)  # 计算每个回归指标结果
        model_metrics_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    df2 = pd.DataFrame(model_metrics_list, index=columns)
    print(df2)
    # print("origin ", origin[:5])
    # 建立时间轴
    t = np.arange(len(x_train))
    plt.plot(t, predict, "ro-", label='predict')
    plt.plot(t, origin, 'gv-', label='origin')
    plt.title("BP network to model")
    plt.xlabel('test data number')
    plt.ylabel('real and predicted values')
    plt.legend(loc='upper right')
    plt.savefig("bp.jpg", dpi=500)
    plt.show()
