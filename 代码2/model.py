#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 17:15
# @Author  : ZWQ  
# @File    : model.py
# @Software: PyCharm


import tensorflow as tf
import numpy as np


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("is_train", 1, "指定程序是预测还是训练")


def oneLayerNN():

    # 1、建立数据占位符，x [None, 192], y_true [None, 5]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, shape=[None, 192])
        y_true = tf.placeholder(tf.int32, [None, 5])

    # 2、建立第一层
    with tf.variable_scope("layer_1"):
        # 随机初始化权重和偏置
        weight_1 = tf.Variable(tf.random_normal([192, 100], mean=0.0, stddev=1.0), name='w_1')
        bias_1 = tf.Variable(tf.constant(0.0, shape=[100]))
        y_1 = tf.matmul(x, weight_1) + bias_1

    # 2、建立一个全连接层网络 w[100, 5], b[5]
    with tf.variable_scope("fc"):
        # 随机初始化权重和偏置
        weight_fc = tf.Variable(tf.random_normal([100, 5], mean=0.0, stddev=1.0), name='w')
        bias_fc = tf.Variable(tf.constant(0.0, shape=[5]))

        # 预测None个样本的输出结果
        y_predict = tf.matmul(y_1, weight_fc) + bias_fc

    # 3、求出所有样本的损失，然后求平均值
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4、梯度下降求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)

    # 5、计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))

        # equal_list  None个样本   [1, 0]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量， 单个
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    # 高纬度变量收集
    tf.summary.histogram("weight_1", weight_1)
    tf.summary.histogram("bias_1", bias_1)

    tf.summary.histogram("weight_fc", weight_fc)
    tf.summary.histogram("bias_fc", bias_fc)

    # 定义一个合并变量的op
    merged = tf.summary.merge_all()

    # 定义一个初始化的op
    init_op = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 建立events文件，然后写入
        filewriter = tf.summary.FileWriter("./summary/", graph=sess.graph)

        if FLAGS.is_train == 1:

            # 迭代步数取训练，更新参数预测
            for i in range(2000):

                # 取出真实存在的特征值和目标值
                data_x = np.loadtxt("data_pre.txt")

                data_y = np.loadtxt("label.txt")

                # 运行train_op训练
                sess.run(train_op, feed_dict={x:data_x, y_true: data_y})

                # 写入每步训练的值
                summary = sess.run(merged, feed_dict={x:data_x, y_true:data_y})
                filewriter.add_summary(summary, i)

                print("训练第%d步，准确率为：%f" %(i, sess.run(accuracy, feed_dict={x:data_x , y_true: data_y})))

            # 保存模型
            saver.save(sess, "model/fc_model")

        else:
            # 加载模型
            saver.restore(sess, "model/fc_model")
            # 如果是0做出预测
            for i in range(200):
                # 每次测试一个数据
                x_test_new = []
                x_test = np.loadtxt("data_pre.txt")[i]
                x_test_new.append(x_test)

                y_test_new = []
                y_test = np.loadtxt("label.txt")[i]
                y_test_new.append(y_test)
                print("第%d个样本,目标是:%d,预测结果是" %(
                    i,
                    y_test[1],
                ))
                print(tf.arg_max(sess.run(y_predict, feed_dict={x:x_test_new, y_true:y_test_new}), 1).eval())

    return None


if __name__ == "__main__":
    oneLayerNN()