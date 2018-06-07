
# coding: utf-8

# In[1]:



'''
版本：v3.2 在tfLenet中增加初始化、运行lenet5的函数 方便复用
修改时间：2018.5.14 11:40
描述：TensorFlow实现LeNet5
修改内容：
    1、在tfLenet中增加初始化、运行lenet5的函数 方便复用

'''

import tensorflow as tf # 导入tensorflow
import numpy as np
import csv

#新建测试数据变量
#新建卷积核变量
#卷积函数构建
class LeNet5(object):
    def __init__(self,
                 dataset_x_y=np.array([[[],[]],[[],[]]]),
                 run_log_file="lenet5_log",
                 n_epochs = 20,
                 batch_size = 500,
                 learn_rate = 0.1,
                 debug = True,
                 stop_accuracy = 0.97
                ):
        class config(object):
            def __init__(self):
                pass
        self.config = config()
        self.config.debug = debug
        self.config.run_log_file=run_log_file
        self.config.n_epochs = n_epochs
        self.config.epoch = 0
        self.config.batch_size = batch_size
        self.config.learn_rate = learn_rate
        self.config.stop_accuracy = stop_accuracy
        
        # 创建日志文件
        row = "epoch","batch","testaccuracy","n_epochs","batch_size","learn_rate"
        self.write_a_row_log(row)
        
        # 训练集
        self.datasets_train_x = dataset_x_y[0][0]
        self.datasets_train_y = np.eye(np.shape(dataset_x_y[0][1])[0],np.max(dataset_x_y[0][1])+1)[dataset_x_y[0][1]]
        # 测试集
        self.datasets_test_x = dataset_x_y[1][0]
        self.datasets_test_y = np.eye(np.shape(dataset_x_y[1][1])[0],np.max(dataset_x_y[1][1])+1)[dataset_x_y[1][1]]
        
        # 输入数据
        self.x = tf.placeholder("float", shape=[self.config.batch_size, 784])  
        # 训练标签数据  
        self.y_ = tf.placeholder("float", shape=[self.config.batch_size,10])  
        
        x_image = tf.reshape(self.x, [-1,28,28,1])

        # 第一层 卷积-池化层
        self.W1 = tf.Variable(tf.random_uniform([5, 5, 1, 25], minval=-1, maxval=1, dtype=tf.float32),trainable=True, dtype=np.float32,name="W1") 
        self.b1 = tf.Variable(tf.zeros([25,], dtype=tf.float32),trainable=True, dtype=np.float32,name="b1")
        self.s1 = tf.nn.conv2d(x_image, self.W1, strides=[1, 1, 1, 1], padding='VALID',name="Layer1_Conv2d")
        self.f1 = tf.nn.max_pool(self.s1,[1,2,2,1],[1,2,2,1],padding="VALID")
        self.output1 = tf.nn.tanh(tf.add(self.f1,self.b1))
        
        # 第二层 卷积-池化层
        self.W2 = tf.Variable(tf.random_uniform([5, 5, 25, 60], minval=-1, maxval=1, dtype=tf.float32, seed=None, name=None),trainable=True, dtype=np.float32,name="W2") 
        self.b2 = tf.Variable(tf.zeros([60,], dtype=tf.float32),trainable=True, dtype=np.float32,name="b2")
        self.s2 = tf.nn.conv2d(self.output1, self.W2, strides=[1, 1, 1, 1], padding='VALID',name="Layer2_Conv2d")
        self.f2 = tf.nn.max_pool(self.s2,[1,2,2,1],[1,2,2,1],padding="VALID")
        self.output2 = tf.nn.tanh(tf.add(self.f2,self.b2))
        
        # 第三层 隐藏层
        n_in =  60*4*4
        n_out = 500
        hideLayerInput = tf.layers.flatten(self.output2)
        self.W_h = tf.Variable(tf.random_uniform([n_in,n_out], minval=-np.sqrt(6.0/(n_in+n_out)), maxval=np.sqrt(6.0/(n_in+n_out)), dtype=tf.float32),trainable=True, dtype=np.float32,name="W_f")
        self.b_h = tf.Variable(tf.zeros([n_out,], dtype=tf.float32),trainable=True, dtype=np.float32,name="b_h")
        self.output3 = tf.tanh(tf.add(tf.matmul(hideLayerInput,self.W_h),self.b_h))
        
        # 第四层 全连接层
        n_in = 500
        n_out = 10
        self.W_f = tf.Variable(tf.zeros([n_in, n_out], dtype=tf.float32, name=None),trainable=True, dtype=np.float32,name="W_f")
        self.b_f = tf.Variable(tf.zeros([n_out,], dtype=tf.float32, name=None),trainable=True, dtype=np.float32,name="b_f")
        # 到10个类别
        self.p_y_given_x = tf.add(tf.matmul(self.output3, self.W_f) ,self.b_f) # M[500*10]
        # 预测值【0-9】
        self.y_predict = tf.argmax(self.p_y_given_x,axis=1)
        
        # 获得session 由外部传入
        self.session = tf.Session()
        # 初始化全局变量
        
        # self.session.run(tf.local_variables_initializer())
    def printinfo(self,*args):
        # 调试模式下打印
        if self.config.debug:print  ''.join(str(i) for i in args)
        
    def write_a_row_log(self,row):
        with open(self.config.run_log_file+"_lenet5_log.csv",'a') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(row)
        
    def close(self):
        # 释放LeNet5清理环境
        # 保存运行图 (特别消耗存储空间 2018.5.8 )
        #writer = tf.summary.FileWriter('./my_graph', self.session.graph)
        #writer.close()
        self.session.close()
        pass #__init__ 结束
    
    def train(self):
        '''
        训练
        '''
        assert self.session!=None

        # 损失函数
        # loss = self.negative_log_likelihood(self.y_) # 自定义损失函数无法反向传播
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,logits=self.p_y_given_x))
        # 优化器
        train_one_step = tf.train.GradientDescentOptimizer(self.config.learn_rate).minimize(loss) 
        # 变量初始化必须放在优化器后面
        self.printinfo("- initializing variables")
        init = tf.global_variables_initializer()
        self.session.run(init)
        self.printinfo("- start training")
        # 停止训练，主要由训练精度控制
        stopTrain = False
        # 迭代训练 达到训练最大轮次或者停止训练标志为真则停止训练
        while (self.config.epoch < self.config.n_epochs) and stopTrain == False:
            self.config.epoch = self.config.epoch + 1
            # 训练批次
            train_batch = np.shape(self.datasets_train_y)[0]/self.config.batch_size
            for index in range(train_batch):
                self.session.run(train_one_step,feed_dict={
                    self.x:self.datasets_train_x[index * self.config.batch_size : (index + 1) * self.config.batch_size],
                    self.y_:self.datasets_train_y[index * self.config.batch_size : (index + 1) * self.config.batch_size]
                })
                if index % 4 == 0:
                    accuracy = self.test()
                    # 如果精度达到设定停止训练精度则停止下一轮训练
                    if accuracy >= self.config.stop_accuracy:
                        stopTrain=True
                    # 显示状态信息 训练轮次 训练批次 测试精度
                    self.printinfo("- epoch ",self.config.epoch," ,batch ",index ,",test accuracy ",'%.2f' % (100*accuracy),"%")
                    # 保存训练log
                    row = self.config.epoch,index,accuracy,self.config.n_epochs,self.config.batch_size,self.config.learn_rate
                    self.write_a_row_log(row)
                    
        self.printinfo("- Finished training ,use time ")
        pass 
    
    def test(self):
        # 测试集平均准确率
        test_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_predict,tf.argmax(tf.cast(self.y_,tf.int64),axis=1)),tf.float32))
        test_batch = np.shape(self.datasets_test_y)[0]/self.config.batch_size
        accuracy = [
                self.session.run(test_accuracy,feed_dict={
                            self.x:self.datasets_test_x[index * self.config.batch_size : (index + 1) * self.config.batch_size],
                            self.y_:self.datasets_test_y[index * self.config.batch_size : (index + 1) * self.config.batch_size]
                })
            for index in xrange(test_batch)
        ]
        test_score = np.mean(accuracy)
        return test_score
    
def initLeNet5_and_Train(datasets,
                         sel_data_index,
                         rundatafile,
                         learn_rate=0.1,
                         n_epochs=20,
                         batch_size=500):
    '''
    版本：v1.0 封装开始训练lenet5的代码
    创建时间：2018.5.12 11:12
    参数：
        datasets 通过loadData读取的手写体数字原始数据
        sel_data_index 算法选择出的数据的序号集合
    '''
        # 构造lenet5输入数据
    dataset_selected_train_data_and_label = (
        [datasets[0][0][i] for i in sel_data_index],
        np.array([datasets[0][1][i] for i in sel_data_index])
    )
    dataInputOfLenet5 = np.array([dataset_selected_train_data_and_label,datasets[2]])
    # 实例化lenet5
    print "... Initialized lenet5"
    lenet5 = LeNet5(dataInputOfLenet5,rundatafile,learn_rate=learn_rate,n_epochs=n_epochs,batch_size=batch_size)
    # 开始训练
    lenet5.train()
    print "... end training"
    lenet5.close()

