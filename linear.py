#线性回归预测链家房源价格
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据标准化的函数
def normalize_feature(df):
    return df.apply(lambda column:(column-column.mean())/column.std())


df = normalize_feature(pd.read_csv("ljhouse.csv",names=["singleprice","roomnum","livingrm","square","decro","price"]))#读入房价数据并标准化
ones = pd.DataFrame({"ones": np.ones(len(df))})# 生成n行1列的矩阵，供常数项使用
df = pd.concat([ones,df],axis=1)#n行1列的矩阵和房价矩阵拼接
#print(df)

X_data = np.array(df[df.columns[0:5]])#读入前5个参数，均为房价的影响因素
y_data = np.array(df[df.columns[-1]]).reshape(len(df),1)#读取房价

#print(X_data.shape,type(X_data))#调试用，查看矩阵类型
#print(y_data.shape,type(y_data))

alpha = 0.01 # 学习率
epoch = 500 # 训练轮数

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, X_data.shape, name='X')# 输入房屋信息参数，300行5列
    y = tf.placeholder(tf.float32, y_data.shape, name='y')# 输出房价，300行1列

with tf.name_scope('hypothesis'):
    W = tf.get_variable("weights",
                        (X_data.shape[1], 1),
                        initializer=tf.constant_initializer()) # 权重变量，5行1列
    # 假设函数 h(x) = w0*x0+w1*x1+w2*x2+w3*x3+w4*x4, 其中x0恒为1
    y_pred = tf.matmul(X, W, name='y_pred')    # 预测值y_pred,300行1列

with tf.name_scope('loss'):  # 损失函数采用最小二乘法，损失为loss_op
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)

with tf.name_scope('train'): # 采用梯度下降法进行优化
    train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# 初始化全局变量
    loss_data = []    # 记录每一次训练的损失值
    # 开始训练模型
    for e in range(1, epoch + 1):
        _, loss, w = sess.run([train_op, loss_op, W], feed_dict={X: X_data, y: y_data})
        loss_data.append(float(loss))  # 记录每一轮损失值变化情况
        if e % 10 == 0:
            log_str = "Epoch %d Loss=%.4g Model: y = %.4gx1 + %.4gx2+ %.4gx3 + %.4gx4 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[3], w[4], w[0]))#每10轮输出一次损失和权重

#测试预测效果
dftest = normalize_feature(pd.read_csv("testlj.csv",names=["singleprice","roomnum","livingrm","square","decro","price"]))#读入测试数据并标准化，下同训练数据
onest = pd.DataFrame({"ones": np.ones(len(dftest))})
dftest = pd.concat([onest,dftest],axis=1)

X_testdata = np.array(dftest[dftest.columns[0:5]])
y_testdata = np.array(dftest[dftest.columns[-1]]).reshape(len(dftest),1)

ytlist=[]#预测值列表
#xtlist=[]#预测所需参数的列表
ylist=[]#真实值列表
for i in range(len(dftest)):
    #for j in range(5):#调试用
		#print(X_testdata[i][j]*w[0])
	yt=X_testdata[i][0]*w[0]+X_testdata[i][1]*w[1]+X_testdata[i][2]*w[2]+X_testdata[i][3]*w[3]+X_testdata[i][4]*w[4]#计算预测值
	ylist.append(y_testdata[i])
	ytlist.append(yt)
	#xtlist.append(i)

ylt=np.array(ylist)
ytt=np.array(ytlist)
#xtt=np.array(xtlist)

x = np.arange(-2,4) #供参考的直线
y = x

plt.title("house price prediction") #图像及横纵坐标的名称
plt.xlabel("predicted prices") 
plt.ylabel("real prices") 

plt.plot(x,y,label="test",color = 'm') #供参考的直线
plt.scatter(ytt, ylt,marker='x',alpha=0.5,edgecolors= 'white')#预测的值和真实值的散点

plt.show()   #绘图



