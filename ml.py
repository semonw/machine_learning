import tensorflow as tf
import time

startClock = time.time()

mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape)) 

print(f"train_x ndim {train_x.ndim}, type:{train_x.dtype}");


#归一化、并转换为tensor张量，数据类型为float32.
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     #添加Flatten层说明输入数据的形状
model.add(tf.keras.layers.Dense(128,activation='relu'))     #添加隐含层，为全连接层，128个节点，relu激活函数
model.add(tf.keras.layers.Dense(10,activation='softmax'))   #添加输出层，为全连接层，10个节点，softmax激活函数
print('\n',model.summary())     #查看网络结构和参数信息

#adam算法参数采用keras默认的公开参数，损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])   

#批量训练大小为64，迭代5次，测试集比例0.2（48000条训练集数据，12000条测试集数据）
history = model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

model.evaluate(X_test,y_test,verbose=2)     #每次迭代输出一条记录，来评价该模型是否有比较好的泛化能力

#保存模型参数
model.save_weights('mnist_weights.h5')
#保存整个模型
model.save('mnist_model.h5')

endClock = time.time()

print(f"程序总计耗时:{endClock-startClock} s")








