import datetime
starttime = datetime.datetime.now()
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import Chapter6.MyDataAll1 as da
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# get the data from the json file
list_train = da.getInput()
x_train  = np.array(list_train[0:len(list_train)-1000])
x_test  = np.array(list_train[len(list_train)-1000:len(list_train)+1])

list_test = da.getOutput()
y_train = np.array(list_test[0:len(list_test)-1000])
y_test = np.array(list_test[len(list_test)-1000:len(list_test)+1])

# get the data matrix size
list = da.getsizeIn()
length_input=list[0]
size_input=list[1]
length_output=da.getsizeOut()
length_output=length_output[0]
dicsize=len(da.dic2)

y_train=pad_sequences(y_train,dtype=float, maxlen=length_output,padding='post')/dicsize
y_test=pad_sequences(y_test,dtype=float,maxlen=length_output,padding='post')/dicsize

# reshape the input data
print(x_train.shape)
print(x_test.shape)
print(len(x_train))
print(len(x_test))
print(y_train.shape)
x_train = x_train.reshape((-1,length_input,size_input,1))
x_test = x_test.reshape((-1,length_input,size_input,1))

print(y_test.shape)

#setting the CNN layers
deep_model = keras.Sequential(
[
    layers.Conv2D(input_shape=((x_train.shape[1], x_train.shape[2], x_train.shape[3])),
                 filters=length_output, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(filters=length_output, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.Conv2D(filters=length_output, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Reshape((-1,length_output)),
    layers.LSTM(length_output,activation='relu',return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(length_output,activation='sigmoid',return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(length_output,activation='relu'),
    layers.Dense(length_output,activation='softmax')
])
# setting the parm
deep_model.compile(optimizer=keras.optimizers.Adam(),
            # loss=keras.losses.SparseCategoricalCrossentropy(),
            loss=keras.losses.BinaryCrossentropy(),
             # loss='mean_absolute_error',
            metrics=['accuracy'])
deep_model.summary()

# model fit
history = deep_model.fit(x_train, y_train,  epochs=2, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

res = deep_model.evaluate(x_test, y_test)
deep_model.save('/Users/dengjie/PycharmProjects/program/Chapter6/Chapter6_3/myCLD_model.h5')
# test the ingredient : test_dish=['Sugar','Fish','Vinegar','Prickly Pepper','Soy Sauce','Chili','Tomato']
test_dish1=['糖','鱼','醋','生抽','辣椒','西红柿']
print(test_dish1)

A1=np.zeros([length_input,size_input])
listt=[]
for str in test_dish1:
    if da.myencode1(str)!=[]:
        listt.append(da.myencode1(str))
for i in range(len(listt)):
    A1[:,i]=np.array(listt[i])
test_dish2=['棒骨','海鲜酱','黑椒粉','花椒粒','菠萝']
print(test_dish2)
A2=np.zeros([length_input,size_input])
listt=[]
for str in test_dish2:
    if da.myencode1(str)!=[]:
        listt.append(da.myencode1(str))
for i in range(len(listt)):
    A2[:,i]=np.array(listt[i])
list=[]
list.append(A1)
list.append(A2)
list=np.array(list)
list = list.reshape((-1,length_input,size_input,1))
predictions = deep_model.predict(list)*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
list3=[]
for num in list2[0]:
    if num>0 and num<len(da.dic2):
        list3.append(np.int(num))
print(da.mydecode(list3))
list3=[]
for num in list2[1]:
    if num>0 and num<len(da.dic2):
        list3.append(np.int(num))
print(da.mydecode(list3))

endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)