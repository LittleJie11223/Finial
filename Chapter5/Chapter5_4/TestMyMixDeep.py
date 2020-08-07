import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import Chapter5.Chapter5_4.DataTest1 as da

# get the data from the json file
list_train = da.getInput()
x_train  = np.array(list_train[0:1000])
x_test  = np.array(list_train[1000:len(list_train)+1])

list_test = da.getOutput()
y_train = np.array(list_test[0:1000])
y_test = np.array(list_test[1000:len(list_test)+1])

list = da.getsizeIn()
length_input=list[1]
size_input=list[0]
length_output=da.getsizeOut()
length_output=length_output[0]
dicsize=len(da.dic2)

dim=length_input*size_input

x_train=pad_sequences(x_train,dtype='float32', maxlen=dim,padding='post')
x_test=pad_sequences(x_test,dtype='float32', maxlen=dim,padding='post')

y_train=pad_sequences(y_train,dtype='float32', maxlen=length_output,padding='post')/dicsize
y_test=pad_sequences(y_test,dtype='float32',maxlen=length_output,padding='post')/dicsize

num_features = len(da.dic2)
embedding_dimension = 100

print('********************************')
print(len(list_train[0]))
print(x_train.shape)
print(x_train[0])
print(y_train.shape)
print(y_train[0])

filter_sizes=[3,4,5]
def convolution():
    inn = layers.Input(shape=(dim, embedding_dimension, 1))
    cnns = []
    for size in filter_sizes:
        conv = layers.Conv2D(filters=length_output, kernel_size=(size, embedding_dimension),
                            strides=1, padding='valid', activation='relu')(inn)
        pool = layers.MaxPool2D(pool_size=(dim-size+1, 1), padding='valid')(conv)
        cnns.append(pool)
    outt = layers.concatenate(cnns)

    model = keras.Model(inputs=inn, outputs=outt)
    return model

def my_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension,
                        input_length=dim),
        layers.Reshape((dim, embedding_dimension, 1)),
        convolution(),
        layers.Flatten(),
        layers.Dense(length_output, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(length_output, activation='sigmoid'),

    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model

model = my_model()
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.01)
model.evaluate(x_test , y_test)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()

# test the ingredient : test_dish=['Sugar','Fish','Vinegar','Prickly Pepper','Soy Sauce','Chili','Tomato']
test_dish=['糖','鱼','醋','生抽','辣椒','西红柿']
print(test_dish)
test_dish=da.myencode1(test_dish)
for i in range(0,dim-len(test_dish)):
    test_dish.append(0)
list=[]
for number in test_dish:
    list.append(number)
list2=[]
list2.append(list)
test_dish=np.array(list2)
predictions = model.predict(test_dish)*dicsize*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
print('#')
print(list2)
list3=[]
for number in list2[0]:
    if number>0 and number<len(da.dic2):
        list3.append(np.int(number))
print(da.mydecode(list3))

test_dish=['棒骨','海鲜酱','黑椒粉','花椒粒','菠萝']
print(test_dish)
test_dish=da.myencode1(test_dish)
for i in range(0,dim-len(test_dish)):
    test_dish.append(0)
list=[]
for number in test_dish:
    list.append(number)
list2=[]
list2.append(list)
test_dish=np.array(list2)
predictions = model.predict(test_dish)*dicsize*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
print('#')
print(list2)
list3=[]
for number in list2[0]:
    if number>0 and number<len(da.dic2):
        list3.append(np.int(number))
print(da.mydecode(list3))