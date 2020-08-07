import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import Chapter4.DataSolver.datasolver as da

# get the data from the json file
list_train = da.getInput()
x_train  = np.array(list_train[0:1100])
x_test  = np.array(list_train[1100:len(list_train)+1])

list_test = da.getOutput()
y_train = np.array(list_test[0:1100])
y_test = np.array(list_test[1100:len(list_test)+1])

list = da.getsizeIn()
length_input=list[1]
size_input=list[0]
length_output=da.getsizeOut()
length_output=length_output[0]
dic=da.getdic()
# Build the mlp model
dim=length_input*size_input
x_train=x_train.reshape(x_train.shape[0],dim)
x_test=x_test.reshape(x_test.shape[0],dim)

def clearzeros(listA):
    list=[]
    for i in range(0,listA.shape[0]):
        list2=[]
        for j in listA[i]:
            if j != 0:
                list2.append(j)
        list.append(list2)
    return list
x_train=clearzeros(x_train)
x_test=clearzeros(x_test)
x_train=pad_sequences(x_train,dtype=float, maxlen=dim)/len(da.dic)
x_test=pad_sequences(x_test,dtype=float, maxlen=dim)/len(da.dic)

y_train=pad_sequences(y_train, maxlen=length_output*3)/len(da.dic)
y_test=pad_sequences(y_test,maxlen=length_output*3)/len(da.dic)

num_features = len(dic)
embedding_dimension = 100

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

def cnn_mulfilter():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension,
                        input_length=dim),
        layers.Reshape((dim, embedding_dimension, 1)),
        convolution(),
        layers.Flatten(),
        layers.Dense(length_output*3, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(length_output*3, activation='sigmoid')

    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model

model = cnn_mulfilter()
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
model.evaluate(x_test , y_test)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()

# test the ingredient : test_dish=['Sugar','Fish','Vinegar','Prickly Pepper','Soy Sauce','Chili','Tomato']
test_dish=['白糖','鲫鱼','醋','生抽','辣椒','西红柿']
print(test_dish)
list=[]
for i in test_dish:
    list.append(da.myencode(dic,i))
test_dish=list
test_dish=pad_sequences(test_dish,dtype=float, maxlen=dim)/len(dic)
predictions = model.predict(test_dish)*len(da.dic)
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
# print(list2[0])
# print(type(list2[0][0]))
list3=[]
for number in list2[0]:
    if number>0 and number<len(da.dic):
        list3.append(np.int(number))
print(da.mydecode(dic,list3))

# test the ingredient : test_dish=['Bang Bone','Hoisin Sauce','Black Pepper Powder','Prickly Prickly','Pineapple']
test_dish=['棒骨','海鲜酱','黑椒粉','花椒粒','菠萝']
print(test_dish)
list=[]
for i in test_dish:
    list.append(da.myencode(dic,i))
test_dish=list
test_dish=pad_sequences(test_dish,dtype=float, maxlen=dim)/len(dic)
predictions = model.predict(test_dish)*len(da.dic)
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
# print(list2[0])
# print(type(list2[0][0]))
list3=[]
for number in list2[0]:
    if number>0 and number<len(da.dic):
        list3.append(np.int(number))
print(da.mydecode(dic,list3))