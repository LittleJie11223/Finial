import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import Chapter5.Chapter5_4.DataWFW2V4 as da

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
# Build the mlp model
dim=length_input

def clearzeros(listA):
    list=[]
    for i in range(0,listA.shape[0]):
        list2=[]
        for j in listA[i]:
            if j != 0:
                list2.append(j)
        list.append(list2)
    return list
print(x_train.shape)
x_train=clearzeros(x_train)
x_test=clearzeros(x_test)
x_train=pad_sequences(x_train,dtype=float, maxlen=dim,padding='post')
x_test=pad_sequences(x_test,dtype=float, maxlen=dim,padding='post')

y_train=pad_sequences(y_train,dtype=float, maxlen=length_output*3,padding='post')/dicsize
y_test=pad_sequences(y_test,dtype=float,maxlen=length_output*3,padding='post')/dicsize

num_features = len(da.dic2)


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=length_output, input_length=dim),
        layers.LSTM(length_output*3, return_sequences=True),
        layers.LSTM(length_output*3, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model
model = lstm_model()
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
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
predictions = model.predict(test_dish)*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
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
predictions = model.predict(test_dish)*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
list3=[]
for number in list2[0]:
    if number>0 and number<len(da.dic2):
        list3.append(np.int(number))
print(da.mydecode(list3))