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
dicsize=len(dic)
# Build the mlp model
# print(x_train.shape)
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
x_train=pad_sequences(x_train,dtype=float, maxlen=dim,padding='post')/dicsize
x_test=pad_sequences(x_test,dtype=float, maxlen=dim,padding='post')/dicsize

y_train=pad_sequences(y_train,dtype=float, maxlen=length_output*3,padding='post')/dicsize
y_test=pad_sequences(y_test,dtype=float, maxlen=length_output*3,padding='post')/dicsize

num_features = len(dic)

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
list=[]
for i in test_dish:
    list.append(da.myencode(dic,i))
test_dish=list
test_dish=pad_sequences(test_dish,dtype=float, maxlen=dim,padding='post')/dicsize
predictions = model.predict(test_dish)*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
list3=[]
for number in list2[0]:
    if number>0 and number<len(dic):
        list3.append(np.int(number))
print(da.mydecode(dic,list3))

# test the ingredient : test_dish=['Bang Bone','Hoisin Sauce','Black Pepper Powder','Prickly Prickly','Pineapple']
test_dish=['棒骨','海鲜酱','黑椒粉','花椒粒','菠萝']
print(test_dish)
list=[]
for i in test_dish:
    list.append(da.myencode(dic,i))
test_dish=list
test_dish=pad_sequences(test_dish,dtype=float, maxlen=dim,padding='post')/dicsize
predictions = model.predict(test_dish)*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
list3=[]
for number in list2[0]:
    if number>0 and number<len(dic):
        list3.append(np.int(number))
print(da.mydecode(dic,list3))