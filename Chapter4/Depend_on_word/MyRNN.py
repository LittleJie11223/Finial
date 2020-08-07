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
x_train=pad_sequences(x_train,dtype=float, maxlen=dim)/dicsize
x_test=pad_sequences(x_test,dtype=float, maxlen=dim)/dicsize

y_train=pad_sequences(y_train,dtype=float, maxlen=length_output*3)/dicsize
y_test=pad_sequences(y_test,dtype=float, maxlen=length_output*3)/dicsize

num_features = len(dic)

def get_model():
    model = keras.Sequential([
        layers.Embedding(num_features, length_output),
        layers.Bidirectional(keras.layers.LSTM(length_output)),
        layers.Dense(length_output*3, activation='relu'),
        layers.Dense(length_output*3, activation='sigmoid')
    ])
    return model
model = get_model()
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
model.evaluate(x_test , y_test)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')

# test the ingredient : test_dish=['Sugar','Fish','Vinegar','Prickly Pepper','Soy Sauce','Chili','Tomato']
test_dish=['糖','鱼','醋','生抽','辣椒','西红柿']
print(test_dish)
list=[]
for i in test_dish:
    list.append(da.myencode(dic,i))
test_dish=list
test_dish=pad_sequences(test_dish,dtype=float, maxlen=dim)/dicsize
predictions = model.predict(test_dish)*dicsize
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
test_dish=pad_sequences(test_dish,dtype=float, maxlen=dim)/dicsize
predictions = model.predict(test_dish)*dicsize
predictions=np.around(predictions,decimals=0)
list2=predictions.tolist()
# print(list2[0])
# print(type(list2[0][0]))
list3=[]
for number in list2[0]:
    if number>0 and number<len(da.dic):
        list3.append(np.int(number))
print(da.mydecode(dic,list3))