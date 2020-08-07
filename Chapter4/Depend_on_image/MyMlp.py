import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import Chapter4.DataSolver.datasolver as da
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

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
dicsize=len(da.dic)
# Build the mlp model
# print(x_train.shape)
dim=length_input*size_input
x_train=x_train.reshape(x_train.shape[0],dim)/dicsize
x_test=x_test.reshape(x_test.shape[0],dim)/dicsize

y_train=pad_sequences(y_train,dtype=float, maxlen=length_output*3)
y_test=pad_sequences(y_test,dtype=float, maxlen=length_output*3)

# bulid the mlp
model = keras.Sequential(
[
    layers.Dense(dim,input_shape=(dim,)),
    layers.Dense(length_output, activation='relu'),
    layers.Dense(length_output*2, activation='relu'),
    layers.Dense(length_output*3, activation='softmax')
])

model.compile(optimizer='adam',
              #kullback_leibler_divergence
             loss='mean_absolute_error',
             metrics=['accuracy'])

model.summary()
# training
history=model.fit(x_train, y_train,  epochs=5, validation_split=0.1)
model.evaluate(x_test , y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

