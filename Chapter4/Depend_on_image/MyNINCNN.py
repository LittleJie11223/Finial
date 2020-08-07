import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import Chapter4.DataSolver.datasolver as da
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# get the data from the json file
list_train = da.getInput()
x_train  = np.array(list_train[0:1100])
x_test  = np.array(list_train[1100:len(list_train)+1])

list_test = da.getOutput()
y_train = np.array(list_test[0:1100])
y_test = np.array(list_test[1100:len(list_test)+1])

# get the data matrix size
list = da.getsizeIn()
length_input=list[0]
size_input=list[1]
length_output=da.getsizeOut()
length_output=length_output[0]
dicsize=len(da.dic)

# reshape the input data
x_train = x_train.reshape((-1,length_input,size_input,1))/dicsize
x_test = x_test.reshape((-1,length_input,size_input,1))/dicsize

y_train=pad_sequences(y_train,dtype=float, maxlen=length_output*3)/dicsize
y_test=pad_sequences(y_test,dtype=float, maxlen=length_output*3)/dicsize

#setting the CNN layers
mymodel = keras.Sequential(
[

    layers.Conv2D(input_shape=((x_train.shape[1], x_train.shape[2], x_train.shape[3])),
                 filters=length_output*2, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=length_output, kernel_size=(1,1), strides=(1,1), padding='valid', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(filters=length_output*2, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=length_output, kernel_size=(1,1), strides=(1,1), padding='valid', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(length_output*2, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(length_output*3, activation='softmax')

])
# setting the parm
mymodel.compile(optimizer=keras.optimizers.Adam(),
             #       loss='categorical_hinge',
             loss='mean_absolute_error',
            metrics=['accuracy'])
mymodel.summary()

# model fit
history = mymodel.fit(x_train, y_train,batch_size=32,  epochs=5, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

res = mymodel.evaluate(x_test, y_test)