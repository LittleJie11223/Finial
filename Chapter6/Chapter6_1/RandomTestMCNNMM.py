from translate import Translator
import numpy as np
import tensorflow as tf
import random as ra
import tensorflow.keras as tt
import Chapter6.RTMCNNData as rta


list = rta.getsizeIn()
length_input=list[0]
size_input=list[1]
length_output=rta.getsizeOut()
length_output=length_output[0]
dicsize=len(rta.dic2)

model = tf.keras.models.load_model('/Users/dengjie/PycharmProjects/program/Chapter6/Chapter6_1/myMCNNMM_model.h5')

def getDish(n):
    test_dish=[]
    # Get ingredients dictionary
    list_ingredients=rta.list_ingredients
    dic_ingredient = []
    for ingredients in list_ingredients:
        for item in ingredients:
            if item in rta.dic1:
                dic_ingredient.append(item)
    tok = tt.preprocessing.text.Tokenizer()
    tok.fit_on_texts(dic_ingredient)
    dic_ingredient = tok.word_index
    # shift the dic
    dic_ingredient = dict([val, key] for key, val in dic_ingredient.items())
    max=rta.column_input

    for i in range(0,n):
        num=ra.randint(4,max)
        list_i=[]
        for k in range(num):
            list_i.append(dic_ingredient[ra.randint(1,len(dic_ingredient)-1)])
        test_dish.append(list_i)

    return test_dish

def myTencode(test_dish):
    list=[]
    for ingredients in test_dish:
        A=np.zeros([rta.row_input, rta.column_input])
        listt=[]
        for item in ingredients:
            listt.append(rta.myencode1(item))
        if listt!=[]:
            for i in range(len(listt)):
                A[:, i]=np.array(listt[i])
        list.append(A)
    test_dish=np.array(list)
    return test_dish

def myPredictions(test_dish):
    predictions=model.predict(test_dish)*dicsize
    predictions = np.around(predictions,decimals=0)
    predictions=predictions.tolist()
    listp=[]
    for l in predictions:
        list1=[]
        for num in l:
            if num > 0 and num < dicsize:
                list1.append(np.int(num))
        listd=[]
        for i in range(0,len(list1)-1):
            if list1[i] == list1[i+1]:
                listd.append(i)
        if listd!=[]:
            for i in range(len(listd)-1,-1,-1):
                list1.pop(listd[i])
        mystr=rta.mydecode(list1)
        mystr=mystr[0:50]
        listp.append(mystr)

    return listp



n=100
test_dish=getDish(n)
test=myTencode(test_dish)
test=test.reshape((-1,length_input,size_input,1))
pred=myPredictions(test)
translator= Translator(from_lang="chinese",to_lang="english")
f=open('/Users/dengjie/PycharmProjects/program/Chapter6/Chapter6_1/MCNNMMRT.txt','w')
for i in range(n):
    f.write(str(i))
    f.write('\n')
    str1=''
    for item in test_dish[i]:
        str1+=item
        str1+=','
    f.write(str1)
    f.write('\n')
    f.write(translator.translate(str1))
    f.write('\n')
    f.write(pred[i])
    f.write('\n')
    f.write(translator.translate(pred[i]))
    f.write('\n')
    f.write('Whether the result is valid:')
    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write('######################')
    f.write('\n')
f.close()
