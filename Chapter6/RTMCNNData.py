import json
import numpy as np
import jieba
import re
import tensorflow.keras as tt
import gensim

# get the data from the json file

fp = open('/Users/dengjie/PycharmProjects/program/Data_StopWords/mstx.json', 'rb')
python_dict = json.load(fp)

# exchange the data into the dict. Make sure the cook steps are only 3 steps.
# There are almost 136000 chinese dishes. Extract almost 1000 dishes into the dict.
list_title=[]
list_steps=[]
list_ingredients=[]
for i in range(len(python_dict)):
    list_title.append(python_dict[i]['title'])
    list_steps.append(python_dict[i]['steps'])
    list_ingredients.append(python_dict[i]['ingredients'])


# clear the data: delete the stop words
# Firstly, put the ingredients as a list and delete the content in the ()
list2=[]
for keys in list_ingredients:
    list3=[]
    str=keys.keys()
    for ingredient in str:
        list3.append(ingredient)
    list2.append(list3)
list_ingredients=list2

list2=[]
for ingredients in list_ingredients:
    list3=[]
    for ingredient in ingredients:
        if '(' in ingredient:
            a = ingredient.split('(', 1)
            list3.append(a[0])
            continue
        if '（' in ingredient:
            a = ingredient.split('（', 1)
            list3.append(a[0])
            continue
        if ingredient == '':
            continue
        if ' ' in ingredient:
            a = ingredient.split(' ', 1)
            list3.append(a[0] + a[1])
            continue
        if ' (' in ingredient:
            a = ingredient.split(' (', 1)
            list3.append(a[0])
            continue
        if ' （' in ingredient:
            a = ingredient.split(' （', 1)
            list3.append(a[0])
            continue
        if '（' in ingredient:
            a = ingredient.split('（', 1)
            list3.append(a[0])
            continue
        if '调料' in ingredient:
            a = ingredient.split('调料', 1)
            a = a[0]
            a = a.split('（', 1)
            list3.append(a[0])
            continue
        if '腌料' in ingredient:
            a = ingredient.split('腌料', 1)
            a = a[0]
            a = a.split('（', 1)
            list3.append(a[0])
            continue
        if '(' or '（' not in ingredient:
            list3.append(ingredient)
            continue
    list2.append(list3)
list_ingredients=list2


# Secondly, clear the stop words in the list_steps and list_ingredients
# import the stop words txt file
stopwords_path = "/Users/dengjie/PycharmProjects/program/Data_StopWords/stopwords.txt"
# make the stop words list
def stopwordslist():
    stopwords = [line.strip() for line in open(stopwords_path, encoding='UTF-8').readlines()]
    return stopwords

# Part-of-speech sentences
def seg_depart(sentence):
    sentence_depart = jieba.cut(sentence.strip())
    stopwords = stopwordslist()
    outstr = ''
    # delete the stop words in the sentences
    for word in sentence_depart:
        if word not in stopwords and len(word) > 1:
            outstr += word
            outstr += " "
    return outstr

# define stop words function
def clearStop(sentence):
    sentence = seg_depart(sentence)
    strinfo = re.compile(' ')
    sentence = strinfo.sub('', sentence)
    return sentence

# delete the stop words in the steps
list2=[]
for step in list_steps:
    list3=[]
    for i in range(len(step)):
        sentence=clearStop(step[i]['content'])
        list3.append(sentence)
    list2.append(list3)
list_steps=list2

z=0
list=[]
for steps in list_steps:
    if steps == '' or steps == []:
        list.append(i)
    z+=1

for i in range(len(list)-1,-1,-1):
    list_steps.pop(list[i])
    list_ingredients.pop(list[i])
# define the dictionary function about the dataset
def mydictionary(list1):
    tok = tt.preprocessing.text.Tokenizer()
    listW=[]

    for steps in list1:
        for step in steps:
            for word in jieba.lcut(step):
                listW.append(word)
    tok.fit_on_texts(listW)
    dic=tok.word_index
    print(listW)
    return dic
dic2 = mydictionary(list_steps)

listW=[]
for steps in list_steps:
    listW11=[]
    for step in steps:
        for word in jieba.cut(step):
            listW11.append(word)
    listW.append(listW11)

mysize=128
model = gensim.models.Word2Vec.load('/Users/dengjie/PycharmProjects/program/Chapter6/MCNNMM_Ingredient_Word2vec')

dic1 = {}
for recipe in listW:
    for word in recipe:
        if word in model.wv.vocab:
            dic1[word]=model[word]

# Use the dictionary to encode and decode
def myencode1(str):
    encode = []
    if str in dic1:
        for num in dic1[str]:
            encode.append(num)
        encode = np.array(encode)
    return encode

def myencode2(str):
    encode = []
    for i in str:
        if i in dic2.keys():
            encode.append(dic2[i])
    return encode

def mydecode(str):
    decode = []
    for i in str:
        for key, value in dic2.items():
            if value == i:
                decode.append(key)
    listi=[]
    for i in range(len(decode)-1):
        if decode[i] == decode[i+1]:
            listi.append(i)
    for i in range(len(listi)-1,-1,-1):
        decode.pop(listi[i])
    mystr=''
    for str in decode:
        mystr+=str
    return mystr

# build the input matrix and output matrix
# Firstly, make sure the size of input and output
row_input=mysize
column_input=1
for ingredient in list_ingredients:
    if column_input <= len(ingredient):
        column_input = len(ingredient)

row_output=100
column_output=1

# Fianlly, build the output, input matrix
# build the input matrix

list_train=[]
for ingredient in list_ingredients:
    A = np.zeros([row_input, column_input])
    list3 = []
    for item in ingredient:
        if item != '':
            list3.append(item)
    if list3 != []:
        listt=[]
        for str in list3:
            if myencode1(str) != []:
                listt.append(myencode1(str))

        if listt!=[]:
            for i in range(len(listt)):
                A[:,i]=np.array(listt[i])
    list_train.append(A)

# build the output matrix

list_test=[]
for step in list_steps:
    list=[]
    for i in range(len(step)):
        list2=[]
        list2=myencode2(jieba.lcut(step[i]))
        for num in list2:
            list.append(num)
    if len(list)>row_output:
        for i in range(len(list)-1,row_output,-1):
            list.pop(i)
    else:
        for i in range(1,row_output-len(list)+1):
            list.append(0)
    list=np.array(list)
    list_test.append(list)

# delete the 0 matrix in the list_test
z=0
listde=[]
for tests in list_test:
    tests=np.array(tests)
    flag = True
    for num in tests:
        if num != 0:
            flag = False
    if flag:
        listde.append(z)
    z+=1

for i in range(len(listde)-1,-1,-1):
    list_train.pop(listde[i])
    list_test.pop(listde[i])

# delete the 0 matrix in the list_train
z=0
listdet=[]
for trains in list_train:
    trains=np.array(trains)
    flag = False
    a=(trains==np.zeros([row_input,column_input])).all()
    if a:
        flag=True
    if flag:
        listdet.append(z)
    z+=1

for i in range(len(listdet)-1,-1,-1):
    list_train.pop(listdet[i])
    list_test.pop(listdet[i])

def getInput():
    return list_train

def getOutput():
    return list_test

def getsizeIn():
    list = [row_input,column_input]
    return list

def getsizeOut():
    list = [row_output, column_output]
    return list