import json
import numpy as np
import jieba
import re
from gensim.models.word2vec import Word2Vec
import tensorflow.keras as tt

# get the data from the json file

fp = open('/Users/dengjie/PycharmProjects/program/Data_StopWords/mstx.json', 'rb')
python_dict = json.load(fp)

# exchange the data into the dict. Make sure the cook steps are only 3 steps.
# There are almost 136000 chinese dishes. Extract almost 1000 dishes into the dict.
list_title=[]
list_steps=[]
list_ingredients=[]
for i in range(len(python_dict)):
    if len(python_dict[i]['steps']) !=3:
        continue
    if len(python_dict[i]['steps'][0]['content']) > 100:
        continue
    if len(python_dict[i]['steps'][1]['content']) > 100:
        continue
    if len(python_dict[i]['steps'][2]['content']) > 100:
        continue
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
        if '('  in ingredient:
            a=ingredient.split('(',1)
            list3.append(a[0])
        if '（' in ingredient:
            a= ingredient.split('（',1)
            list3.append(a[0])
        if ingredient == '':
            continue
        if ' ' in ingredient:
            a= ingredient.split(' ',1)
            list3.append(a[0]+a[1])
        if ' (' in ingredient:
            a = ingredient.split(' (', 1)
            list3.append(a[0])
        if ' （' in ingredient:
            a = ingredient.split(' （', 1)
            list3.append(a[0])
        if '(' or '（' not in ingredient:
            list3.append(ingredient)
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
    for i in range(3):
        sentence=clearStop(step[i]['content'])
        list3.append(sentence)
    list2.append(list3)
list_steps=list2

## Check the steps data one by one
# for step in list_steps:
#     print('%d : 1: %s ; 2: %s ; 3: %s ' %(i,step[0],step[1],step[2]))
#     i+=1
#     if i%10 == 0:
#         print('###################################')
#         print('')
# get the mass data index location
list=[
0   ,7,   21,   23,    25,   53,    67,   78 , 81,    89,    91,   94 ,   112 ,  121,   139,    148,   154,
161    , 171     ,173    , 192     , 199    , 200   , 201   , 205   , 206  , 212   , 213   , 215  , 224   , 225,
231    , 239,   240,       246,     249,      265,     270,     286 ,   294,   295,     300,    302,    307,    308,
313      ,323   ,324   , 328    , 329      ,330  , 341  , 343   ,347   ,350 , 359  , 362 ,363 ,368  ,
372  ,373,   385, 386,  388,   394, 399,  405,   412,   414,  416,   421,   424,  425,  428,  429,
430  ,431  ,432 ,438  ,439  ,440 ,441 ,442 ,443 ,445  ,450  ,456  ,458  ,459   ,466  ,470  , 476 ,478, 486,
488  , 489 ,  492, 493, 494 ,496 ,497, 498,  500 , 501, 502, 509,    511,513,514,516,518,519,
520, 525,526, 528,533,534,539,542,543,546,548,549,552,554,560,562,565,568,569,570,
571,572,573,577,590,593,594,598,599,600,606,609,610,614,615,617,619,620,624,625,629,
633,635,636,641,642,648,649,650,654,658,659,660,661,665,667,669,670,672,673,677,679,
680,682,684,688,694,699,706,710,711,712,713,714,719,720,721,722,728,730,732,733,734,
740,744,745,748,751,759,763,769,771,772,773,774,776,780,781,785,787,790,791,794,798,799,
802,804,805,808,809,815,817,818,820,823,825,827,830,834,835,838,839,840,842,845,846,848,
851,853,856,857,858,861,863,868,869,870,874,877,879,882,888,889,901,907,908,909,911,914,915,919,
921,922,923,925,927,928,930,931,932,935,936,939,940,944,945,948,952,954,955,956,957,
960,961,962,963,968,969,972,974,976,982,985,990,991,992,999,1000,1003,1005,1006,1007,
1010,1011,1012,1016,1019,1020,1022,1024,1031,1032,1036,1037,1038,1041,1043,1045,1048,1049,
1057,1058,1059,1063,1066,1068,1069,1074,1077,1079,1081,1084,1087,1088,1089,1093,1094,1096,1097,1098,1099,
1103,1104,1107,1109,1110,1111,1112,1113,1116,1118,1120,1122,1123,1126,1127,1128,
1133,1137,1138,1139,1142,1146,1147,1148,1150,1151,1155,1156,1158,1161,1167,1168,1169,
1170,1172,1173,1175,1176,1177,1178,1179,1180,1182,1183,1185,1186,1187,1188,1189,
1190,1191,1192,1194,1195,1196,1197,1200,1201,1202,1206,1209,1210,1211,1216,1218,1219,
1221,1222,1224,1228,1230,1232,1235,1239,1240,1241,1242,1244,1246,1247,1248,
1250,1252,1253,1255,1257,1258,1259,1261,1262,1263,1267,1268,1271,1274,1275,1277,1278,
1280,1281,1283,1284,1285,1288,1289,1290,1291,1292,1294,1296,1298,1299,1300,1301,1305,1307,1308,1309,
1310,1311,1314,1317,1319,1320,1321,1323,1326,1327,1329,1330,1331,1335,1337,1340,1341,1343,1344,1345,1346,1347,1349,
1351,1353,1354,1357,1359,1361,1362,1363,1364,1365,1366,1367,1368,1370,1371,1372,1373,1374,1376,
1380,1381,1382,1383,1388,1389,1390,1391,1392,1394,1396,1400,1401,1402,1403,1410,1411,1412,1414,1417,1418,
1420,1421,1427,1428,1430,1431,1432,1433,1437,1438,1439,1440,1442,1444,1447,1449,1450,1452,1454,1455,
1460,1461,1465,1467,1469,1471,1473,1477,1479,1481,1482,1485,1486,1487,1489,1491,1492,1496,1497,1498,1499,
1500,1502,1503,1504,1509,1511,1515,1518,1519,1521,1524,1531,1533,1535,1536,1537,1538,1539,
1540,1541,1543,1547,1550,1551,1552,1556,1558,1560,1562,1563,1564,1565,1567,1568,1569,
1571,1573,1577,1580,1582,1584,1585,1586,1588,1589,1590,1591,1592,1593,1594,1597,1599,
1604,1605,1607,1608,1609,1610,1611,1612,1619,1620,1621,1623,1626,1631,1633,1634,1636,1637,1639,
1642,1645,1648,1649,1655,1657,1658,1659,1663,1664,1665,1668,1670,1674,1676,1677,1678,
1680,1681,1683,1684,1687,1690,1691,1694,1695,1700,1702,1703,1708,1712,1716,1717,1718,
1720,1722,1725,1727,1729,1731,1733,1734,1739,1745,1747,1748,1749,1750,1751,1754,1756,1759,
1763,1764,1765,1767,1768,1769,1770,1772,1774,1776,1777,1780,1782,1784,1788,1791,1794,1797,1798,
1808,1809,1811,1812,1813,1815,1817,1818,1821,1823,1824,1825,1827,1829,1830,1831,1832,1833,1834,
1840,1841,1843,1844,1845,1847,1855,1857,1859,1860,1861,1862,1863,1864,1865,1866,1867,1868,
1870,1871,1874,1875,1877,1879,1880,1881,1882,1883,1885,1887,1890,1891,1892,1893,1894,
1900,1901,1902,1903,1905,1907,1908,1909,1911,1914,1915,1916,1918,1919,1921,1922,1924,1925,1926,1927,1928,
1931,1932,1933,1934
]
# delete these data in the steps and ingredient
for i in range(len(list)-1,-1,-1):
    list_steps.pop(list[i])
    list_ingredients.pop(list[i])

list=[
8, 10,18,32,43,
59,60,63,146,152,189,215,267,
300,305,309,311,316,322,325,343,
349,355,400,444,452,458,470,546,
601,608,639,674,936,950,1017,1042
]

for i in range(len(list)-1,-1,-1):
    list_steps.pop(list[i])
    list_ingredients.pop(list[i])

list=[
138 , 382, 474, 840, 913, 1042
]

for i in range(len(list)-1,-1,-1):
    list_steps.pop(list[i])
    list_ingredients.pop(list[i])

list=[
127,238,306,327,396,411,432,451,490,551,
579,598,615,641,661,703,740,770,825,833,
948,1083,1097,
]

for i in range(len(list)-1,-1,-1):
    list_steps.pop(list[i])
    list_ingredients.pop(list[i])

# define the word2vec model about the dataset and define the dictionary
listW1=[]
for ingredient in list_ingredients:
    listW22=[]
    for word in ingredient:
        listW22.append(word)
    listW1.append(listW22)
mysize=128
model = Word2Vec(listW1, size=mysize, window=5, min_count=1, workers=4)

dic1 = {}
for recipe in listW1:
    for word in recipe:
        dic1[word]=model[word]

dic2=[]
tok = tt.preprocessing.text.Tokenizer()
listW=[]
for steps in list_steps:
    for step in steps:
        for word in jieba.lcut(step):
            listW.append(word)
tok.fit_on_texts(listW)
dic2=tok.word_index

def myencode1(str):
    encode = []
    for words in str:
        for num in dic1[words]:
            encode.append(num)
    return encode

def myencode2(str):
    encode=[]
    for words in str:
        encode.append(dic2[words])
    return encode

def mydecode(str):
    decode = ''
    for i in str:
        for key, value in dic2.items():
            if value == i:
                decode+=key
    return decode

# build the input matrix and output matrix
# Firstly, make sure the size of input and output
row_input=0
column_input=1
for ingredient in list_ingredients:
    for item in ingredient:
        if row_input <= len(item):
            row_input = len(item)

row_output=0
column_output=3
for step in list_steps:
    for i in range(3):
        if row_output <= len(jieba.lcut(step[i])):
            row_output = len(jieba.lcut(step[i]))
row_input*=mysize
# Fianlly, build the output, input matrix
# build the input matrix

list_train=[]
for ingredient in list_ingredients:
    A = np.zeros([row_input, column_input])
    list3 = []
    for item in ingredient:
        if item != '':
            list3.append(item)
    list2 =[]
    list2 = myencode1(list3)
    for i in range(row_input - len(list2)):
        list2.append(0)
    A=list2
    list_train.append(A)

# build the output matrix
list_test=[]
for step in list_steps:
    list=[]
    for i in range(3):
        list2 = []
        list3 = jieba.lcut(step[i])
        list2=myencode2(list3)
        for num in list2:
            list.append(num)
    list_test.append(list)

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

