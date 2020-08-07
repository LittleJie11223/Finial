
import requests
from bs4 import BeautifulSoup
import re


class xiachufang():

    def __init__(self):
        self.count = 1  # account
        self.comp = re.compile('[^A-^a-z^0-9^\u4e00-\u9fa5]')  # delete the special word sign
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36',
            'referer': 'http://www.xiachufang.com/category/52107/'
        }
    def get_url(self):
        while self.count <= 100:
            url = 'http://www.xiachufang.com/category/52107/?page='+str(self.count)  # page url
            self.count += 1
            re = requests.get(url, headers=self.headers)
            if re.status_code == 200:
                page_data = BeautifulSoup(re.text, 'html.parser')
                yield page_data
            else:
                print('link fail！')
    def get_data(self,page_data):
        Menu_table = page_data.find('div', class_="normal-recipe-list").findAll("div",class_="info pure-u")
        for index,meun in enumerate(Menu_table):
            tag_a = meun.find('a')
            foods_name = (tag_a.text[17:-13])
            foods_name = self.comp.sub('', foods_name)  # delete the sign with the title
            Foodstuff = meun.find('p', class_="ing ellipsis").text[1:-1]
            foods_url = r'http://www.xiachufang.com/'+tag_a['href']
            foods = (self.count, index, foods_name, Foodstuff, foods_url)
            yield foods

    def Save_foods(self, foods):
        food_name = ('%s-%s-%s'%((foods[0]-1), foods[1], foods[2]))
        food_content = 'food ：'+foods[3]+'\n link：'+foods[4]

        path = r'./data/'


        with open('./data/'+food_name+'.txt', 'w', encoding='utf-8') as f:
            f.write(food_content)
            print('download %s success' % food_name)

    def fun(self):
        for page_data in self.get_url():
            for foods in self.get_data(page_data):
                self.Save_foods(foods)

if __name__=='__main__':
    x = xiachufang()
    x.fun()