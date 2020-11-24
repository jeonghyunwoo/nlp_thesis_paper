# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:50:52 2020

@author: jeong
"""
#%% 라이브러리 준비
import pandas as pd
import numpy as np
from selenium import webdriver
import pyautogui
from bs4 import BeautifulSoup as soup
from newspaper import Article
import gc, re, time
# 뉴스읽어오기 함수 
def nws(link):
    nw = Article(link)
    try:
        nw.download();nw.parse()
        news = nw.text
    except:
        news = ""
    return news

# 1314~1853
#%% 링크만 읽어오기 
# 2018.4~2017.11
links = []
driver = webdriver.Chrome() 
for p in range(1326,1564):
    s = time.time()
    i = p+1
    url = 'https://www.investing.com/news/economy/'+str(i)
    driver.get(url)
    html = driver.page_source
    a = soup(html,'lxml')
    print('html get')
    tit_ref = a.find_all('div',class_='textDiv')
    press_date = a.find_all('span','articleDetails')
    tit, ref, press, dates = [],[],[],[]
    for j in range(len(press_date)):
        try:
            dat_ = press_date[j].find('span','date').get_text().replace('\xa0-\xa0','')
            pres_ = press_date[j].find('span').get_text()
            tit_ = tit_ref[j].find('a','title')['title']
            ref_ = 'https://investing.com'+tit_ref[j].find('a','title')['href']
            if re.findall('economy',ref_) != []:
                dates.append(dat_)
                press.append(pres_)
                tit.append(tit_)
                ref.append(ref_)
            else:
                pass
        except:
            pass
    info = pd.DataFrame(dict(title=tit,press=press,date=dates,href=ref))
    links.append(info)
    print('page '+str(i)+' complete')
    
#    news = [nws(r) for r in ref]   
driver.close()
lk0 = pd.concat(links)
lk1 = lk0.drop_duplicates(subset=['date','href']).reset_index(drop=True)
lk2 = lk1.loc[~lk1.date.str.contains('2020'),:]
lk2.to_pickle('data/link_1326_1564.pkl')
del lk0,lk1
print('link 수집 완료')

#%% 링크합치기 
import glob
#fl = glob.glob('data/link_*');fl
#lks = pd.concat([pd.read_pickle(f) for f in fl])
# 중복 링크 제거 
#links = lks.drop_duplicates(subset=['href']).reset_index(drop=True)
#lks.shape # 9067
#links.shape # 8651
#links.to_pickle('data/combined_links.pkl')

lks = pd.read_pickle('data/link_1326_1564.pkl')
links = lks.drop_duplicates(subset=['href']).reset_index(drop=True)
links.to_pickle('data/links_201711_201804.pkl')

#%% 링크에서 뉴스추출  
#links = pd.read_pickle('data/combined_links.pkl')
links = pd.read_pickle('data/links_201711_201804.pkl')
dats = pd.to_datetime(links.date)
links = links.loc[dats.between('2017-11','2018-0567')]
totn = links.shape[0]
news = []
for i,r in enumerate(links['href']):
    i += 1
    s = time.time()
    content = nws(r)
    news.append(content)
    time.sleep(i % 2 + 1)
    rst = 'ok' if content!='' else 'not ok'
    e = time.time() - s
    elaps = time.strftime("%M:%S",time.gmtime(e))
    print(str(i),'/',str(totn),rst,'(',elaps,')')
print('all news fetch done!!!')    
links['text'] = news
#links.head()
#links.tail()
links.to_csv('data/econ_nws_add8.csv',index=False)


