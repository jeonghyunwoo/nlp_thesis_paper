# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:24:19 2020

@author: jeong
"""
#%% library
import texthero as hero
import pandas as pd
import gc, time, glob, os
#%% testing
enws = pd.read_csv('data/economy_news_전처리.csv',parse_dates=['dat'])
enws = enws.loc[enws.dat >='2016'].reset_index(drop=True)
enws.shape # 33031 9
# ~23929 (2016~2019), 23930~ (2020)
ner = hero.named_entities(enws.text[:10])

#%% ner 
enws = pd.read_csv('data/economy_news_전처리.csv',parse_dates=['dat'])
enws = enws.loc[enws.dat >='2016'].reset_index(drop=True)
enws.shape # 33031 9
# ~23929 (2016~2019), 23930~ (2020)
s = time.time()
ner = hero.named_entities(enws.text)
e = time.time()-s
time.strftime('%M:%S',time.gmtime(e))
#'23:21'
tr_ner = pd.Series([[i[0]+'/'+i[1] for i in nr] for nr in ner])[:23930]
te_ner = pd.Series([[i[0]+'/'+i[1] for i in nr] for nr in ner])[23930:]
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
tagged = [TaggedDocument(t,tags=[i]) for i,t in enumerate(tr_ner)]

model = Doc2Vec(vector_size=50, epochs=10)
model.build_vocab(tagged)
s = time.time()
model.train(tagged,total_examples=model.corpus_count, epochs=model.epochs)
e = time.time()-s
time.strftime('%M:%S',time.gmtime(e))

nerser = pd.concat([tr_ner,te_ner])
dv = [model.infer_vector(n) for n in nerser]
dvdf = pd.DataFrame(dv,columns=['ndv'+str(i+1) for i in range(50)])
dvdf.shape
nerdv = pd.concat([enws[['id','dat']],dvdf],axis=1)
nerdv.shape # 33031 52
nerdv.to_csv('data/nerdv.csv',index=False)
#%%
