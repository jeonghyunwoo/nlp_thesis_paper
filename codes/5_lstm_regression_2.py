# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:03:26 2020
원본: d:/proj/thesis_paper/lstm_take10.ipynb
@author: jeong
"""

#%% 패키지로딩 및 함수 
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import os, glob, time, gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def metr(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    rmse = mse**0.5
    return (rmse, r2)
    
#%% 데이터 가공 및 PCA fittng
df = pd.read_csv('data/lstm_data5.csv',parse_dates=['date'])    
targ = df[['date','spx','vix']].copy()
shft = [1,3,5,7]
for i in shft:
    targ['spx'+str(i)] = targ.spx.shift(-i)
    targ['vix'+str(i)] = targ.vix.shift(-i)
allcols = df.columns.tolist()
print(allcols)

# doc2vec PCA fitting
train_yrs = [[2016,2017,2018,2019]]
test_yrs = [2020]
d2v = ['v'+str(i+1) for i in range(300)]
pca = PCA(n_components=200)
trpca = df.loc[df.date.dt.year.isin(train_yrs[0]),d2v]
tepca = df.loc[df.date.dt.year==test_yrs[0],d2v]
pca.fit(trpca)
pcn = np.sum(np.where(pca.explained_variance_ratio_.cumsum()<=0.8,1,0))
print('explained variance 80% components: ',pcn)
pca = PCA(n_components=pcn)
pca.fit(trpca)
trpca = pd.DataFrame(pca.fit_transform(trpca),columns=['pc'+str(i+1) for i in range(pcn)])
tepca = pd.DataFrame(pca.transform(tepca),columns=['pc'+str(i+1) for i in range(pcn)])
dfpca = pd.concat([trpca,tepca],axis=0).reset_index(drop=True)
print('dfpca shape:',dfpca.shape)
print('df shape:',df.shape)

df_ = pd.concat([df,dfpca],axis=1)

#%% feature 종류
txvx = ['vix', 'compound_4', 'compound_7', 'compound_14', 'compound_3', 'compound_6', 'compound_5', 'compound_10', 'compound_12', 'compound_9', 'compound_2', 'compound_11', 'compound_1', 'compound_13', 'compound_8', 'polarity_4', 'polarity_7', 'polarity_14', 'polarity_3', 'polarity_6', 'polarity_5', 'polarity_10', 'polarity_12', 'polarity_9', 'polarity_2', 'polarity_11', 'polarity_1', 'polarity_13', 'polarity_8', 'neg_avg', 'neg_std', 'pos_avg', 'pos_std', 'neu_avg', 'neu_std', 'compound_avg', 'compound_std', 'polarity_avg', 'polarity_std', 'subjectivity_avg', 'subjectivity_std', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v46', 'v47', 'v48', 'v49', 'v50', 'v51', 'v52', 'v53', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v63', 'v64', 'v65', 'v66', 'v67', 'v68', 'v69', 'v70', 'v71', 'v72', 'v73', 'v74', 'v75', 'v76', 'v77', 'v78', 'v79', 'v80', 'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88', 'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v105', 'v106', 'v107', 'v108', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v123', 'v124', 'v125', 'v126', 'v127', 'v128', 'v129', 'v130', 'v131', 'v132', 'v133', 'v134', 'v135', 'v136', 'v137', 'v138', 'v139', 'v140', 'v141', 'v142', 'v143', 'v144', 'v145', 'v146', 'v147', 'v148', 'v149', 'v150', 'v151', 'v152', 'v153', 'v154', 'v155', 'v156', 'v157', 'v158', 'v159', 'v160', 'v161', 'v162', 'v163', 'v164', 'v165', 'v166', 'v167', 'v168', 'v169', 'v170', 'v171', 'v172', 'v173', 'v174', 'v175', 'v176', 'v177', 'v178', 'v179', 'v180', 'v181', 'v182', 'v183', 'v184', 'v185', 'v186', 'v187', 'v188', 'v189', 'v190', 'v191', 'v192', 'v193', 'v194', 'v195', 'v196', 'v197', 'v198', 'v199', 'v200', 'v201', 'v202', 'v203', 'v204', 'v205', 'v206', 'v207', 'v208', 'v209', 'v210', 'v211', 'v212', 'v213', 'v214', 'v215', 'v216', 'v217', 'v218', 'v219', 'v220', 'v221', 'v222', 'v223', 'v224', 'v225', 'v226', 'v227', 'v228', 'v229', 'v230', 'v231', 'v232', 'v233', 'v234', 'v235', 'v236', 'v237', 'v238', 'v239', 'v240', 'v241', 'v242', 'v243', 'v244', 'v245', 'v246', 'v247', 'v248', 'v249', 'v250', 'v251', 'v252', 'v253', 'v254', 'v255', 'v256', 'v257', 'v258', 'v259', 'v260', 'v261', 'v262', 'v263', 'v264', 'v265', 'v266', 'v267', 'v268', 'v269', 'v270', 'v271', 'v272', 'v273', 'v274', 'v275', 'v276', 'v277', 'v278', 'v279', 'v280', 'v281', 'v282', 'v283', 'v284', 'v285', 'v286', 'v287', 'v288', 'v289', 'v290', 'v291', 'v292', 'v293', 'v294', 'v295', 'v296', 'v297', 'v298', 'v299', 'v300']
vx = ['vix']
tx = txvx[1:]
v3 = ['v1','v2','v3']
cp = ['polarity_avg','compound_avg']
trf1 = ['neg_avg','v282','v136','compound_avg','v73','pos_avg','v97','v190','v134','v43']
rf10 = ['neg_avg', 'compound_avg', 'v282', 'v73', 'v170', 'v274', 'v30', 'v279', 'v249', 'v300']
xgb10 = ['v282', 'compound_avg', 'v30', 'v74', 'v170', 'v104', 'v165', 'v73', 'v216', 'v37']
vx1rf10 = ['neg_avg', 'v282', 'compound_avg', 'v73', 'compound_11', 'polarity_11', 'v136', 'v190', 'v134', 'v174']
topic_sents = ['compound_4', 'compound_7', 'compound_14', 'compound_3', 'compound_6', 'compound_5', 'compound_10', 'compound_12', 'compound_9', 'compound_2', 'compound_11', 'compound_1', 'compound_13', 'compound_8', 'polarity_4', 'polarity_7', 'polarity_14', 'polarity_3', 'polarity_6', 'polarity_5', 'polarity_10', 'polarity_12', 'polarity_9', 'polarity_2', 'polarity_11', 'polarity_1', 'polarity_13', 'polarity_8']
total_sents = ['neg_avg', 'neg_std', 'pos_avg', 'pos_std', 'neu_avg', 'neu_std', 'compound_avg', 'compound_std', 'polarity_avg', 'polarity_std', 'subjectivity_avg', 'subjectivity_std']
d2v = ['v'+str(i+1) for i in range(300)]
d2vpca = ['pc'+str(i+1) for i in range(pcn)]
tx1 = topic_sents+total_sents+d2v
tx2 = topic_sents+total_sents+d2vpca
ndv = ['ndv'+str(i+1) for i in range(50)]
tx3 = topic_sents+total_sents+ndv
tx4 = topic_sents+ndv
tx5 = tx3+d2v
tx6 = tx3+d2vpca

#%% make_trte: 모형설계 및 train, test
def make_trte(lookback = 1, fcols = topic_sents, fl_name = 'topic_sents', target = 'vix1'):
    '''
    args: lookback, fcols, fl_name, target
    '''
    train_yr = train_yrs[0]
    test_yr = test_yrs[0]
    #lookback = 1
    #fcols = topic_sents
    #fl_name = 'topic_sents'
    #target = 'vix1'
    lookback = lookback
    fcols = fcols
    fl_name = fl_name
    target = target

    print('train:',train_yr,',test:',test_yr)

    tr = df_.loc[df_.date.dt.year.isin(train_yr)]
    te = df_.loc[df_.date.dt.year == test_yr]
    tr = tr.merge(targ[['date',target]],how='left',on='date')
    te = te.merge(targ[['date',target]],how='left',on='date')
    
    tr = tr.loc[:,fcols+[target]]
    te = te.loc[:,fcols+[target]].dropna(subset=[target])
    
    feature_col = tr.columns.tolist()[:-1]
    label_col = target
    
    sc_x = MinMaxScaler(feature_range = (0,1))
    sc_y = MinMaxScaler(feature_range = (0,1))
    x_tr_scaled = sc_x.fit_transform(tr[feature_col])
    y_tr_scaled = sc_y.fit_transform(tr[label_col].values.reshape(-1,1))
    
    x_tr = []
    y_tr = []
    maxlen = len(x_tr_scaled)
    for i in range(lookback, maxlen):
        if len(fcols) > 1:
            x_tr.append(x_tr_scaled[i-lookback:i,:-1]) # x가 여러개일 때
        else:
            x_tr.append(x_tr_scaled[i-lookback:i]) # x가 vix 1개 일때만 
        y_tr.append(y_tr_scaled[i-1]) 
    x_tr, y_tr = np.array(x_tr), np.array(y_tr)
    # test data
    total_data = pd.concat([tr,te])
    inputs = total_data[len(total_data)-len(te)-lookback:]
    x_te_scaled = sc_x.transform(inputs[feature_col])
    y_te_scaled = sc_y.transform(inputs[label_col].values.reshape(-1,1))
    maxlen = len(inputs)
    x_te = []
    y_te = []
    for i in range(lookback,maxlen):
        if len(fcols) > 1:
            x_te.append(x_te_scaled[i-lookback:i,:-1]) # x가 여러개일때
        else:
            x_te.append(x_te_scaled[i-lookback:i]) # x가 vix 하나일때
        y_te.append(y_te_scaled[i-1])
    x_te, y_te = np.array(x_te), np.array(y_te)
    
    return (tr,te),(x_tr, y_tr),(x_te,y_te),(sc_x, sc_y)


#%% model_build
def model_build(fl_name,lookback,target,x_tr_shape=None):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=x_tr_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    
#    model.compile(optimizer='adam',loss='mean_squared_error')
    model.compile(optimizer='adam',loss=root_mean_squared_error)
    return model

#(tr,te),(x_tr, y_tr),(x_te, y_te),(sc_x, sc_y) = make_trte()
#tr.vix1.value_counts()
#%% test1 (target:vix1)
#fl = ['topic_sents','total_sents','d2v','d2vpca','tx1','tx2']
fl = ['ndv','tx3','tx4','tx5','tx6']
lkbk = []
feats = []
rmse = []
r2 = []
s = time.time()
for i,feat in enumerate([ndv,tx3,tx4,tx5,tx6]):
    for j in [1,3,5,7,14]:
        fcols = feat
        fl_name = fl[i]
        lookback = j
        target = 'vix1'
        (tr,te),(x_tr, y_tr),(x_te, y_te),(sc_x, sc_y) = make_trte(lookback=lookback,fcols=fcols,fl_name=fl_name,target=target)
        
        try:
            model = model_build(fl_name=fl_name,lookback=lookback,target=target,x_tr_shape=x_tr.shape[1:])
        except:
            model = model_build(fl_name=fl_name,lookback=lookback,target=target,x_tr_shape=x_tr.shape[1:])
            
        early_stop = EarlyStopping(monitor = 'val_loss', patience = 30)
        filename = 'd:/proj/thesis_paper/dl_models/{ts}_{tg}_lkbk{lb}_lstm.h5'.format(ts=fl_name,lb=lookback,tg=target)
        checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')

        history = model.fit(x_tr,y_tr,epochs=200,batch_size=30,validation_split=.2,
                            callbacks=[early_stop,checkpoint])
        
        model.load_weights(filename)
        y_pred = model.predict(x_te)
        y_pred = sc_y.inverse_transform(y_pred)
        y_true = te[target].values.reshape(-1,1)
        
        ms = metr(y_true,y_pred)
        lkbk.append(lookback)
        feats.append(fl_name)
        rmse.append(ms[0])
        r2.append(ms[1])        
        
        # test기간 plot
        fig, ax = plt.subplots(figsize=(10,4))
        plt.plot(y_true, color='green', label = 'Real VIX', ls='--')
        plt.plot(y_pred, color='red', label = 'Predicted VIX', ls='-')
        plt.xlabel('Time in days')
        plt.ylabel('Real VIX')
        plt.legend()
        plt.savefig('plot/{ts}_{tg}_lkbk{lb}_lstm_test.png'.format(ts=fl_name,tg=target,lb=lookback),dpi=500)
        # 전체기간 plot
        x_trte = np.concatenate([x_tr,x_te])
        all_y_pred = model.predict(x_trte)
        all_y_pred = sc_y.inverse_transform(all_y_pred)
        all_y_true = pd.concat([tr,te])[target].values.reshape(-1,1)
        fig, ax = plt.subplots(figsize=(10,4))
        plt.plot(all_y_true, color='green', label = 'Real VIX', ls='--')
        plt.plot(all_y_pred, color='red', label = 'Predicted VIX', ls='-')
        plt.xlabel('Time in days')
        plt.ylabel('Real VIX')
        plt.axvline(1419,ls='-',color='grey')
        plt.legend()
        plt.savefig('plot/{ts}_{tg}_lkbk{lb}_lstm_total.png'.format(ts=fl_name,tg=target,lb=lookback),dpi=500)
        
        print('{ts},{tg},{lb} rmse:{m1}, r2:{m2}'.format(ts=fl_name,tg=target,lb=lookback,m1=ms[0],m2=ms[1]))
        gc.collect()
e = time.time()-s        
time.strftime('%H:%M:%S',time.gmtime(e))
#'00:21:29'
#%%
perfs = dict(feature=feats,
             target='vix1',
             lookback=lkbk,
             rmse=rmse,
             r2 = r2)
perfs = pd.DataFrame(perfs)
perfs
sns.boxplot('feature','rmse',data=perfs)
perfs.to_csv('data/vix1_ner_perfs.csv')    


#%% test2 - spx1 target
#fl = ['topic_sents','total_sents','d2v','d2vpca','tx1','tx2']
fl = ['ndv','tx3','tx4','tx5','tx6']
lkbk = []
feats = []
rmse = []
r2 = []
s = time.time()
for i,feat in enumerate([ndv,tx3,tx4,tx5,tx6]):
    for j in [1,3,5,7,14]:
        fcols = feat
        fl_name = fl[i]
        lookback = j
        target = 'spx1'
        (tr,te),(x_tr, y_tr),(x_te, y_te),(sc_x, sc_y) = make_trte(lookback=lookback,fcols=fcols,fl_name=fl_name,target=target)
        
        try:
            model = model_build(fl_name=fl_name,lookback=lookback,target=target,x_tr_shape=x_tr.shape[1:])
        except:
            model = model_build(fl_name=fl_name,lookback=lookback,target=target,x_tr_shape=x_tr.shape[1:])
            
        early_stop = EarlyStopping(monitor = 'val_loss', patience = 30)
        filename = 'd:/proj/thesis_paper/dl_models/{ts}_{tg}_lkbk{lb}_lstm.h5'.format(ts=fl_name,lb=lookback,tg=target)
        checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')

        history = model.fit(x_tr,y_tr,epochs=200,batch_size=30,validation_split=.2,
                            callbacks=[early_stop,checkpoint])
        
        model.load_weights(filename)
        y_pred = model.predict(x_te)
        y_pred = sc_y.inverse_transform(y_pred)
        y_true = te[target].values.reshape(-1,1)
        
        ms = metr(y_true,y_pred)
        lkbk.append(lookback)
        feats.append(fl_name)
        rmse.append(ms[0])
        r2.append(ms[1])        
        
        # test기간 plot
        fig, ax = plt.subplots(figsize=(10,4))
        plt.plot(y_true, color='green', label = 'Real S&P Index', ls='--')
        plt.plot(y_pred, color='red', label = 'Predicted S&P Index', ls='-')
        plt.xlabel('Time in days')
        plt.ylabel('Real S&P Index')
        plt.legend()
        plt.savefig('d:/proj/thesis_paper/plot/{ts}_{tg}_lkbk{lb}_lstm_test.png'.format(ts=fl_name,tg=target,lb=lookback),dpi=500)
        # 전체기간 plot
        x_trte = np.concatenate([x_tr,x_te])
        all_y_pred = model.predict(x_trte)
        all_y_pred = sc_y.inverse_transform(all_y_pred)
        all_y_true = pd.concat([tr,te])[target].values.reshape(-1,1)
        fig, ax = plt.subplots(figsize=(10,4))
        plt.plot(all_y_true, color='green', label = 'Real S&P Index', ls='--')
        plt.plot(all_y_pred, color='red', label = 'Predicted S&P Index', ls='-')
        plt.xlabel('Time in days')
        plt.ylabel('Real S&P Index')
        plt.axvline(1419,ls='-',color='grey')
        plt.legend()
        plt.savefig('d:/proj/thesis_paper/plot/{ts}_{tg}_lkbk{lb}_lstm_total.png'.format(ts=fl_name,tg=target,lb=lookback),dpi=500)
        
        print('{ts},{tg},{lb} rmse:{m1}, r2:{m2}'.format(ts=fl_name,tg=target,lb=lookback,m1=ms[0],m2=ms[1]))
        gc.collect()
        
e = time.time() - s
time.strftime('%H:%M:%S',time.gmtime(e))        
#'28:41'
#%%
perfs3 = dict(feature=feats,
             target='spx1',
             lookback=lkbk,
             rmse=rmse,
             r2 = r2)
perfs3 = pd.DataFrame(perfs3)
perfs3
fig,ax = plt.subplots(figsize=(10,4))
sns.boxplot('feature','rmse',data=perfs3,ax=ax)
perfs3.to_csv('d:/proj/thesis_paper/data/spx1_ner_perfs.csv')            

#%% spx bind
path = 'd:/proj/thesis_paper/data/'
spx1perf = pd.concat([pd.read_csv(path+x) for x in ['spx1_perfs.csv','spx1_ner_perfs.csv']])
spx1perf.drop('Unnamed: 0',axis=1,inplace=True)
spx1perf = spx1perf.loc[~spx1perf.feature.str.contains('_v')].copy()
spx1perf.pivot_table(index=['target','lookback'],columns='feature',values='rmse').to_clipboard()

#%% custom predict
(tr,te),(x_tr, y_tr),(x_te, y_te),(sc_x, sc_y) = make_trte(lookback=14,fcols=tx4,fl_name='comb4',target='vix1')
model = model_build(fl_name='comb4',lookback=14,target='vix1',x_tr_shape=x_tr.shape[1:])
model.load_weights('d:/proj/thesis_paper/dl_models/tx4_vix1_lkbk14_lstm.h5')
y_pred = model.predict(x_te)
y_pred = sc_y.inverse_transform(y_pred)
y_true = te['vix1'].values.reshape(-1,1)
preds = pd.DataFrame({'y_pred':y_pred.reshape(-1),'y_true':y_true.reshape(-1)})
preds.to_csv('논문작성/data/vix1_test_preds.csv',index=False)
# 전체
x_trte = np.concatenate([x_tr,x_te])
all_y_pred = model.predict(x_trte)
all_y_pred = sc_y.inverse_transform(all_y_pred)
all_y_true = pd.concat([tr,te])['vix1'].values.reshape(-1,1)
preds = pd.DataFrame({'y_pred':all_y_pred.reshape(-1),'y_true':all_y_true.reshape(-1)})
preds.to_csv('논문작성/data/vix1_test_preds.csv',index=False)