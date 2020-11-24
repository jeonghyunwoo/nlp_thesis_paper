# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:53:48 2020

@author: jeong
"""
#%%
import pandas as pd
import numpy as np
import os, glob, time, gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

#%% 데이터 가공 및 PCA fittng
df = pd.read_csv('d:/proj/thesis_paper/data/lstm_data5.csv',parse_dates=['date'])    
targ = df[['date','spx','vix']].copy()
shft = [1,3,5,7]
for i in shft:
    targ['spx'+str(i)] = targ.spx.shift(-i)
    targ['vix'+str(i)] = targ.vix.shift(-i)
targ['vixud1'] = np.where(targ.vix1>targ.vix,1,0)
targ['spxud1'] = np.where(targ.spx1>targ.spx,1,0)
allcols = df.columns.tolist()
print(allcols)
#targ.groupby(targ.date.dt.year)['vixud1'].value_counts().unstack(level=1).to_clipboard()
#targ.groupby(targ.date.dt.year)['spxud1'].value_counts().unstack(level=1).to_clipboard()


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

def metr(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    rmse = mse**0.5
    return (rmse, r2)
def clmetr(y_true,y_pred,y_score):
    '''
    [acc,bal_acc,kappa,auc,f1,prec,recall]
    '''
#    y_score = model.predict(x_test).reshape(-1)
#    pred = pd.Series(model.predict_classes(x_test).reshape(-1))
#    truth = pd.Series(y_test.reshape(-1))
    acc = accuracy_score(y_true,y_pred)
    bal_acc = balanced_accuracy_score(y_true,y_pred)
    kappa = cohen_kappa_score(y_true,y_pred)
    auc = roc_auc_score(y_true,y_score)
    f1 = f1_score(y_true,y_pred)    
    prec = precision_score(y_true,y_pred)
    recall = recall_score(y_true, y_pred)
    return [acc,bal_acc,kappa,auc,f1,prec,recall]
def make_trte_cl(lookback = 1, fcols = topic_sents, fl_name = 'topic_sents', target = 'vixud1'):
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

#    print('train:',train_yr,',test:',test_yr)

    tr = df_.loc[df_.date.dt.year.isin(train_yr)]
    te = df_.loc[df_.date.dt.year == test_yr]
    tr = tr.merge(targ[['date',target]],how='left',on='date')
    te = te.merge(targ[['date',target]],how='left',on='date')
    
    tr = tr.loc[:,fcols+[target]]
    te = te.loc[:,fcols+[target]].dropna(subset=[target])
    
    feature_col = tr.columns.tolist()[:-1]
    label_col = target
    
    sc_x = MinMaxScaler(feature_range = (0,1))
    x_tr_scaled = sc_x.fit_transform(tr[feature_col])
    ytr = tr[label_col].values.reshape(-1,1)
    
    x_tr = []
    y_tr = []
    maxlen = len(x_tr_scaled)
    for i in range(lookback, maxlen):
        if len(fcols) > 1:
            x_tr.append(x_tr_scaled[i-lookback:i,:-1]) # x가 여러개일 때
        else:
            x_tr.append(x_tr_scaled[i-lookback:i]) # x가 vix 1개 일때만 
        y_tr.append(ytr[i-1]) # x가 a:b 인 경우 x의 마지막 값은 b-1까지이므로 y도 같은 위치의 값을 가져오기 위해 -1을 해준다 
    x_tr, y_tr = np.array(x_tr), np.array(y_tr)
    # test data
    total_data = pd.concat([tr,te])
    inputs = total_data[len(total_data)-len(te)-lookback:]
    x_te_scaled = sc_x.transform(inputs[feature_col])
    yte = inputs[label_col].values.reshape(-1,1)

    maxlen = len(inputs)
    x_te = []
    y_te = []
    for i in range(lookback,maxlen):
        if len(fcols) > 1:
            x_te.append(x_te_scaled[i-lookback:i,:-1]) # x가 여러개일때
        else:
            x_te.append(x_te_scaled[i-lookback:i]) # x가 vix 하나일때
        y_te.append(yte[i-1])
    x_te, y_te = np.array(x_te), np.array(y_te)
    
    return (tr,te),(x_tr, y_tr),(x_te,y_te),(sc_x)

#%%
(tr,te),(x_tr, y_tr),(x_te,y_te),(sc_x) = make_trte_cl(target='vixud1')
tr['gb'] = 'train'
te['gb'] = 'test'
pd.concat([tr,te]).to_csv('data/cl_vixud_trte.csv',index=False)
pd.concat([tr,te]).to_csv('data/cl_spxud_trte.csv',index=False)

#%% vixud1 classification
s = time.time()
xgb_perfs = []
fl = ['ndv','tx3','tx4','tx5','tx6']
mtrs = ['acc','bal_acc','kap','auc','f1','prec','recall']
for i,feat in enumerate([ndv,tx3,tx4,tx5,tx6]):
    for j in [1,3,5,7,14]:
        fcols = feat
        fl_name = fl[i]
        lookback = j
#        target = 'spxud1' 
        target = 'vixud1'
        (tr,te),(x_tr, y_tr),(x_te, y_te),(sc_x) = make_trte(lookback=lookback,fcols=fcols,fl_name=fl_name,target=target)
        
        dim1, dim2 = x_tr.shape[1],x_tr.shape[2]
        
        xgb_xtr = x_tr.reshape(-1,dim1*dim2)
        xgb_ytr = y_tr.reshape(-1)
        xgb_xte = x_te.reshape(-1,dim1*dim2)
        xgb_yte = y_te.reshape(-1)

        xgb = XGBClassifier(n_estimators=300,learning_rate=0.1,subsample=.5,n_jobs=-1,random_state=8)
        xgb.fit(xgb_xtr,xgb_ytr)
        xgb.score(xgb_xtr,xgb_ytr)
        y_pred = xgb.predict(xgb_xte)
        y_score = xgb.predict_proba(xgb_xte)[:,1]
        meas = clmetr(xgb_yte,y_pred,y_score)
        meas = pd.DataFrame(dict(metric=mtrs,val=meas))
        meas['target']=target
        meas['feats']=fl_name
        meas['lookback']=lookback
        xgb_perfs.append(meas)
        print(fl_name,str(j),' done')
e = time.time() - s
time.strftime('%H:%M:%S',time.gmtime(e))
gc.collect()
#%% performance 저장
path = 'd:/proj/thesis_paper/data/'
xgb_perfs = pd.concat(xgb_perfs)
#xgb_perfs.to_csv('data/xgb_vixud1_perfs.csv',index=False)        
xgb_perfs.to_csv(path+'xgb_vixud1_ner_perfs.csv',index=False)        
#%%
xgbperf = [pd.read_csv(path+x) for x in ['xgb_vixud1_perfs.csv','xgb_vixud1_ner_perfs.csv']]
xgbperf = pd.concat(xgbperf)
xgbperf.pivot_table(index=['target','metric','lookback'],
                    columns='feats',values='val').to_clipboard()
xgbperf.groupby(['metric'])['val'].max().to_clipboard()


#%% spxud1 classification
s = time.time()
xgb_perfs = []
fl = ['ndv','tx3','tx4','tx5','tx6']
mtrs = ['acc','bal_acc','kap','auc','f1','prec','recall']
for i,feat in enumerate([ndv,tx3,tx4,tx5,tx6]):
    for j in [1,3,5,7,14]:
        fcols = feat
        fl_name = fl[i]
        lookback = j
        target = 'spxud1'
        (tr,te),(x_tr, y_tr),(x_te, y_te),(sc_x) = make_trte(lookback=lookback,fcols=fcols,fl_name=fl_name,target=target)
        
        dim1, dim2 = x_tr.shape[1],x_tr.shape[2]
        
        xgb_xtr = x_tr.reshape(-1,dim1*dim2)
        xgb_ytr = y_tr.reshape(-1)
        xgb_xte = x_te.reshape(-1,dim1*dim2)
        xgb_yte = y_te.reshape(-1)

        xgb = XGBClassifier(n_estimators=300,learning_rate=0.1,subsample=.5,n_jobs=-1,random_state=8)
        xgb.fit(xgb_xtr,xgb_ytr)
        xgb.score(xgb_xtr,xgb_ytr)
        y_pred = xgb.predict(xgb_xte)
        y_score = xgb.predict_proba(xgb_xte)[:,1]
        meas = clmetr(xgb_yte,y_pred,y_score)
        meas = pd.DataFrame(dict(metric=mtrs,val=meas))
        meas['target']=target
        meas['feats']=fl_name
        meas['lookback']=lookback
        xgb_perfs.append(meas)
        print(fl_name,str(j),' done')
e = time.time() - s
time.strftime('%H:%M:%S',time.gmtime(e))
gc.collect()
#%% performance 저장
path = 'd:/proj/thesis_paper/data/'
xgb_perfs = pd.concat(xgb_perfs)
xgb_perfs.to_csv(path+'xgb_spxud1_ner_perfs.csv',index=False)        
#%%
xgbspxperf = [pd.read_csv(path+x) for x in ['xgb_spxud1_perfs.csv','xgb_spxud1_ner_perfs.csv']]
xgbspxperf = pd.concat(xgbspxperf)
xgbspxperf.pivot_table(index=['target','metric','lookback'],
                    columns='feats',values='val').to_clipboard()
xgbspxperf.groupby(['metric'])['val'].max().to_clipboard()
