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


#%% xgb batch
# subsample을 도입해보자(rf의 mtry 같은 것)

s = time.time()
xgb_perfs = []
fl = ['topic_sents','total_sents','d2v','d2vpca','tx1','tx2']
mtrs = ['acc','bal_acc','kap','auc','f1','prec','recall']
for i,feat in enumerate([topic_sents,total_sents,d2v,d2vpca,tx1,tx2]):
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
#%% performance 저장
xgb_perfs = pd.concat(xgb_perfs)
#xgb_perfs.to_csv('data/xgb_vixud1_perfs.csv',index=False)        
xgb_perfs.to_csv('data/xgb_spxud1_perfs.csv',index=False)        

#%% performance 표로 만들기 
import pandas as pd
#df = pd.read_csv('data/xgb_vixud1_perfs.csv')
df = pd.read_csv('data/xgb_spxud1_perfs.csv')
df = xgb_perfs
df = df[['metric','target','lookback','feats','val']].copy()
df.sort_values(['metric','lookback'],inplace=True)
perf = df.pivot_table(index=['metric','target','lookback'],columns='feats',values='val')
mcols = ['topic_sents','total_sents','d2v','d2vpca','tx1','tx2']
perf.loc[:,mcols].to_clipboard()

from sklearn.metrics import brier_score_loss
brier_score_loss(y_true,y_score)
