import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import features as F
tr=pd.read_csv('input/train.csv'); te=pd.read_csv('input/test.csv'); y=tr.TARGET.values
X,XT,CAT=F.build(tr,te,groupstats=False)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
def ordinal(d):
    d=d.copy()
    for c in CAT: d[c]=d[c].cat.codes
    return d
def cv(name,fit_predict):
    t=time.time(); oof=np.zeros(len(X)); tp=np.zeros(len(XT)); a=[];tra=[]
    for i,j in folds:
        pv,pt,ptr=fit_predict(i,j); oof[j]=pv; tp+=pt/5; a.append(roc_auc_score(y[j],pv)); tra.append(roc_auc_score(y[i],ptr))
    print(f"{name:14s} CV {np.mean(a):.4f} +/- {np.std(a):.4f} | train {np.mean(tra):.4f} | folds {np.round(a,4).tolist()} | {time.time()-t:.0f}s",flush=True)
    np.save(f'preds/oof_{name}.npy',oof); np.save(f'preds/test_{name}.npy',tp)
m=sys.argv[1]
if m=='lr':
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    D=pd.get_dummies(pd.concat([X,XT]),columns=CAT,dummy_na=True,dtype=float)
    na=D.isnull().any(); 
    for c in D.columns[na]: D[c+'_na']=D[c].isnull().astype(float)
    D=D.clip(D.quantile(.005),D.quantile(.995),axis=1) if False else D
    A,B=D.iloc[:len(X)].reset_index(drop=True),D.iloc[len(X):].reset_index(drop=True)
    def fp(i,j):
        med=A.iloc[i].median(); lo=A.iloc[i].quantile(.01); hi=A.iloc[i].quantile(.99)
        f=lambda d: d.fillna(med).clip(lo,hi,axis=1)
        s=StandardScaler().fit(f(A.iloc[i])); g=lambda d: s.transform(f(d))
        mdl=LogisticRegression(C=0.05,max_iter=300,random_state=0).fit(g(A.iloc[i]),y[i])
        return mdl.predict_proba(g(A.iloc[j]))[:,1],mdl.predict_proba(g(B))[:,1],mdl.predict_proba(g(A.iloc[i]))[:,1]
    cv('lr',fp)
elif m in('rf','et'):
    from sklearn.ensemble import RandomForestClassifier as RF, ExtraTreesClassifier as ET
    Xo,XTo=ordinal(X).fillna(-1),ordinal(XT).fillna(-1)
    def fp(i,j):
        K=dict(n_estimators=250,min_samples_leaf=40,max_features=0.3,n_jobs=1,random_state=0)
        mdl=(RF(max_depth=14,**K) if m=='rf' else ET(**K)).fit(Xo.iloc[i],y[i])
        return mdl.predict_proba(Xo.iloc[j])[:,1],mdl.predict_proba(XTo)[:,1],mdl.predict_proba(Xo.iloc[i])[:,1]
    cv(m,fp)
elif m=='lgb':
    import lightgbm as lgb
    P=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_child_samples=100,subsample=0.8,subsample_freq=1,
           colsample_bytree=0.5,reg_lambda=5,n_estimators=3000,random_state=42,n_jobs=1,verbose=-1)
    def fp(i,j):
        mdl=lgb.LGBMClassifier(**P).fit(X.iloc[i],y[i],eval_set=[(X.iloc[j],y[j])],eval_metric='auc',callbacks=[lgb.early_stopping(100,verbose=False)])
        return mdl.predict_proba(X.iloc[j])[:,1],mdl.predict_proba(XT)[:,1],mdl.predict_proba(X.iloc[i])[:,1]
    cv('lgb',fp)
elif m=='xgb':
    import xgboost as xgb
    def fp(i,j):
        mdl=xgb.XGBClassifier(n_estimators=3000,learning_rate=0.05,max_depth=5,min_child_weight=20,subsample=0.8,colsample_bytree=0.5,
            reg_lambda=5,tree_method='hist',enable_categorical=True,eval_metric='auc',early_stopping_rounds=100,n_jobs=1,random_state=42)
        mdl.fit(X.iloc[i],y[i],eval_set=[(X.iloc[j],y[j])],verbose=False)
        return mdl.predict_proba(X.iloc[j])[:,1],mdl.predict_proba(XT)[:,1],mdl.predict_proba(X.iloc[i])[:,1]
    cv('xgb',fp)
elif m=='cat':
    from catboost import CatBoostClassifier
    Xc=X.copy(); XTc=XT.copy()
    for c in CAT: Xc[c]=Xc[c].astype(object).fillna('NA').astype(str); XTc[c]=XTc[c].astype(object).fillna('NA').astype(str)
    def fp(i,j):
        mdl=CatBoostClassifier(iterations=2500,learning_rate=0.06,depth=6,l2_leaf_reg=5,eval_metric='AUC',random_seed=42,
            thread_count=1,verbose=0,early_stopping_rounds=100,cat_features=CAT)
        mdl.fit(Xc.iloc[i],y[i],eval_set=(Xc.iloc[j],y[j]))
        return mdl.predict_proba(Xc.iloc[j])[:,1],mdl.predict_proba(XTc)[:,1],mdl.predict_proba(Xc.iloc[i])[:,1]
    cv('cat',fp)
