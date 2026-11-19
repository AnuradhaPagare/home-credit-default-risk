import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import features as F
tr=pd.read_csv('input/train.csv'); te=pd.read_csv('input/test.csv'); y=tr.TARGET.values
X,XT,CAT=F.build(tr,te,groupstats=False); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
b=json.load(open('best_lgb.json')); n=int(b['iters']*0.1/0.03)
P=dict(objective='binary',learning_rate=0.03,n_estimators=n,subsample_freq=1,random_state=42,n_jobs=1,verbose=-1,**b['params'])
oof=np.zeros(len(X)); tp=np.zeros(len(XT)); a=[];tra=[]
for i,j in folds:
    m=lgb.LGBMClassifier(**P).fit(X.iloc[i],y[i])      # fixed rounds, validation fold NOT used for stopping
    oof[j]=m.predict_proba(X.iloc[j])[:,1]; tp+=m.predict_proba(XT)[:,1]/5; a.append(roc_auc_score(y[j],oof[j])); tra.append(roc_auc_score(y[i],m.predict_proba(X.iloc[i])[:,1]))
    imp=m.booster_.feature_importance('gain'); 
    fi=imp if 'fi' not in dir() else fi+imp
print(f'lgbtuned fixed-rounds({n}) CV {np.mean(a):.4f} +/- {np.std(a):.4f} | train {np.mean(tra):.4f} | folds {np.round(a,4).tolist()}',flush=True)
np.save('preds/oof_lgbt.npy',oof); np.save('preds/test_lgbt.npy',tp)
pd.Series(fi/5,index=X.columns).sort_values(ascending=False).to_csv('feature_importance_gain.csv')
