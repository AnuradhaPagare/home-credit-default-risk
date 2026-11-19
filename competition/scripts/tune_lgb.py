import json, numpy as np, pandas as pd, lightgbm as lgb, optuna, warnings; warnings.filterwarnings('ignore'); optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import features as F
tr=pd.read_csv('input/train.csv'); te=pd.read_csv('input/test.csv'); y=tr.TARGET.values
X,XT,CAT=F.build(tr,te,groupstats=False)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
def obj(t):
    P=dict(objective='binary',learning_rate=0.1,n_estimators=2000,num_leaves=t.suggest_int('num_leaves',6,48),
      min_child_samples=t.suggest_int('min_child_samples',40,500),subsample=t.suggest_float('subsample',0.6,1.0),subsample_freq=1,
      colsample_bytree=t.suggest_float('colsample_bytree',0.25,0.8),reg_lambda=t.suggest_float('reg_lambda',1,50,log=True),
      reg_alpha=t.suggest_float('reg_alpha',1e-3,10,log=True),min_split_gain=t.suggest_float('min_split_gain',0,0.5),
      max_bin=t.suggest_categorical('max_bin',[127,255]),cat_smooth=t.suggest_float('cat_smooth',5,50),random_state=42,n_jobs=1,verbose=-1)
    a=[];its=[]
    for k,(i,j) in enumerate(folds):
        m=lgb.LGBMClassifier(**P).fit(X.iloc[i],y[i],eval_set=[(X.iloc[j],y[j])],eval_metric='auc',callbacks=[lgb.early_stopping(50,verbose=False)])
        a.append(roc_auc_score(y[j],m.predict_proba(X.iloc[j])[:,1])); its.append(m.best_iteration_)
        t.report(np.mean(a),k)
        if k==1 and t.should_prune(): raise optuna.TrialPruned()
    t.set_user_attr('iters',float(np.mean(its))); print(f'trial {t.number} AUC {np.mean(a):.4f} iters {np.mean(its):.0f}',flush=True); return np.mean(a)
s=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=0),pruner=optuna.pruners.MedianPruner(n_startup_trials=4))
s.enqueue_trial(dict(num_leaves=31,min_child_samples=100,subsample=0.8,colsample_bytree=0.5,reg_lambda=5,reg_alpha=0.001,min_split_gain=0,max_bin=255,cat_smooth=10))
import time; t0=time.time()
s.optimize(obj,n_trials=40,timeout=1500)
b=s.best_trial; print('BEST',b.number,b.value,b.params,b.user_attrs,flush=True)
json.dump(dict(params=b.params,iters=b.user_attrs['iters'],auc=b.value),open('best_lgb.json','w'))
