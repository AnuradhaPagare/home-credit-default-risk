import numpy as np, pandas as pd, sys, itertools
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
y=pd.read_csv('input/train.csv').TARGET.values
names=sys.argv[1:]
O=np.column_stack([np.load(f'preds/oof_{n}.npy') for n in names]); T=np.column_stack([np.load(f'preds/test_{n}.npy') for n in names])
for n,o in zip(names,O.T): print(f'{n:8s} OOF AUC {roc_auc_score(y,o):.4f}')
print('corr\n',pd.DataFrame(O,columns=names).corr().round(3))
rk=lambda M:(pd.DataFrame(M).rank(pct=True)).values   # rank-normalise so scales are comparable
f=lambda w:-roc_auc_score(y,O@(np.abs(w)/np.abs(w).sum()))
r=minimize(f,np.ones(len(names))/len(names),method='Nelder-Mead',options=dict(xatol=1e-3,fatol=1e-6))
w=np.abs(r.x)/np.abs(r.x).sum(); print('weights',dict(zip(names,w.round(3))),'blend OOF AUC %.4f'%-r.fun)
print('equal-weight OOF AUC %.4f'%roc_auc_score(y,O.mean(1)))
# check weight stability: fit weights on half the rows, score on the other half
rng=np.random.RandomState(0); idx=rng.permutation(len(y)); a,b=idx[:len(y)//2],idx[len(y)//2:]
g=lambda w,ii:-roc_auc_score(y[ii],O[ii]@(np.abs(w)/np.abs(w).sum()))
wa=minimize(g,np.ones(len(names))/len(names),args=(a,),method='Nelder-Mead').x; wa=np.abs(wa)/np.abs(wa).sum()
print('weights fit on half A -> AUC on half B: %.4f (equal weights on B: %.4f, best single on B: %.4f)'%(roc_auc_score(y[b],O[b]@wa),roc_auc_score(y[b],O[b].mean(1)),max(roc_auc_score(y[b],o[b]) for o in O.T)))
np.save('preds/blend_w.npy',w); np.save('preds/blend_names.npy',np.array(names))
pred=T@w
sub=pd.read_csv('input/sample_submission.csv'); sub['TARGET']=pred
te=pd.read_csv('input/test.csv'); assert (te.SK_ID_CURR.values==sub.SK_ID_CURR.values).all()
sub.to_csv('output/submission.csv',index=False)
s=pd.read_csv('output/submission.csv'); ref=pd.read_csv('input/sample_submission.csv')
assert s.shape==ref.shape and list(s.columns)==list(ref.columns) and (s.SK_ID_CURR.values==ref.SK_ID_CURR.values).all()
assert s.TARGET.notna().all() and np.isfinite(s.TARGET).all() and s.TARGET.between(0,1).all()
print('submission OK', s.shape, 'TARGET range %.4f-%.4f mean %.4f'%(s.TARGET.min(),s.TARGET.max(),s.TARGET.mean()))
