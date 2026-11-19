"""Feature engineering for Home Credit default risk. No target is used anywhere in this file,
and group statistics are fitted on TRAIN only and then mapped onto test."""
import numpy as np, pandas as pd

CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","ORGANIZATION_TYPE"]
EXT = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]
GROUPS = ["ORGANIZATION_TYPE","OCCUPATION_TYPE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE"]

def _safe_div(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def basic_clean(d):
    d = d.copy()
    d["DAYS_EMPLOYED_ANOM"] = (d.DAYS_EMPLOYED == 365243).astype(int)
    d.loc[d.DAYS_EMPLOYED == 365243, "DAYS_EMPLOYED"] = np.nan
    d.loc[d.ORGANIZATION_TYPE == "XNA", "ORGANIZATION_TYPE"] = np.nan
    d.loc[d.CODE_GENDER == "XNA", "CODE_GENDER"] = np.nan
    return d

def add_features(d):
    d = basic_clean(d)
    f = pd.DataFrame(index=d.index)
    f["AGE_YEARS"] = -d.DAYS_BIRTH / 365.25
    f["EMPLOYED_YEARS"] = -d.DAYS_EMPLOYED / 365.25
    f["EMPLOYED_RATIO"] = _safe_div(d.DAYS_EMPLOYED, d.DAYS_BIRTH)
    f["REG_RATIO"] = _safe_div(d.DAYS_REGISTRATION, d.DAYS_BIRTH)
    f["IDPUB_RATIO"] = _safe_div(d.DAYS_ID_PUBLISH, d.DAYS_BIRTH)
    f["PHONE_RATIO"] = _safe_div(d.DAYS_LAST_PHONE_CHANGE, d.DAYS_BIRTH)
    f["CAR_AGE_RATIO"] = _safe_div(d.OWN_CAR_AGE, -d.DAYS_BIRTH / 365.25)
    f["CAR_AGE_VS_EMPLOYED"] = d.OWN_CAR_AGE - f.EMPLOYED_YEARS
    f["CAR_AGE_CAPPED"] = d.OWN_CAR_AGE.isin([64, 65]).astype(int)
    f["CREDIT_INCOME"] = _safe_div(d.AMT_CREDIT, d.AMT_INCOME_TOTAL)
    f["ANNUITY_INCOME"] = _safe_div(d.AMT_ANNUITY, d.AMT_INCOME_TOTAL)
    f["CREDIT_TERM"] = _safe_div(d.AMT_CREDIT, d.AMT_ANNUITY)
    f["CREDIT_GOODS"] = _safe_div(d.AMT_CREDIT, d.AMT_GOODS_PRICE)
    f["GOODS_INCOME"] = _safe_div(d.AMT_GOODS_PRICE, d.AMT_INCOME_TOTAL)
    f["DOWN_PAYMENT"] = d.AMT_GOODS_PRICE - d.AMT_CREDIT
    f["INCOME_PER_PERSON"] = _safe_div(d.AMT_INCOME_TOTAL, d.CNT_FAM_MEMBERS)
    f["CREDIT_PER_PERSON"] = _safe_div(d.AMT_CREDIT, d.CNT_FAM_MEMBERS)
    f["ANNUITY_PER_PERSON"] = _safe_div(d.AMT_ANNUITY, d.CNT_FAM_MEMBERS)
    f["CHILD_RATIO"] = _safe_div(d.CNT_CHILDREN, d.CNT_FAM_MEMBERS)
    f["INCOME_PER_YEAR_EMPLOYED"] = _safe_div(d.AMT_INCOME_TOTAL, f.EMPLOYED_YEARS + 1)
    e = d[EXT]
    f["EXT_MEAN"] = e.mean(axis=1); f["EXT_STD"] = e.std(axis=1)
    f["EXT_MIN"] = e.min(axis=1);   f["EXT_MAX"] = e.max(axis=1)
    f["EXT_PROD"] = e.prod(axis=1, skipna=False)
    f["EXT_WEIGHTED"] = (d.EXT_SOURCE_1*2 + d.EXT_SOURCE_2*3 + d.EXT_SOURCE_3*4) / 9
    f["EXT_NA_COUNT"] = e.isnull().sum(axis=1)
    f["EXT_MEAN_X_TERM"] = f.EXT_MEAN * f.CREDIT_TERM
    f["EXT_MEAN_X_AGE"] = f.EXT_MEAN * f.AGE_YEARS
    f["EXT2_X_CREDIT_INCOME"] = d.EXT_SOURCE_2 * f.CREDIT_INCOME
    f["EXT3_X_EMPLOYED"] = d.EXT_SOURCE_3 * f.EMPLOYED_YEARS
    f["EXT_MEAN_X_ANNUITY_INC"] = f.EXT_MEAN * f.ANNUITY_INCOME
    req = [c for c in d.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    f["REQ_TOTAL"] = d[req].sum(axis=1, min_count=1)
    f["REQ_RECENT"] = d[["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_QRT"]].sum(axis=1, min_count=1)
    f["REQ_MISSING"] = d["AMT_REQ_CREDIT_BUREAU_YEAR"].isnull().astype(int)
    f["FLAG_CONTACT_SUM"] = d.FLAG_EMP_PHONE + d.FLAG_WORK_PHONE + d.FLAG_MOBIL
    return pd.concat([d, f], axis=1)

class GroupStats:
    """Unsupervised group means fitted on train features only (no target)."""
    VALS = ["AMT_INCOME_TOTAL","AMT_CREDIT","EXT_MEAN","EMPLOYED_YEARS","CREDIT_INCOME"]
    def fit(self, d):
        self.tab = {g: d.groupby(g, observed=True)[self.VALS].mean() for g in GROUPS}
        self.freq = {c: d[c].value_counts(normalize=True) for c in CATS}
        return self
    def transform(self, d):
        out = {}
        for g in GROUPS:
            m = d[[g]].join(self.tab[g], on=g)[self.VALS]
            for v in ["AMT_INCOME_TOTAL","EXT_MEAN","EMPLOYED_YEARS","CREDIT_INCOME"]:
                out[f"{g}_{v}_DIFF"] = d[v] - m[v]
            out[f"{g}_INCOME_REL"] = _safe_div(d.AMT_INCOME_TOTAL, m.AMT_INCOME_TOTAL)
        for c in CATS:
            out[f"{c}_FREQ"] = d[c].map(self.freq[c])
        return pd.concat([d, pd.DataFrame(out, index=d.index)], axis=1)

def build(train, test, groupstats=True):
    """Returns (Xtr, Xte, cat_cols). Categoricals use a shared pandas category dtype."""
    tr, te = add_features(train.drop(columns=["TARGET"], errors="ignore")), add_features(test)
    if groupstats:
        gs = GroupStats().fit(tr); tr, te = gs.transform(tr), gs.transform(te)
    drop = ["SK_ID_CURR"]
    tr, te = tr.drop(columns=drop), te.drop(columns=drop)
    for c in CATS:
        cats = sorted(tr[c].dropna().unique())
        tr[c] = pd.Categorical(tr[c], categories=cats); te[c] = pd.Categorical(te[c], categories=cats)
    return tr, te, CATS
