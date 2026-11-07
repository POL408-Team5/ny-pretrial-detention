import pandas as pd, numpy as np
d=pd.read_csv("NYS for Web 2022.csv", low_memory=False, encoding="utf-8-sig")
nyc=["New York","Kings","Bronx","Queens","Richmond"]
d["nyc"]=d.County_Name.isin(nyc); R="Release Decision at Arraign"
d=d[d[R]!="Disposed at arraign"]          # decision only meaningful for continuing cases
d["detained"]=d[R].isin(["Remanded"]) | (d[R].eq("Bail-set") & d.Bail_Set_and_Posted_at_Arraign.eq("N"))
d["bail_or_remand"]=d[R].isin(["Remanded","Bail-set"])
d["released"]=d[R].isin(["ROR","Nonmonetary release"]) | (d[R].eq("Bail-set") & d.Bail_Set_and_Posted_at_Arraign.eq("Y"))
d["warrant"]=d.Warrant_Ordered_btw_Arraign_and_Dispo.eq("Y")
d["rearr_any"]=d.rearrest.notna() & ~d.rearrest.astype(str).str.contains("No Arrest",case=False)
print("rearrest values:", d.rearrest.value_counts(dropna=False).head(6).to_dict())
print("rearrest_180:", d.rearrest_180.value_counts(dropna=False).head(6).to_dict())
print("Bail_Made_Indicator (bail-set):", d[d[R]=="Bail-set"].Bail_Made_Indicator.value_counts(dropna=False,normalize=True).round(3).to_dict())
print("\n== Q1 detention at arraignment (remand or bail not posted), continuing cases")
for lab,g in [("all",d),("NYC",d[d.nyc]),("rest",d[~d.nyc])]:
    print(f"{lab:5s} n={len(g):7d} bail/remand={g.bail_or_remand.mean():.3f} detained={g.detained.mean():.3f}")
print("\nNYC felony: bail/remand rate by race:")
f=d[d.nyc & d.Top_Severity_at_Arraign.eq("Felony")]
print(f.groupby("Race").bail_or_remand.agg(['mean','size']).round(3))
print("\nNYC felony: bail/remand by violent-felony ind x prior_vfo:")
print(f.pivot_table(values="bail_or_remand",index="Top_Charge_at_Arraign_Violent_Felony_Ind",columns="prior_vfo_cnt",aggfunc="mean").round(3))
print("\nbail/remand rate by county (felony, top 12 by n):")
c=d[d.Top_Severity_at_Arraign.eq("Felony")].groupby("County_Name").bail_or_remand.agg(['mean','size']).sort_values('size',ascending=False).head(12).round(3); print(c)
print("\nNYC felony: judge-level bail/remand rate spread (judges with >=200 felony cases):")
j=f.groupby("Judge_Name").bail_or_remand.agg(['mean','size']); j=j[j['size']>=200]
print("n judges",len(j), "p10/p50/p90:", j['mean'].quantile([.1,.5,.9]).round(3).to_dict())
print("\n== Q2 bail set but never posted (share of bail-set cases):", round((d[d[R]=='Bail-set'].Bail_Made_Indicator.astype(str).str.upper().ne('Y')).mean(),3))
print("median cash bail NYC felony:", f[f[R]=="Bail-set"].First_Bail_Set_Cash.median())
print("\n== Q3 warrant among released (NYC):")
rel=d[d.released & d.nyc]
print("n released", len(rel), "warrant rate", round(rel.warrant.mean(),3), " rearrest_any", round(rel.rearr_any.mean(),3))
print("warrant by severity:", rel.groupby("Top_Severity_at_Arraign").warrant.mean().round(3).to_dict())
print("warrant by prior_misd (0,1,2,3+):", rel.groupby(rel.prior_misd_cnt.clip(upper=3)).warrant.mean().round(3).to_dict())
print("warrant by pend_misd:", rel.groupby("pend_misd").warrant.mean().round(3).to_dict())
print("warrant by supervision:", rel.groupby("supervision").warrant.mean().round(3).to_dict())
print("warrant by race:", rel.groupby("Race").warrant.mean().round(3).to_dict())
print("warrant by arrest type:", rel.groupby("Arrest_Type").warrant.mean().round(3).to_dict())
print("\nquick AUC check: logistic on released NYC, features = severity, weight, VFO, priors, pending, supervision, age, arrest type")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
X=pd.get_dummies(rel[["Top_Severity_at_Arraign","Top_Charge_Weight_at_Arraign","Top_Charge_at_Arraign_Violent_Felony_Ind","Arrest_Type","Arraign Charge Category"]].astype(str))
for c in ["prior_vfo_cnt","prior_nonvfo_cnt","prior_misd_cnt","pend_vfo","pend_nonvfo","pend_misd","supervision","Age_at_Arrest"]:
    X[c]=pd.to_numeric(rel[c],errors="coerce").fillna(-1)
y=rel.warrant.astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=408,stratify=y)
m=LogisticRegression(max_iter=2000).fit(Xtr,ytr); print("AUC warrant:",round(roc_auc_score(yte,m.predict_proba(Xte)[:,1]),3))
y2=rel.rearr_any.astype(int); Xtr,Xte,ytr,yte=train_test_split(X,y2,test_size=0.3,random_state=408,stratify=y2)
m=LogisticRegression(max_iter=2000).fit(Xtr,ytr); print("AUC rearrest:",round(roc_auc_score(yte,m.predict_proba(Xte)[:,1]),3))
# Q1 AUC: predict bail/remand among continuing NYC cases from case info only
g=d[d.nyc]; X=pd.get_dummies(g[["Top_Severity_at_Arraign","Top_Charge_Weight_at_Arraign","Top_Charge_at_Arraign_Violent_Felony_Ind","Arrest_Type","Arraign Charge Category"]].astype(str))
for c in ["prior_vfo_cnt","prior_nonvfo_cnt","prior_misd_cnt","pend_vfo","pend_nonvfo","pend_misd","supervision","Age_at_Arrest"]:
    X[c]=pd.to_numeric(g[c],errors="coerce").fillna(-1)
y=g.bail_or_remand.astype(int); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=408,stratify=y)
m=LogisticRegression(max_iter=2000).fit(Xtr,ytr); print("AUC bail/remand decision:",round(roc_auc_score(yte,m.predict_proba(Xte)[:,1]),3))
