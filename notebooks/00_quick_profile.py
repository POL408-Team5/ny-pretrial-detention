import pandas as pd, numpy as np, sys
f=sys.argv[1]
d=pd.read_csv(f, low_memory=False, encoding="utf-8-sig")
print("rows",len(d),"cols",len(d.columns))
print("\nALL COLUMNS:"); print(list(d.columns))
nyc=["New York","Kings","Bronx","Queens","Richmond"]
d["nyc"]=d.County_Name.isin(nyc)
print("\nNYC share:", round(d.nyc.mean()*100,1),"%  n_nyc=",d.nyc.sum())
print("\nCourt_Type:\n", d.Court_Type.value_counts(dropna=False).head(8))
print("\nTop_Severity_at_Arraign:\n", d.Top_Severity_at_Arraign.value_counts(dropna=False))
print("\nRelease Decision at Arraign (all / NYC):")
print(pd.concat([d["Release Decision at Arraign"].value_counts(normalize=True,dropna=False).round(3),
                 d[d.nyc]["Release Decision at Arraign"].value_counts(normalize=True,dropna=False).round(3)],axis=1,keys=["all","nyc"]))
print("\nRelease decision by severity (NYC):")
print(pd.crosstab(d[d.nyc].Top_Severity_at_Arraign, d[d.nyc]["Release Decision at Arraign"], normalize="index").round(3))
for c in ["Race","Ethnicity","Gender","Arrest_Type","Representation_Type"]:
    print(f"\n{c}:\n", d[c].value_counts(dropna=False,normalize=True).round(3).head(8))
pri=[c for c in d.columns if c.startswith("prior_") or c.startswith("pend_") or c=="supervision"]
print("\nprior/pending cols:",pri)
for c in pri: print(c, d[c].value_counts(dropna=False).head(6).to_dict())
rc=[c for c in d.columns if "rearrest" in c.lower() or "warrant" in c.lower() or "Days" in c]
print("\nrearrest/warrant cols:",rc)
for c in rc[:6]: print(c, d[c].value_counts(dropna=False).head(6).to_dict())
rel=d[d["Release Decision at Arraign"].isin(["ROR","Nonmonetary release","Bail-set"])&d.nyc]
print("\nNYC released-ish n=",len(rel))
if "Bench_Warrant_Issued" in d.columns: print("Bench warrant rate (NYC released):", rel.Bench_Warrant_Issued.value_counts(normalize=True,dropna=False).round(3).to_dict())
bail=d[d["Release Decision at Arraign"]=="Bail-set"]
print("\nBail-set n=",len(bail)," posted:",bail.Bail_Set_and_Posted_at_Arraign.value_counts(normalize=True,dropna=False).round(3).to_dict())
print("First_Bail_Set_Cash quantiles:", bail.First_Bail_Set_Cash.describe(percentiles=[.25,.5,.75,.9]).round(0).to_dict())
print("\nJudge_Name nunique:", d.Judge_Name.nunique(), " Court_Name nunique:", d.Court_Name.nunique(), " County:", d.County_Name.nunique())
print("\nmissing share top:", d.isna().mean().sort_values(ascending=False).head(12).round(3).to_dict())
