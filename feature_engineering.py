import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class TourismFeatureEngineer:
    def __init__(self,target,corr_threshold=0.95,variance_threshold=0.0,importance_quantile=0.50):
        self.target=target; self.corr_threshold=corr_threshold; self.variance_threshold=variance_threshold; self.importance_quantile=importance_quantile
    def _base(self,X):
        out=pd.DataFrame(index=X.index)
        for c in self.num_cols: out[c]=pd.to_numeric(X[c],errors='coerce')
        for c in self.cat_cols:
            s=X[c].fillna('Unknown').astype(str).str.strip(); maps=getattr(self,'freq_maps',getattr(self,'maps',{})); out[c+'_freq']=s.map(maps[c]).fillna(0.0)
        if 'VisitMonth' in out.columns:
            m=out['VisitMonth'].fillna(0); out['VisitMonth_sin']=np.sin(2*np.pi*m/12); out['VisitMonth_cos']=np.cos(2*np.pi*m/12)
        return out
    def fit(self,X,y):
        drop=['TransactionId','UserId',self.target,'VisitModeId']
        if self.target=='VisitMode': drop.append('VisitMode')
        self.drop_cols=drop; self.cat_cols=[c for c in X.select_dtypes(include='object').columns if c not in drop]; self.num_cols=[c for c in X.columns if c not in drop and c not in self.cat_cols]
        self.freq_maps={}
        for c in self.cat_cols:
            s=X[c].fillna('Unknown').astype(str).str.strip(); self.freq_maps[c]=s.value_counts(normalize=True).to_dict()
        Z=self._base(X).replace([np.inf,-np.inf],np.nan); self.base_features=Z.columns.tolist(); self.medians_=Z.median().to_dict(); Z=Z.fillna(self.medians_)
        vt=VarianceThreshold(self.variance_threshold); vt.fit(Z); self.variance_features=[c for c,k in zip(Z.columns,vt.get_support()) if k]; Z=Z[self.variance_features]
        corr=Z.corr().abs(); upper=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)); self.corr_dropped=[c for c in upper.columns if any(upper[c]>self.corr_threshold)]; self.corr_features=[c for c in Z.columns if c not in self.corr_dropped]; Z=Z[self.corr_features]
        sel=RandomForestRegressor(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42) if self.target=='Rating' else RandomForestClassifier(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42,class_weight='balanced_subsample')
        sel.fit(Z,y); imp=pd.Series(sel.feature_importances_,index=Z.columns).sort_values(ascending=False); self.importances=imp.to_dict(); self.importance_threshold=float(imp.quantile(self.importance_quantile)); self.selected_features=imp[imp>=self.importance_threshold].index.tolist();
        if len(self.selected_features)<8:self.selected_features=imp.head(min(8,len(imp))).index.tolist()
        self.selector_model=sel; return self
    def transform(self,X):
        base=getattr(self,'base_features',getattr(self,'base',None)); med=getattr(self,'medians_',getattr(self,'med',{})); Z=self._base(X).replace([np.inf,-np.inf],np.nan).reindex(columns=base); Z=Z[self.variance_features][self.corr_features][self.selected_features]
        for c in Z.columns: Z[c]=Z[c].fillna(med.get(c,0))
        return Z.astype(float)

class FE(TourismFeatureEngineer):
    pass
