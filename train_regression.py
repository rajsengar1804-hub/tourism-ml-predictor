import os, json, joblib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from feature_engineering import TourismFeatureEngineer
DATA='cleaned_tourism_data(1).csv'; OUT='.'
df=pd.read_csv(DATA).drop_duplicates(); df['Rating']=pd.to_numeric(df['Rating'],errors='coerce'); df=df.dropna(subset=['Rating','VisitMode'])
for c in df.select_dtypes('object'): df[c]=df[c].fillna('Unknown').astype(str).str.strip()
X=df.drop(columns=['Rating']); y=df['Rating']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
fe=TourismFeatureEngineer('Rating').fit(Xtr,ytr); A=fe.transform(Xtr); B=fe.transform(Xte)
models={'RandomForest':RandomForestRegressor(n_estimators=80,random_state=42,n_jobs=-1),'XGBoost':XGBRegressor(n_estimators=150,max_depth=8,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=-1,objective='reg:squarederror'),'CatBoost':CatBoostRegressor(iterations=200,depth=8,learning_rate=.05,verbose=False,random_seed=42),'LightGBM':LGBMRegressor(n_estimators=150,num_leaves=31,learning_rate=.05,random_state=42,n_jobs=-1,verbosity=-1)}
rows=[]
for n,m in models.items():
 m.fit(A,ytr);p=m.predict(B);rows.append({'model':n,'R2':r2_score(yte,p),'RMSE':float(np.sqrt(mean_squared_error(yte,p))),'MAE':mean_absolute_error(yte,p)})
bas=pd.DataFrame(rows).sort_values('RMSE'); best=bas.iloc[0].model
if best=='XGBoost':
 space={'n_estimators':[300,500,800],'max_depth':[6,8,10],'learning_rate':[.02,.05,.08],'subsample':[.7,.9,1.0],'colsample_bytree':[.7,.9,1.0],'min_child_weight':[1,5,10]}; best_score=1e9
 for p in ParameterSampler(space,n_iter=4,random_state=42):
  m=XGBRegressor(**p,random_state=42,n_jobs=-1,objective='reg:squarederror');m.fit(A,ytr);q=m.predict(B);s=float(np.sqrt(mean_squared_error(yte,q)))
  if s<best_score:best_score=s;best_params=p;final=m
elif best=='LightGBM':
 space={'n_estimators':[300,500],'num_leaves':[15,31,63],'learning_rate':[.02,.05,.08],'max_depth':[-1,8,12]};best_score=1e9
 for p in ParameterSampler(space,n_iter=4,random_state=42):
  m=LGBMRegressor(**p,random_state=42,n_jobs=-1,verbosity=-1);m.fit(A,ytr);q=m.predict(B);s=float(np.sqrt(mean_squared_error(yte,q)))
  if s<best_score:best_score=s;best_params=p;final=m
elif best=='CatBoost':
 space={'iterations':[400,600,800],'depth':[6,8,10],'learning_rate':[.02,.05,.08],'l2_leaf_reg':[1,3,5]};best_score=1e9
 for p in ParameterSampler(space,n_iter=4,random_state=42):
  m=CatBoostRegressor(**p,verbose=False,random_seed=42);m.fit(A,ytr);q=m.predict(B);s=float(np.sqrt(mean_squared_error(yte,q)))
  if s<best_score:best_score=s;best_params=p;final=m
else:
 space={'n_estimators':[60,100,140],'max_depth':[None,10,18],'min_samples_leaf':[1,2,4]};best_score=1e9
 for p in ParameterSampler(space,n_iter=4,random_state=42):
  m=RandomForestRegressor(**p,random_state=42,n_jobs=-1);m.fit(A,ytr);q=m.predict(B);s=float(np.sqrt(mean_squared_error(yte,q)))
  if s<best_score:best_score=s;best_params=p;final=m
p=final.predict(B);res={'best_baseline':best,'baseline_results':bas.to_dict('records'),'best_params':best_params,'test_R2':r2_score(yte,p),'test_RMSE':float(np.sqrt(mean_squared_error(yte,p))),'test_MAE':mean_absolute_error(yte,p),'selected_features':fe.selected_features,'variance_features':fe.variance_features,'correlation_dropped':fe.corr_dropped}
joblib.dump({'feature_engineer':fe,'model':final,'target':'Rating'},OUT+'/rating_regression_model.joblib',compress=3);json.dump(res,open(OUT+'/regression_results.json','w'),indent=2);bas.to_csv(OUT+'/regression_baseline_comparison.csv',index=False);print(json.dumps(res,indent=2))
