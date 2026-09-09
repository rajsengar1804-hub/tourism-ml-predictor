from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib

BASE = Path(__file__).resolve().parent

DATA = BASE / "cleaned_tourism_data(1).csv"
REG = BASE / "rating_regression_model.joblib"
CLF = BASE / "visitmode_classification_model.joblib"

st.set_page_config(page_title='Tourism ML Predictor',page_icon='🌍',layout='wide')
@st.cache_resource
def load_models():
    reg_bundle = joblib.load(REG)
    clf_bundle = joblib.load(CLF)
    return reg_bundle, clf_bundle


@st.cache_data
def load_data():
    return pd.read_csv(DATA)
reg_bundle,clf_bundle=load_models(); df=load_data()
reg_fe,reg_model=reg_bundle['feature_engineer'],reg_bundle['model']
clf_fe,clf_le,clf_model=clf_bundle['feature_engineer'],clf_bundle['label_encoder'],clf_bundle['model']

st.title('🌍 Tourism Rating & Visit Mode Prediction')
st.caption('Two independent tree-based ML pipelines: Rating regression and VisitMode classification.')
mode=st.sidebar.radio('Select prediction',['Rating — Regression','VisitMode — Classification'])

# Raw fields used by the trained feature engineers. IDs are numeric; categorical fields are user-selectable.
num_cols=['VisitYear','VisitMonth','AttractionId','ContinentId','RegionId','CountryId','CityId','AttractionCityId','AttractionTypeId']
cat_cols=['CityName','Country','Region','Continent','Attraction','AttractionAddress','AttractionType']

def make_input(extra_target):
    row={}
    c1,c2=st.columns(2)
    with c1:
        row['VisitYear']=st.number_input('Visit Year',min_value=2000,max_value=2100,value=int(df.VisitYear.median()))
        row['VisitMonth']=st.number_input('Visit Month',min_value=1,max_value=12,value=int(df.VisitMonth.median()))
        row['AttractionId']=st.number_input('Attraction ID',min_value=0,value=int(df.AttractionId.median()),step=1)
        row['ContinentId']=st.number_input('Continent ID',min_value=0.0,value=float(df.ContinentId.median()))
        row['RegionId']=st.number_input('Region ID',min_value=0.0,value=float(df.RegionId.median()))
        row['CountryId']=st.number_input('Country ID',min_value=0.0,value=float(df.CountryId.median()))
        row['CityId']=st.number_input('City ID',min_value=0.0,value=float(df.CityId.median()))
        row['AttractionCityId']=st.number_input('Attraction City ID',min_value=0,value=int(df.AttractionCityId.median()),step=1)
        row['AttractionTypeId']=st.number_input('Attraction Type ID',min_value=0,value=int(df.AttractionTypeId.median()),step=1)
    with c2:
        for c in cat_cols:
            vals=sorted(df[c].dropna().astype(str).unique().tolist())
            default=vals[0] if vals else 'Unknown'
            row[c]=st.selectbox(c,vals,index=vals.index(default) if default in vals else 0)
    if extra_target=='reg':
        vals=sorted(df['VisitMode'].dropna().astype(str).unique().tolist())
        row['VisitMode']=st.selectbox('Visit Mode (input feature)',vals)
    else:
        row['Rating']=st.slider('Rating (input feature)',1,5,4)
    return pd.DataFrame([row])

if mode.startswith('Rating'):
    st.subheader('Rating Regression')
    st.write('Predict the expected attraction rating from tourism/visit attributes.')
    x=make_input('reg')
    if st.button('Predict Rating',type='primary'):
        X=reg_fe.transform(x)
        pred=float(reg_model.predict(X)[0]); pred=float(np.clip(pred,1,5))
        st.metric('Predicted Rating',f'{pred:.2f} / 5')
        st.caption(f'Features used after selection: {len(reg_fe.selected_features)}')
else:
    st.subheader('VisitMode Classification')
    st.write('Predict whether the visit is Business, Couples, Family, Friends, or Solo.')
    x=make_input('clf')
    if st.button('Predict Visit Mode',type='primary'):
        X=clf_fe.transform(x)
        pred_id=int(np.asarray(clf_model.predict(X)).ravel()[0]); pred=clf_le.inverse_transform([pred_id])[0]
        st.metric('Predicted Visit Mode',pred)
        if hasattr(clf_model,'predict_proba'):
            probs=clf_model.predict_proba(X)[0]
            proba=pd.DataFrame({'VisitMode':clf_le.classes_,'Probability':probs}).sort_values('Probability',ascending=False)
            st.dataframe(proba.style.format({'Probability':'{:.2%}'}),hide_index=True,use_container_width=True)
        st.caption(f'Features used after selection: {len(clf_fe.selected_features)} | Balancing: class_weight=balanced')

with st.expander('Model pipeline'):
    st.markdown('**Regression:** feature engineering → variance threshold → multicollinearity removal → Random Forest model-based importance → Random Forest/XGBoost/CatBoost/LightGBM baseline → best model hyperparameter tuning → saved model.')
    st.markdown('**Classification:** feature engineering → variance threshold → multicollinearity removal → Random Forest model-based importance → four tree-model baselines → class_weight=balanced vs SMOTE → selected class-weight approach → hyperparameter tuning → saved model.')
