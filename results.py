import streamlit as st
from xgb_limit import *

xgb = limit_model('XGBoost_V_03')

chart_xgb = xgb.cleaning_and_prediction()

metrics_xgb = xgb.metrics()

metrics_xgb['date'] = pd.to_datetime('today').date()

st.write('Prediction metrics of XGBoost')
st.dataframe(metrics_xgb.style.format(precision=2))
st.write('Prediction metrics of LGBM')

lines_xgb = xgb.compare()

st.title('XGBoost metrics Vs Wings')
st.line_chart(lines_xgb.loc[:,['actual', 'predicted']], width=1, height=0)

compare_xgb = xgb.compare()
st.dataframe(compare_xgb.style.format(precision=0))


#%%
