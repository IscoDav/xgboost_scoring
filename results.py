import streamlit as st
from xgb_limit import *
from lgb_limit import *
xgb = limitXGB('XGBoost_V_02')
lgb = limitLGB('LGBM_V_02')



chart_xgb = xgb.cleaning_and_prediction()
chart_lgb = lgb.cleaning_and_prediction()

st.dataframe(chart_lgb.style.format(precision=0))


st.write(f'## Shape of filtered dataset is {chart_lgb.shape}')

metrics_xgb = xgb.metrics()
metrics_lgb = lgb.metrics()

metrics_xgb['date'] = pd.to_datetime('today').date()
metrics_lgb['date'] = pd.to_datetime('today').date()

st.write('Prediction metrics of XGBoost')
st.dataframe(metrics_xgb.style.format(precision=2))
st.write('Prediction metrics of LGBM')
st.dataframe(metrics_lgb.style.format(precision=2))

lines_xgb = xgb.compare()
lines_lgb = lgb.compare()

st.title('XGBoost metrics Vs Wings')
st.line_chart(lines_xgb.loc[:,['actual', 'predicted']], width=1, height=0)

st.title('XGBoost metrics Vs Wings')
st.line_chart(lines_lgb.loc[:,['actual', 'predicted']], width=1, height=0)

compare_xgb = xgb.compare()
st.dataframe(compare_xgb.style.format(precision=0))

compare_lgb = lgb.compare()
st.dataframe(compare_lgb.style.format(precision=0))

#%%
