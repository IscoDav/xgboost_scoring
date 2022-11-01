import streamlit as st
from limit_module import *
model = limit_model('XGBoost_V_02')



chart_data = model.cleaning_and_prediction()

st.dataframe(chart_data.style.format(precision=0))


st.write(f'## Shape of filtered dataset is {chart_data.shape}')

metrics = model.metrics()
metrics['date'] = pd.to_datetime('today').date()
st.write('## Prediction metrics')

st.dataframe(metrics.style.format(precision=2))


lines = model.compare()

st.title('XGBoost metrics Vs Wings')
st.line_chart(lines.loc[:,['actual', 'predicted']], width=1, height=0)

compare = model.compare()
st.dataframe(compare.style.format(precision=0))

#%%
