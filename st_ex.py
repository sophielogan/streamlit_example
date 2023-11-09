
from configparser import LegacyInterpolation
from tarfile import data_filter
import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

st.title('Maverick Turbine Info')
st.header('Nov 2022-23')
DATE_COLUMN = 'time'


def load_data(nrows):
    data = pd.read_excel(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


uploaded_file = st.file_uploader("Upload your file here...", type=['xlsx', 'csv'], accept_multiple_files=True)

all_dfs = []
if uploaded_file is not None:
    for file in uploaded_file:
        with st.spinner(text='In progress'):
         time.sleep(3)
         st.success('Done')   
        st.balloons()   
         
        if file.name=='MaverickWTG001Data.csv':
            dataframe = pd.read_csv(file)
            dataframe[DATE_COLUMN]= pd.to_datetime(dataframe[DATE_COLUMN])
            st.write("filename:", file.name)
            st.write(dataframe)
            all_dfs.append(dataframe)
        else:
            dataframe_pow = pd.read_csv(file)
            all_dfs.append(dataframe_pow)

df = all_dfs[0].merge(all_dfs[1], how='left', left_on='Power', right_on='value')
st.write('Merged data')
st.write(df)

with st.spinner(text='In progress'):
  time.sleep(3)
  st.success('Done')   
st.snow()   

dataframe = all_dfs[1]
dataframe_pow = all_dfs[0]

st.subheader('Maverick Windspeed in 2023')
hist_values = np.histogram(dataframe['value'].dropna())
st.bar_chart(hist_values)

st.subheader('Maverick Variables Distribution')
var_of_interest =st.selectbox("Pick one", ["Maverick.Turbines.WTG-001.V_WIN", "Maverick.Turbines.WTG-001.P_ACT", "Maverick.Turbines.WTG-001.N_ROT_PLC"], placeholder="Wind Speed")
filtered_df = dataframe[dataframe['tag']==var_of_interest]
st.bar_chart(filtered_df['value'].dropna())

st.subheader('Wind Speed Distribution')
st.scatter_chart(df, y='Wind Speed', x='time', color='tag')


trainmodel= st.sidebar.checkbox("Train model", False)
dokfold= st.sidebar.checkbox("DO KFold", False)
distView=st.sidebar.checkbox("Dist View", False)
_3dplot=st.sidebar.checkbox("3D plots", False)
linechart=st.sidebar.checkbox("Linechart",False)

if trainmodel:
	st.header("Modeling")    
	y=dataframe_pow["Wind Speed"]
	X=dataframe_pow[["Power"]].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	lrgr = LinearRegression()
	lrgr.fit(X_train,y_train)
	pred = lrgr.predict(X_test)

	mse = mean_squared_error(y_test,pred)
	rmse = sqrt(mse)

	st.markdown(f"""
	Linear Regression model trained :
		- MSE:{mse}
		- RMSE:{rmse}
	""")
	st.success('Model trained successfully')
if trainmodel:
    st.scatter_chart(dataframe_pow, y='Power', x='Wind Speed') 
