import streamlit as st
import pandas as pd
st.title('🎈 Machine Learning App')

st.info('This app builds a machine learning model')
with st.expander('Data'):
  st.write('**Raw Data**')
  df=pd.read_csv("https://raw.githubusercontent.com/psykick-21/penguin-species-prediction-streamlit/ea896a194e70140227e638724c7af21349bedf99/datasets/penguins_cleaned.csv")
  df

  st.write('**X**')
  X=df.iloc[:,1:]
  X

  st.write('**y**')
  y=df.iloc[:,0]
  y
with st.expander('Data Visulization'):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

# Data Preparation
with st.sidebar:
  st.header('Input Features')
  island=st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  gender=st.selectbox('Gender','('male','female'))
  bill_length_mm=st.slider('Bill length (mm)',32.1,59.6,43.9) #(min,max,avg)
  
  
