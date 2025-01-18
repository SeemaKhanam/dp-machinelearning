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
  gender=st.selectbox('Gender',('male','female'))
  bill_length_mm=st.slider('Bill length (mm)',32.1,59.6,43.9) #(min,max,avg)
  bill_depth_mm=st.slider('Bill Depth (mm)',13.1,21.5,17.2)
  flipper_length_mm=st.slider('Flipper length (mm)',172.0,231.0,201.0)
  body_mass_g=st.slider('Body mass (g)',2700.0,6300.0,4207.0)
  
  # create a dataframe for the input features
  data={'island':island,
        'bill_length_mm':bill_length_mm,
        'bill_depth_mm':bill_depth_mm,
        'flipper_length_mm':flipper_length_mm,
        'body_mass_g':body_mass_g,
        'gender':gender}
  input_df=pd.DataFrame(data,index[0])
  input_df

        
        
