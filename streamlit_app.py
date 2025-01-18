import streamlit as st
import pandas as pd
import numpy as pd
from sklearn.ensemble import RandomForestClassifier
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
  input_df=pd.DataFrame(data,index=[0])
  input_penguins=pd.concat([input_df,X],axis=0)
# Preprocessing 
#encode x
encode=['island','sex']
df_penguine=pd.get_dummies(input_penguins,prefix=encode)
x=df_penguins[1:]
input_row=df_penguins[:1]

#Encode y
target_mapper={'Adelie':0,
               'Chinstrap':1,
               'Gentoo':2}
def target_encode(val):
  return target_mapper[val]

y=y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y

# Model training and inference 
## train the ML model
clf=RandomForestClassifier()
clf.fit(X,y)

#apply model to make prediction 
prediction=clf.predict(input_row)
prediction_proba=clf.predict_proba(input_row)
df_predict_proba=pd.DataFrame(prediction_proba)
df_prediction_proba.columns=['Adelie','chinstrap','Gentoo']
df_prediction_proba.rename(columns={0:'Adelie',
                                    1:'Chinstrap',
                                    2:'Ghentoo'})

#Display predicted species
# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))

                                    
