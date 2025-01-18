import streamlit as st
import pandas as pd
st.title('🎈 Machine Learning App')

st.info('This app builds a machine learning model')
df=pd.read_csv("https://raw.githubusercontent.com/psykick-21/penguin-species-prediction-streamlit/ea896a194e70140227e638724c7af21349bedf99/datasets/penguins_cleaned.csv")
df
