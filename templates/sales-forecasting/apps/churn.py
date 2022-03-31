import databutton as db
import pandas as pd
import streamlit as st


@db.streamlit('/apps/churn')
def app():
    df = pd.read_csv('./data/raw/churn.csv')
    st.text('Welcome to my site, people!!')
    st.dataframe(df)
