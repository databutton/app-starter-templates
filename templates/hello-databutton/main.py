import databutton as db
import streamlit as st

@db.streamlit(route='/hello', name="Hello Databutton", cpu=1, memory='2056Gi')
def hello():
    st.title('Hello, Databutton')
