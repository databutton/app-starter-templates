import streamlit as st
import pandas as pd
import numpy as np
import databutton as db

st.title('Web-store sales')
stats = ['Quantity', 'Sales', 'Profit', 'Discount']
frames = {}
for stat in stats:
    frames[stat] = pd.read_csv('./data/generated/'+stat+'.csv')
SubCategories = frames['Sales']['Sub-Category'].unique()
Cities = frames['Sales']['City'].unique()
option = st.selectbox('Quantity', options=stats)
disp = frames[option].groupby('Sub-Category').sum()
# if(st.button(label='All')):
#     disp = frames['Quantity']
# if(st.button(label='Category')):
#     disp = frames['Quantity'].groupby('Category').sum()
# if(st.button(label='Sub-Category')):
#     disp = frames['Quantity'].groupby('Sub-Category').sum()
# if(st.button(label='City')):
#     disp = frames['Quantity'].groupby('City').sum()
st.dataframe(disp)
