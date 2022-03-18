import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import databutton as db


@db.streamlit('/app/salesApp')
def app():
    stats = ['Quantity', 'Sales', 'Profit', 'Discount']
    views = ['Category', 'City', 'Product']
    frames = {}
    for stat in stats:
        frames[stat] = pd.read_csv('./data/predicted/'+stat+'-estimated.csv')

    st.sidebar.title('Sales projection app')
    option = st.sidebar.selectbox(
        label='Which statistic do you want to examine?', options=stats)
    df = frames[option]

    view = st.sidebar.selectbox(label='View statistics by', options=views)
    stlabel = ''
    if(view == 'Category'):
        stlabel = 'Estimated sales by Category'
        dG = df.groupby('Sub-Category').sum()
    elif(view == 'City'):
        stlabel = 'Estimated sales by City'
        dG = df.groupby('City').sum()
    elif(view == 'Product'):
        stlabel = 'Estimated sales by Product'
        dG = df.groupby('Product Name').sum()

    plot_what = st.selectbox(label='Plot trend for', options=dG.index)
    g = dG.loc[plot_what][:10]
    g = g[::-1]
    fig = px.line(x=g.index, y=g, title=plot_what)
    st.plotly_chart(fig)

    st.write(stlabel)
    slice_ = ['January 2018 Estimated']
    s = dG.style.set_properties(
        **{'background-color': '#b987ffff'}, subset=slice_)
    st.dataframe(s, height=8000, width=1000)

    #builder = GridOptionsBuilder.from_dataframe(df)
    #builder.configure_side_bar(filters_panel=True, columns_panel=True, defaultToolPanel="columns")
    #builder.configure_default_column(groupable=True, filterable=True, sorteable=True, resizable=True, aggFunc='sum')
    #go = builder.build()
    #AgGrid(df, gridOptions=go, enable_enterprise_modules=True, height=800, width=1000)
