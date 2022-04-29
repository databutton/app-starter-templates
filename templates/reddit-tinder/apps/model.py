import pandas as pd
from lib.config import DATA_KEY
import databutton as db
import streamlit as st
import torch
import numpy as np

from lib.reddit_data import get_data
from lib.rmodel import Dataset, BertClassifier, train, predict, evaluate
from lib.slack import post_message_to_slack




@db.streamlit('/app/pipeline')
def reddit_pipeline():
    st.title('Ridder Pipeline Test')


    col1, col2 = st.columns(2)
    with col1:
        numh = st.number_input(label='How many hours to get', value=3)
    with col2:
        get = st.button('GO!')

    if(get):
        st.markdown('### Step 1 - Pull data from Reddit')
        df = get_data(numh)
        st.write(df[['time', 'title', 'selftext', 'full_link']])

        st.markdown('### Step 2 - Filter relevant posts')
        model = torch.load(open('model/bert.torch','rb'))
        df['text'] = df['title'] + ' '+ df['selftext']
        df_pred = df['text'].copy()
        pred = predict(model, df_pred)
        st.write(pred)


        st.markdown('### Step 3 - Write relevants to Slack')
        post_message_to_slack("hmmm - No interesting Reddit posts it seems? Someone keen on labeling a bit more?")


@db.streamlit('/app/train')
def reddit_pipeline():
    st.title('Train new model')
    if st.button('TRAIN NEW MODEL'):
        datapath = f'while_waiting_for_store.csv'
        df = pd.read_csv(datapath)
        df=df[df.columns[2:]]
        df = df.fillna(value=str(''))
        df['text'] = df['title'] + '  ' + df['selftext']
        df['text'] = df['text'].fillna(value='').astype(str)
        df_clean = df[df.relevance!=-1]
        df_good = df_clean[df_clean['relevance']==1]
        df_bad  = df_clean[df_clean['relevance']==0]
        n = min(len(df_good), len(df_bad))
        df_clean = pd.concat([df_good.sample(n), df_bad.sample(n)])
        df_clean.describe()

        np.random.seed(112)
        df_train, df_val, df_test = np.split(df_clean.sample(frac=1, random_state=42), 
                                     [int(.8*len(df_clean)), int(.9*len(df_clean))])

        print(len(df_train),len(df_val), len(df_test))
        EPOCHS = 2
        model = BertClassifier()
        LR = 1e-6
              
        train(model, df_train, df_val, LR, EPOCHS)

        evaluate(model, df_test)
        torch.save(model, 'model/bert.torch')

