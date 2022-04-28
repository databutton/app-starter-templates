import pandas as pd
from lib.config import DATA_KEY
import databutton as db
import streamlit as st


@db.streamlit('/label')
def dash2():
    df = pd.read_csv(DATA_KEY)
    random_post = df[df['relevance'] == -1].sample(n=1)
    random_record = random_post.to_records()[0]
    st.title('Rinder')
    st.markdown(
        f"### { random_record['title'] }\n\n{ random_record['selftext'] }\n\n{ random_record['score'] } 👍\n\n{ random_record['num_comments'] } 🗣")

    def label_data(index, relevance):
        df.loc[index, 'relevance'] = relevance
        df.to_csv(DATA_KEY, index=0)

    st.button('Bad', on_click=label_data, args=(random_post.index, 0))
    st.button('Good', on_click=label_data, args=(random_post.index, 1))


@db.streamlit('/see-labels')
def see_labels():
    df = pd.read_csv(DATA_KEY)
    labeled = df[df['relevance'] != -1]
    st.dataframe(labeled)
