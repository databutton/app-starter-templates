import pandas as pd
import databutton as db
import streamlit as st


@db.streamlit('/label')
def dash2():
    df = pd.read_csv('./while_waiting_for_a_store/data.csv')
    random_post = df[df['relevance'] == -1].sample(n=1)
    random_record = random_post.to_records()[0]
    st.title('Rinder')
    st.markdown(
        f"### { random_record['title'] }\n\n{ random_record['selftext'] }\n\n{ random_record['score'] } 👍\n\n{ random_record['num_comments'] } 🗣")
    if st.button('Bad', help='No thanks!'):
        random_post['relevance'] = 0
        df.loc[random_post.index] = random_post
        df.to_csv('./while_waiting_for_a_store/data.csv', index=0)

    if st.button('Good', help='I want more of this'):
        random_post['relevance'] = 1
        df.loc[random_post.index] = random_post
        df.to_csv('./while_waiting_for_a_store/data.csv', index=0)