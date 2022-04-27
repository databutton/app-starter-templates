import pandas as pd
import databutton as db
import streamlit as st
import ray


# Store
# { index: revelance }

# db.put(f"relevance-{index}", relevance)
@db.streamlit('/rinder')
def dash():
    ray.init()
    df = pd.read_csv('./notebooks/full_dataset.csv')
    random_post = df[df['relevance'] == -1].sample(n=1).to_records()[0]
    st.title('Rinder')
    st.markdown(
        f"### { random_post['title'] }\n\n{ random_post['selftext'] }\n\n{ random_post['score'] } 👍\n\n{ random_post['num_comments'] } 🗣")
    st.button('Bad', help='No thanks!')
    if st.button('Good', help='I want more of this'):
        pass
