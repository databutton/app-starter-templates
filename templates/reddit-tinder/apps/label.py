from lib.config import DATA_KEY
import databutton as db
import streamlit as st


@db.streamlit('/label', cpu='2', memory='4Gi')
def dash2():
    df = db.dataframes.get(DATA_KEY)
    random_post = df[df['relevance'] == -1].sample(n=1)
    random_record = random_post.to_records()[0]
    st.title('Rinder')
    st.markdown(
        f"### { random_record['title'] }\n\n{ random_record['selftext'] }\n\n{ random_record['score'] } 👍\n\n{ random_record['num_comments'] } 🗣")

    def label_data(index, relevance):
        df.loc[index, 'relevance'] = relevance
        db.dataframes.put(df, DATA_KEY)

    st.button('Bad', on_click=label_data, args=(random_post.index, 0))
    st.button('Good', on_click=label_data, args=(random_post.index, 1))


@db.streamlit('/see-labels')
def see_labels():
    df = db.dataframes.get(DATA_KEY)
    labeled = df[df['relevance'] != -1]
    st.dataframe(labeled)
