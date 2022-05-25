import databutton as db
import streamlit as st


@db.streamlit(route="/hello", name="Hello Databutton")
def hello():
    st.title("Hello, Databutton")


@db.streamlit(route="/tutorial", name="Tutorial")
def tutorial():
    st.title("Tutorial")


@db.streamlit(route="/compute", name="Compute intensive app", cpu=8, memory="16Gi")
def compute():
    st.title("Compute")


@db.repeat(seconds=10 * 60)
def repeating_job():
    # Check for new data
    # Do some work on that data
    # Write that data to db.dataframes
    # Send slack notification
    print("Success!")
