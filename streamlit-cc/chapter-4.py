import streamlit as st
import pandas as pd

st.title("Chai Sales Dashboard")

file = st.file_uploader("Enter your csv file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Data Preview: ")
    st.dataframe(df)