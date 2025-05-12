import streamlit as st

st.title("Hello Chai App")
st.subheader("Brewed with streamlit")
st.text("Welcome to first interactive applliation!")
st.write("write something!!")
st.write([1, 2, 3])
st.write({
    "name": "srikar",
    "age": 20
})

chai = st.selectbox("Select your favourite chai: ", ["masala chai", "ginger tea", "iced lemon tea", "oolong tea", "kesar chai", "black tea", "green tea"])
st.write(f"You choose {chai}")