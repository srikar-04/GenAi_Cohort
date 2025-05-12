import streamlit as st

st.header("Chai Taste Poll")

col1, col2 = st.columns(2)

with col1:
    st.header("masala chai")
    votel = st.button("vote masala chai")

with col2:
    st.header("adrak chai")
    vote2= st.button("vote adrak chai")

if votel:
    st.success("Thanks for choosing masala chai.")
elif vote2:
    st.success("Thanks for voting adrak chai.")

st.sidebar.header("Sidebar")

name = st.sidebar.text_input("Enter your name")
tea = st.sidebar.selectbox("Choice of tea", ["ginger", "iced-lemon", "kesar"])

st.sidebar.write(f"Welcome {name}! your {tea} is getting ready!")

st.header("Expander")
with st.expander("chai making instructions"):
    st.write("""
    1. with sugar at base, boil water to caremalize sugar.
    2. add crushed ginger in the boiling water and spices you like.
    3. add tea leaves and boil on high flame fo 2 mins
    4. add milk to it and let it boil for a while
    5. your tea is ready, serve hot ☕
""")