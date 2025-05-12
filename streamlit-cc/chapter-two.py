import streamlit as st

st.title("Chai Maker App")

if st.button("Make Your Own Chai"):
    st.success("Your chai is being brewed")

add_masala = st.checkbox("Add Masala")

if add_masala:
    st.write("Masala is added to chai!!")

flavours = st.selectbox("add required favours", ["ginger", "elaichi", "tulasi", "mint", "kesar"])

sugar = st.slider("sugar level", 0, 5, 2) # "2" is the default value

st.write(f'no of sugar spoons are : {sugar}')

cups = st.number_input("How Many Cups? ", min_value=1, max_value=10)

st.write(f"ordering {cups} cups of chai......")

name = st.text_input("Enter Your Name")

if name:
    st.write(f"Welcome {name} your chai is on the way!!")