import streamlit as st
import secret_keys_3

st.title("Indian Railways")


domain = st.text_input("Enter domain")


if domain:
    response = langchain_helper.generate_names_and_services(domain)
    st.header(response["name"].strip())
    st.write("**services**")
    services = response["services"].strip().split(",")
    for i in services:
        st.write("-",i)

