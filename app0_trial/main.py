import streamlit as st
import langchain_helper

st.title("Startup name generator")

domain = st.text_input("Enter domain")


if domain:
    response = langchain_helper.generate_names_and_services(domain)
    st.header(response["name"].strip())
    st.write("**services**")
    services = response["services"].strip().split(",")
    for i in services:
        st.write("-",i)
