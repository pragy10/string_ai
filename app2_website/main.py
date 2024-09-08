import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS


st.title("Information Extraction")

domain = st.text_input("Website URL")

url_button = st.button("Go!")

main_placeholder = st.empty()

from secret_keys_2 import google_api
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = google_api
api_key = os.environ["GOOGLE_API_KEY"]

llm = GoogleGenerativeAI(model="gemini-pro", max_output_tokens=250)

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if url_button and domain:
    loader = UnstructuredURLLoader(urls=[domain])
    main_placeholder.text("Loading URL...")
    
    try:
        data = loader.load()

        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", "-", " "]
        )

        main_placeholder.text("Splitting text...")
        doc = r_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_index = FAISS.from_documents(doc, embeddings)

        st.session_state.vector_index = vector_index

        main_placeholder.text("Vector DB ready...")

    except Exception as e:
        main_placeholder.text(f"Error loading URL: {str(e)}")


query = st.text_input("Query:")

if query and st.session_state.vector_index:
    retriever = st.session_state.vector_index.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    result = chain({"question": query}, return_only_outputs=True)

    st.header("Answer")
    st.write(result["answer"])
