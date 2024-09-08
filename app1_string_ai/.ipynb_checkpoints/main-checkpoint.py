import streamlit as st
import os
import pickle 
import langchain
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
import warnings
warnings.filterwarnings("ignore")

st.title("String AI")

st.text("Queries: ")
query = st.text_input("Ask your doubts!")

if query:
    from secret_keys_1 import google_api

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = google_api

    api_key = os.environ["GOOGLE_API_KEY"]

    llm = GoogleGenerativeAI(model = "gemini-pro",max_output_tokens=250)

    loader = TextLoader("ondc.txt", encoding='utf-8')
    data = loader.load()

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 0,
        separators = ["\n\n","\n",".","-"," "]
    )

    doc = r_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_index = FAISS.from_documents(doc, embeddings)

    file_path = "vector_index_file.pkl"
    with open(file_path,"wb") as f:
        pickle.dump(vector_index,f)

    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorIndex = pickle.load(f)

    retriever = vectorIndex.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = retriever)

    langchain.debug = True
    chain({"question":query},return_only_outputs= True)
    chain

    


    

