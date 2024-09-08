import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
import mysql.connector
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI



st.title("Indian Railways")

from secret_keys_3 import google_api
os.environ["GOOGLE_API_KEY"] = google_api
#os.environ["OPENAI_API_KEY"] = openai_api


template = """Based on the table schema below, write ONLY the SQL query that would answer the user's question. do not give explanations of the answer. and display the query as string format:
{schema}
Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

db_uri = {
    "user": "root",
    "password": "042005",
    "host": "localhost",
    "database": "new_schema"
}

db_uri_1 = "mysql+mysqlconnector://root:042005@localhost:3306/new_schema"
db = SQLDatabase.from_uri(db_uri_1)

cnx = mysql.connector.connect(**db_uri)
cursor = cnx.cursor()

cursor.execute("SHOW TABLES")

table_names = [row[0] for row in cursor.fetchall()]

for table_name in table_names:
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    column_names = [row[0] for row in cursor.fetchall()]
    st.write(f"Table name: {table_name}")
    st.write("Columns: "+", ".join(map(str, column_names)))

cursor.close()
cnx.close()

st.header("Queries: ")
query = st.text_input("Ask question")

def get_schema(_):
    return db.get_table_info()

get_schema(None)

if query:
    llm =  ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    # llm =  ChatOpenAI()

    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind()
        | StrOutputParser()
    )

    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_template(template)

    def run_query(query):
        return db.run(query)

    full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=get_schema,
            response=lambda vars: run_query(vars["query"].split("\n")[1]),
        )
        | prompt_response
        | llm
        | StrOutputParser()
    )

    answer = full_chain.invoke({"question":query})
    st.write(answer)


