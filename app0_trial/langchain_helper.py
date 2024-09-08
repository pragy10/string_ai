from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCthpvKpnO67ykBVNPRw6Oevzv__vFXYPs"

api_key = os.environ["GOOGLE_API_KEY"]

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.7)

def generate_names_and_services(domain):
    pt_name1 = PromptTemplate(
        input_variables = ['domain'],
        template = "give A SINGLE name for a {domain} startup"
    )

    pt_name2 = PromptTemplate(
        input = ['name'],
        template = "suggest some services that {name} provides"
    )

    op = StrOutputParser()

    chain1 = LLMChain(prompt = pt_name1, llm = llm, output_parser = op, output_key = 'name')
    chain2 = LLMChain(prompt = pt_name2, llm = llm, output_parser = op, output_key = 'services')

    sq_chain = SequentialChain(
        chains = [chain1, chain2],
        input_variables = ['domain'],
        output_variables = ['name','services']
    )

    response = sq_chain(domain)
    return response