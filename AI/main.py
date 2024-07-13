from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq # Chage this to use gemini langchain library
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# This too and call it llm_gemini instead of just llm
#Initialize Model
"""llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key="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"
)"""

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key="AIzaSyD7MQMIz5Kg0l31A5y7vHDFc_E8YENrc7o")



url = input("Article Link: ")
#Load the blog
loader = WebBaseLoader(url)
docs = loader.load()

#Define the Summarize Chain
template = """Write a clear summary of the following in 6 or more points:
"{text}"
CONCISE SUMMARY:"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(llm=llm_gemini, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

#Invoke Chain
response=stuff_chain.invoke(docs)
print(response["output_text"])