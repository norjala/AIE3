import os
import chainlit as cl
import tiktoken
import openai
import fitz  
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings 
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI 
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Set environment variables
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Load embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

loader = PyMuPDFLoader("./data/Airbnb-10k.pdf")
documents = loader.load()

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=50,
    length_function = tiktoken_len
)

split_documents = text_splitter.split_documents(documents)

# Creating a Qdrant vector store
qdrant_vector_store = Qdrant.from_documents(
    split_documents,
    embeddings,
    location=":memory:",
    collection_name="Airbnb-10k",
)

# Create a retriever
retriever = qdrant_vector_store.as_retriever()

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
system
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context,\
    say you don't know. However, if it is a general question about the company, you can answer it and say that you are getting this information from the open web.
User Query:
{query}
Context:
{context}
assistant
"""
# Note that we do not have the response here. We have assistant, we ONLY start, but not followed by <|eot_id> as we do not have a response YET.

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Define the LLM
llm = ChatOpenAI(model_name="gpt-4o")

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"query" : "<>"}
    # "query" : populated by getting the value of the "query" key
    # "context"  : populated by getting the value of the "query" key and chaining it into the base_retriever
    {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "query" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | llm, "context": itemgetter("context")}
)

# Sets initial chat settings 
@cl.on_chat_start  
async def start_chat():
    """
    This function will be called at the start of every user session. 
    We will build our LCEL RAG chain here, and store it in the user session. 
    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """
    settings = {
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 500,
        "frequency_penalty": 0,
        "top_p": 1,
        
    }
    cl.user_session.set("settings", settings)

# Initializes user session w/ settings and retrieves context based on query

@cl.on_message 
async def handle_message(message: cl.Message):
    settings = cl.user_session.get("settings")

    response = retrieval_augmented_qa_chain.invoke({"query": message.content})

    # Retrieve and sends content back to user
    content = response["response"].content
    pretty_content = content.strip()  

    await cl.Message(content=pretty_content).send()
