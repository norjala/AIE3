import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from qdrant_client import QdrantClient
from qdrant_client import models  # Add this import as well

# Load environment variables from .env file
load_dotenv()

# Environment variables
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]
VECTOR_STORE_PATH = "/home/user/app/data/vectorstore"

# Document loader
document_loader = PyMuPDFLoader("./data/Airbnb-10k.pdf")
documents = document_loader.load()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

# Load embeddings
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to create vector store
def create_vector_store():
    print("Indexing Files")
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    client = QdrantClient(path=VECTOR_STORE_PATH)
    
    # Create the collection first
    client.create_collection(
        collection_name="Airbnb-10k",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    
    # Fix: Change embed_document to embed_documents
    vectors = openai_embeddings.embed_documents([doc.page_content for doc in split_documents])
    payload = [{"page_content": doc.page_content} for doc in split_documents]
    
    client.upload_collection(
        collection_name="Airbnb-10k",
        vectors=vectors,
        payload=payload,
        batch_size=32,
    )
    
    print("Vectorstore created and documents indexed.")
    return client

# Function to check if collection exists
def check_collection_exists():
    client = QdrantClient(path=VECTOR_STORE_PATH)
    collections = client.get_collections()
    return any(collection.name == "Airbnb-10k" for collection in collections.collections)

# Check if vector store path exists and load or create vector store
if os.path.exists(VECTOR_STORE_PATH):
    if check_collection_exists():
        client = QdrantClient(path=VECTOR_STORE_PATH)
        print("Loaded Vectorstore")
    else:
        print("Collection not found. Creating a new collection.")
        client = create_vector_store()
else:
    client = create_vector_store()

retriever = Qdrant(
    client=client, 
    collection_name="Airbnb-10k",
    embeddings=openai_embeddings
).as_retriever(search_kwargs={"k": 2})  # Limit to 2 documents    client=client, 

# Define the prompt template
RAG_PROMPT_TEMPLATE = """\
\
system
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context,\
    say you don't know.

\
user    
User Query:
{query}

Context:
{context}

\
assistant
"""
#Note that we do not have the response here. We have assistent, we ONLY start, but not followed by <|eot_id> as we do not have a response YET.

### 2. CREATE PROMPT TEMPLATE
rag_prompt =PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""
### 1. CREATE HUGGINGFACE ENDPOINT FOR LLM
hf_llm = HuggingFaceEndpoint(
    endpoint_url=f"{HF_LLM_ENDPOINT}",
    max_new_tokens=246,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)

@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 
    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "Airbnb 10k Filer"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 
    We will build our LCEL RAG chain here, and store it in the user session. 
    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = ( {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
                      
                       | rag_prompt | hf_llm
                    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.
    We will use the LCEL RAG chain to generate a response to the user query.
    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()