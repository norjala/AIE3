import os
import getpass
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
#from langchain.tools import tool
#from langchain_community.agent_toolkits import SQLDatabaseToolkit
#from langchain_community.agent_toolkits import SQLDatabaseToolkit
#from langchain.sql_database import SQLDatabase
#from sqlalchemy import create_engine
from langchain.tools import tool
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
#from qdrant_client import QdrantClient
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
import os
import chainlit as cl
from dotenv import load_dotenv
load_dotenv(override=True)



from uuid import uuid4

unique_id = uuid4().hex[0:8]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"SquadAI - {unique_id}"

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

document_loader = CSVLoader("./squadusersinfo.psv", csv_args={'delimiter': '|'})
documents = document_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
#client = QdrantClient(location=":memory:")
hf_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



for i in range(0, len(split_documents), 32):
  if i == 0:
    vectorstore = FAISS.from_documents(split_documents[i:i+32], hf_embeddings)
    continue
  vectorstore.add_documents(split_documents[i:i+32])


hf_retriever = vectorstore.as_retriever()


RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. You will search the interest of user from the stored list of users. If you cannot find a match, you will look for the matches from the chat history.
If you still cannot find any match, respond with Sorry, at present there is no match for your interest and ask if this user can be matched if any future interest matches this user. 
Ask for all necessary details to consider this user as a future match.
If the question is outside of finding a match, say I don't know.
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

model = ChatOpenAI(model="gpt-4o")  #reduce inference cost


rag_chain = (
    {"context": itemgetter("question") | hf_retriever, "question": itemgetter("question")}
    | rag_prompt | model | StrOutputParser()
)


#result = rag_chain.invoke({"question" : "Anyone interested in squash?"})
#print(result)


@tool
def matchUser( query): 
    """A tool to find the details matching the user details from the store based on user query."""
    result = rag_chain.invoke({"question" : query})
    return result

tools= [TavilySearchResults(max_results=1), matchUser]

tool_executor = ToolExecutor(tools)


model = ChatOpenAI(temperature=0, streaming=True)

functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)



def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]   
    if "function_call" not in last_message.additional_kwargs:
        return "end"  
    else:
        return "continue"


def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)  
    return {"messages": [response]}


def call_tool(state):
    messages = state['messages'] 
    last_message = messages[-1]  
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )  
    response = tool_executor.invoke(action) 
    function_message = FunctionMessage(content=str(response), name=action.tool)   
    return {"messages": [function_message]}

def print_messages(messages):
  next_is_tool = False
  initial_query = True
  for message in messages["messages"]:
    if "function_call" in message.additional_kwargs:
      print()
      print(f'Tool Call - Name: {message.additional_kwargs["function_call"]["name"]} + Query: {message.additional_kwargs["function_call"]["arguments"]}')
      next_is_tool = True
      continue
    if next_is_tool:
      print(f"Tool Response: {message.content}")
      next_is_tool = False
      continue
    if initial_query:
      print(f"Initial Query: {message.content}")
      print()
      initial_query = False
      continue
    print()
    print(f"Agent Response: {message.content}")

def construct_response(messages):
  next_is_tool = False
  initial_query = True
  response = ""
  for message in messages["messages"]:
    if "function_call" in message.additional_kwargs:
      print()
      print(f'Tool Call - Name: {message.additional_kwargs["function_call"]["name"]} + Query: {message.additional_kwargs["function_call"]["arguments"]}')
      next_is_tool = True
      continue
    if next_is_tool:
      print(f"Tool Response: {message.content}")
      if "url" not in message.content:
        response = response + message.content
      next_is_tool = False
      continue
    if initial_query:
      print(f"Initial Query: {message.content}")
      print()
      initial_query = False
      continue
    print()
    print(f"Agent Response: {message.content}")
    response = response + message.content
    return response

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")


workflow.add_conditional_edges(
    "agent", 
    should_continue,
    {        
        "continue": "action",      
        "end": END
    }
)

workflow.add_edge('action', 'agent')

app = workflow.compile()

#messages = [HumanMessage(content="Any user named Ganesh. Where to play cricket")]
#inputs = {"messages" : [HumanMessage(content="Anyone interested in cricket? Provide more information about who is interested. Also get me some locations where I can play cricket in Toronto")]}

#result = app.invoke({"messages": messages})

#print_messages(result)
#messages = app.invoke(inputs)

#print_messages(messages)
@cl.on_message
async def run_convo(message: cl.Message):   
    msg = cl.Message(content="")
    await msg.send()  
    await cl.sleep(1) #hack to simulate loader!

    inputs = {"messages": [HumanMessage(content=message.content)]}

    res = app.invoke(inputs, config=RunnableConfig(callbacks=[
        cl.LangchainCallbackHandler(
            to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]      
        )]))
    
    content = construct_response(res)
    #for response in (res["messages"]):
     #   if message.content not in response:
      #      content = content+response.content
    await cl.Message(content=content).send() 
