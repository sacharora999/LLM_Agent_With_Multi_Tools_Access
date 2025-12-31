import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

from langchain.tools import tool

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
print(arxiv.name)
arxiv.invoke("Attention is all you need")


api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
print(wiki.name)


from langchain_core.messages import AIMessage, HumanMessage
from pprint import pprint

tools=[arxiv,wiki]

llm_with_tools=llm.bind_tools(tools=tools)
result = llm_with_tools.invoke([HumanMessage(content="What is attention is all you need ?", name="Sachin")]).tool_calls


from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

def superbot(state: State):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}

graph=StateGraph(State)

graph.add_node("superbot", superbot)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "superbot")
graph.add_conditional_edges(
    "superbot",
    # If the latest message from assistant is a tool call -> tools_condition routes to tool
    # If the latest message from assistant is not a tool call -> tools_condition routes to END
    tools_condition
)
graph.add_edge("tools", END)
graph_builder = graph.compile()

from IPython.display import Image, display
display(Image(graph_builder.get_graph().draw_mermaid_png()))

graph_builder.invoke({'messages': "What is AI"})
