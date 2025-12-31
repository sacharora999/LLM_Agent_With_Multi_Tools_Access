## Send API to dynamically trigger the worker

import os
from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)


from typing_extensions import Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, List
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


class Section(BaseModel):
    name:str=Field(description="Name for this section of the report")
    description:str=Field(description="Brief Overfiew of topics")


class Sections(BaseModel):
    sections:List[Section]=Field(description="Sections of the report")

llm_with_structure=llm.with_structured_output(Sections)

from langgraph.constants import Send
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    topic: str
    sections: List[Section]
    completed_sections: Annotated[
        list, operator.add
    ]
    final_report: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[
        list, operator.add
    ]  



from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
def orchestrator(state: State):
    """orchestrator that generates plan for the report"""

    report_sections = llm_with_structure.invoke(
        [
            SystemMessage(content="Generate a plan for the report"),
            HumanMessage(content=f"Here is the report topic : {state['topic']}")
        ]
    )

    print("Report Sections : ", report_sections)
    return {"sections" : report_sections.sections}


def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no premmble for each section. Use markdown formatting"
            ),
            HumanMessage(
                content=f"Here is the section name : {state['section'].name} and description : {state['section'].description}"
            )
        ]
    )
    return {"completed_sections": [section.content]}    

def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    return [Send("llm_call", {"section": s}) for s in state["sections"]]


def synthesizer(state: State):
    """synthesize full report from sections"""

    completed_sections = state["completed_sections"]

    completed_report_sections = "\n\n--\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}



from langgraph.graph import StateGraph, START, END

graph=StateGraph(State)

graph.add_node("orchestrator", orchestrator)
graph.add_node("llm_call", llm_call)
graph.add_node("synthesizer", synthesizer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
graph.add_edge("llm_call", "synthesizer")
graph.add_edge("synthesizer", END)


graph_builder = graph.compile()
display(Image(graph_builder.get_graph().draw_mermaid_png()))

state=graph_builder.invoke({"topic": "Create a report on Agentic AI RAGs"})
from IPython.display import Markdown
Markdown(state["final_report"])

