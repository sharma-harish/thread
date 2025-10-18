import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph_project.state import AgentState
from langgraph_project.agents.agents import (
    Orchestrator,
    GeneralQuestionAgent,
    AskMoreInfoAgent,
    RAGAgent,
    GenerateResponseAgent
)
from langgraph_project.agents.configuration import AgentsConfiguration
from langgraph_project.tools.documentation_retriever_tool import retrieve_documentation_vectorbase
from langgraph_project.utils import create_tool_node_with_fallback, save_graph_image, _print_event
from langgraph_project.tools.user_retriever_tool import retrieve_user_information_vectorbase
from langgraph_project.settings import settings
import os
import weave

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# remove all unnecessary logs
logging.getLogger("httpx").setLevel(logging.WARNING)


main_logger = logging.getLogger("main")
main_logger.debug("Debug Mode Active")
os.environ['WANDB_API_KEY']=settings.wandb_api_key

# Define the graph
builder = StateGraph(AgentState)
weave.init('iams-harish-na/thread')
# Orchestrator
builder.add_node("orchestrator", Orchestrator())
# Ask for more info agent
builder.add_node("ask_user_more_info_agent", AskMoreInfoAgent())
# Responds to a general question (no RAG)
builder.add_node("respond_to_general_question_agent", GeneralQuestionAgent())
# RAG agent - with the tool for retrieving information
builder.add_node("rag_user_agent", RAGAgent(
    AgentsConfiguration.llm.bind_tools([retrieve_user_information_vectorbase])
))
builder.add_node("rag_documentation_agent", RAGAgent(
    AgentsConfiguration.llm.bind_tools([retrieve_documentation_vectorbase])
))
# Tools nodes
builder.add_node(
    "retrieve_user_information_vectorbase",
    create_tool_node_with_fallback([retrieve_user_information_vectorbase])
)

builder.add_node(
    "retrieve_documentation_vectorbase",
    create_tool_node_with_fallback([retrieve_documentation_vectorbase])
)
# Generate responses after executing RAG
builder.add_node("generate_response_agent", GenerateResponseAgent())

# Edges
builder.add_edge(START, "orchestrator")

def route_agent(
    state: AgentState
) -> Literal["rag_user_agent", "ask_user_more_info_agent", "respond_to_general_question_agent"]:
    _type = state["message"]["type"]
    if _type == "user":
        return "rag_user_agent"
    elif _type == "documentation":
        return "rag_documentation_agent"
    elif _type == "more-info":
        return "ask_user_more_info_agent"
    elif _type == "general":
        return "respond_to_general_question_agent"
    else:
        raise ValueError(f"Unknown router type {_type}")

builder.add_conditional_edges(
    "orchestrator", route_agent, ["rag_user_agent", "rag_documentation_agent", "ask_user_more_info_agent", "respond_to_general_question_agent"]
)
builder.add_edge("rag_user_agent", "retrieve_user_information_vectorbase")
builder.add_edge("retrieve_user_information_vectorbase", "generate_response_agent")
builder.add_edge("rag_documentation_agent", "retrieve_documentation_vectorbase")
builder.add_edge("retrieve_documentation_vectorbase", "generate_response_agent")

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"

save_graph_image(graph, "./img", "graph.png")

with weave.attributes({"my_awesome_attribute": "value"}):
    result = graph.invoke({"messages": [("user", "what is the primary contact number for carol")]})

print(result)
# _printed = set()
# events = graph.stream(
#     {"messages": ("user", "What is John Doe's account number?")}, stream_mode="values" # RAG
#     # {"messages": ("user", "Can you tell me John's number?")}, stream_mode="values" # More info
#     # {"messages": ("user", "Which stocks showed the most growth in the last 5 years?")}, stream_mode="values" # General
#
# )
#
# for event in events:
#     _print_event(event, _printed)