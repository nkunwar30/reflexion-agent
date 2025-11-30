from dotenv import load_dotenv

from typing import List

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.prebuilt import ToolNode

from chain import revisor, first_responder
from tool_executor import execute_tools

load_dotenv()


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def draft_node(state: State) -> State:
    response = first_responder.invoke(input={"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: State) -> State:
    response = revisor.invoke(input={"messages": state["messages"]})
    return {"messages": [response]}


MAX_ITERATIONS = 2
builder = StateGraph(State)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: State) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        print(f"Reached max iterations: {MAX_ITERATIONS}, terminating.")
        return END
    return "execute_tools"


builder.add_conditional_edges(
    "revise",
    event_loop,
    {
        "execute_tools": "execute_tools",
        END: END,
    },
)

# builder.set_start_node("draft")
builder.add_edge(START, "draft")
graph = builder.compile()

# print(graph.get_graph().draw_ascii())
png_bytes = graph.get_graph().draw_mermaid_png()
with open("reflexion_agent_graph.png", "wb") as f:
    f.write(png_bytes)

if __name__ == "__main__":
    print("Hello Reflexion-Agent")

    state = {
        "messages": [
            HumanMessage(
                content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
            ),
        ]
    }

    print("\n Running Reflexion-Agent Graph...\n")

    for step in graph.stream(input=state, stream_mode="values"):
        msg = step["messages"][-1]
        print(msg)
        print("\n............................................\n")

    print("Reflection finished.\n\n")

    """



    res= graph.invoke(
        input={
            "messages": [
                {
                    "type": "human",
                    "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
                }
            ]
        }
    )
    print(res[-1].tool_calls[0]["arguments"]["answer"])
    """
