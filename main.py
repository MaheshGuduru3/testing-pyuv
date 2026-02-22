from dartgpt_rag.src.document_chat.chat import Chat
from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from dartgpt_rag.model.chat_model.chat_model import ChatModel



# roufe work
# from dartgpt_rag.src.graph.graph_route import State
from langgraph.graph import START , END , StateGraph 
from langgraph.prebuilt import ToolNode , tools_condition
# from IPython.display import Image , 
from typing import Literal,List,Any
from pydantic import BaseModel
from langchain_tavily import TavilySearch
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults


# wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, load_all_available_meta=True))

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
tavily_tool = TavilySearch(
    tavily_api_key=""
)
class State(BaseModel):
    messages: List[Any]
    choice: Literal["RAG", "LLM"]
def rag_model_chat(state: State):
    chats = Chat()
    result = chats.chat_with_docs(state.messages[-1].content)

    return {
        "messages": state.messages + [AIMessage(content=result)]
    }
def tavily_model_chat(state: State):
    model = ChatModel().load_chat_model().bind_tools([tavily_tool])
    response = model.invoke(state.messages)
    print(response,'hcfg')
    return {
        "messages": state.messages + [response]
    }
def get_model_chat(state: State):
    return state.choice
graph = StateGraph(State)

graph.add_node("RAG", rag_model_chat)
graph.add_node("LLM", tavily_model_chat)
graph.add_node("TOOLS", ToolNode([tavily_tool]))

graph.add_conditional_edges(
    START,
    get_model_chat,
    {
        "RAG": "RAG",
        "LLM": "LLM",
    }
)

graph.add_edge("RAG", END)

graph.add_conditional_edges(
    "LLM",
    tools_condition,
)

graph.add_edge("TOOLS", "LLM")

# graph.add_edge("LLM", END)

app = graph.compile()

res = app.invoke(
    {
        "messages": [HumanMessage(content="what is java")],
        "choice": "LLM",
    }
)

print(res)

print(res,'hgvhg')

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with uv!"}


@app.post('/chat')
async def chat_with_rag(query):
    chats = Chat()
    results = chats.chat_with_docs(query)
    return results


DATA_DIR = Path("data")  # already exists

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    print(file)
    # Validate PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Secure filename
    filename = Path(file.filename).name
    file_path = DATA_DIR / filename

    # Check if file already exists
    if file_path.exists():
        return {
            "status": "exists",
            "message": f"File '{filename}' already exists"
        }

    # Save the PDF
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "saved",
        "message": "PDF uploaded successfully",
        "filename": filename
    }




def main():
    print("Hello from pypractice!")


if __name__ == "__main__":
    main()  
