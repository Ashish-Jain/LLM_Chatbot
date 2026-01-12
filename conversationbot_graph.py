import warnings
warnings.filterwarnings("ignore")

import httpx
import requests
import yfinance as yf
from duckduckgo_search import DDGS

from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.tools import Tool
from langchain_community.chat_message_histories import RedisChatMessageHistory

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool




llm = ChatGroq(model='moonshotai/kimi-k2-instruct-0905',groq_api_key='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',http_client=httpx.Client(verify=False))

# pdf_file = "India_Holidays_and_Gift_Policy.pdf"
# loader = PyPDFLoader(pdf_file)   # <-- your PDF file
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
# docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore.save_local("pdf_vectorstore")

@tool("pdf_knowledge_base")
def pdf_knowledge_base(query: str) -> str:
    """
      Use this tool to answer questions about the company holiday list and gift policy document.
    """
    print("start: Inside pdf_search with query {0}".format(query))
    """Get holiday details and company gift policy and process using India_Holidays_and_Gift_Policy.pdf"""
    vectorstore = FAISS.load_local(folder_path="pdf_vectorstore", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

@tool("get_dividends")
def get_dividends(symbol: str) -> str:
    """
    Get dividend information for a stock symbol
    """
    url = "https://api.massive.com/v3/reference/dividends"
    api_key = "q1bvkUCP8TDc__Q_uPLgQL2FupngRprR"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "ticker": symbol
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    response.raise_for_status()
    return response.json()

@tool("get_stock_info")
def get_stock_info(ticker: str) -> str:
    """
    Get stock price and company info using yfinance
    """
    print("start: Inside get stock Info with ticker {0}".format(ticker))
    stock = yf.Ticker(ticker)
    info = stock.info
    print("end : Inside get stock Info with ticker {0}".format(ticker))
    return f"""
    Company: {info.get('longName')}
    Ticker: {ticker}
    Price: {info.get('currentPrice')}
    Market Cap: {info.get('marketCap')}
    Currency: {info.get('currency')}
    """

@tool("web_search")
def web_search(query: str) -> str:
    """
    Search Indian news sources for the latest updates.
    """
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "country": "in",
        "max": 5,
        "apikey": "b84c97e68502419f4465e65fd3f926ca"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    articles = r.json().get("articles", [])
    return "\n\n".join(
        f"{a['title']} — {a['source']['name']}\n{a['url']}"
        for a in articles
        )
    # """
    # Search the web for latest information
    # """
    # print("start: Inside web_search with query {0}".format(query))
    # with DDGS(timeout=20) as ddgs:
    #     results = list(ddgs.text(query, max_results=3))
    # return str(results)

tools = [
    pdf_knowledge_base,
    web_search,
    get_stock_info,
    get_dividends,
]

class AgentState(TypedDict):
    messages: List[BaseMessage]

react_agent = create_react_agent(llm, tools)
tool_node = ToolNode(tools)

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return "__end__"

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", react_agent)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    graph.add_edge("tools", "agent")

    return graph.compile()

def start_chat(query: str, messages: list[BaseMessage]) -> tuple[str, list[BaseMessage]]:
    graph = build_graph()

    messages = messages + [HumanMessage(content=query)]

    result = graph.invoke({"messages": messages})

    messages = result["messages"]

    return result["messages"][-1].content
