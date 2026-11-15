import ast
import json
import math
import operator
import os
import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command


load_dotenv()

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent
DOCS_PATH = BASE_DIR / "extracted_docs"      # full text, one file per chat
INDEX_PATH = BASE_DIR / "faiss_indexes"      # vector index, one folder per chat
DOCS_PATH.mkdir(exist_ok=True)
INDEX_PATH.mkdir(exist_ok=True)

MAX_HISTORY_MESSAGES = 30    # only the last N messages are sent to the model
CHAPTER_PART_SIZE = 10000    # characters returned per get_chapter call

# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b", temperature=0, timeout=30, max_retries=1
)

# Loaded only when first needed, so the app starts fast
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


# ============================================================
# PER-CHAT DOCUMENT STORAGE
# ============================================================

_store_cache: dict[str, FAISS] = {}


def _safe_id(thread_id: str) -> str:
    """Keep only safe characters so a thread id can never escape the folder."""
    return re.sub(r"[^A-Za-z0-9_-]", "", str(thread_id))


def _text_file(thread_id: str) -> Path:
    return DOCS_PATH / f"{_safe_id(thread_id)}.txt"


def _meta_file(thread_id: str) -> Path:
    return DOCS_PATH / f"{_safe_id(thread_id)}.json"


def _index_dir(thread_id: str) -> Path:
    return INDEX_PATH / _safe_id(thread_id)


def get_document_name(thread_id: str) -> str | None:
    """Name of the PDF attached to this chat, or None."""
    if not thread_id or not _text_file(thread_id).exists():
        return None
    try:
        return json.loads(_meta_file(thread_id).read_text(encoding="utf-8")).get(
            "file_name", "document.pdf"
        )
    except Exception:
        return "document.pdf"


def ingest_rag_document(file_path: str, thread_id: str, file_name: str = "document.pdf") -> dict:
    """Read the PDF, save full text, build a vector index. Both belong to one chat."""
    pages = PyPDFLoader(file_path).load()

    if not pages:
        raise ValueError("No pages could be read from this PDF.")

    total_chars = sum(len(p.page_content.strip()) for p in pages)
    if total_chars < 100:
        raise ValueError(
            "No readable text found. This looks like a scanned PDF (images only). "
            "It needs OCR first."
        )

    # 1) Full text with page markers (used by get_chapter)
    parts = []
    for i, page in enumerate(pages):
        page_no = page.metadata.get("page", i) + 1
        page.metadata["page"] = page_no
        parts.append(f"===== PAGE {page_no} =====\n{page.page_content}")

    _text_file(thread_id).write_text("\n\n".join(parts), encoding="utf-8")

    # 2) Chunks + vector index (used by rag_tool)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(_index_dir(thread_id)))
    _store_cache[thread_id] = store

    _meta_file(thread_id).write_text(
        json.dumps({"file_name": file_name}), encoding="utf-8"
    )

    return {"pages": len(pages), "chunks": len(chunks)}


def _load_store(thread_id: str):
    if thread_id in _store_cache:
        return _store_cache[thread_id]

    index_dir = _index_dir(thread_id)
    if not index_dir.exists():
        return None

    # Safe here: we created this index ourselves
    store = FAISS.load_local(
        str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
    )
    _store_cache[thread_id] = store
    return store


def _thread_id_from(config: RunnableConfig) -> str | None:
    return (config or {}).get("configurable", {}).get("thread_id")


# ============================================================
# TOOLS
# ============================================================

@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Search the uploaded PDF for passages related to a question or topic.

    Use for specific questions about the document (facts, definitions,
    "where does it talk about X"). For a full chapter summary, use get_chapter.

    Args:
        query: The question or topic to search for.
    """
    thread_id = _thread_id_from(config)
    store = _load_store(thread_id) if thread_id else None

    if store is None:
        return "No PDF has been uploaded in this chat yet. Ask the user to upload one."

    results = store.similarity_search(query, k=5)

    if not results:
        return "No relevant passages found in the document."

    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in results
    )


_NUM_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]


def _chapter_pattern(n: int) -> re.Pattern:
    word = _NUM_WORDS[n] if n < len(_NUM_WORDS) else str(n)
    return re.compile(rf"(?im)^[ \t]*chapter[ \t]+(?:{n}|{word})\b")


@tool
def get_chapter(chapter_number: int, config: RunnableConfig, part: int = 1) -> str:
    """
    Get the full text of one chapter from the uploaded PDF.
    Use for requests like "summarize chapter 4" or "show the code in chapter 4".

    Args:
        chapter_number: The chapter number, for example 4.
        part: Which part to read (long chapters are split). Starts at 1.
    """
    thread_id = _thread_id_from(config)
    if not thread_id or not _text_file(thread_id).exists():
        return "No PDF has been uploaded in this chat yet. Ask the user to upload one."

    text = _text_file(thread_id).read_text(encoding="utf-8")

    start_pat = _chapter_pattern(chapter_number)
    end_pat = _chapter_pattern(chapter_number + 1)

    # A chapter title often appears twice (table of contents + real start).
    # The real chapter is the longest match.
    best = ""
    for match in start_pat.finditer(text):
        end = end_pat.search(text, match.end())
        section = text[match.start(): end.start() if end else len(text)]
        if len(section) > len(best):
            best = section

    if not best:
        return (
            f"Could not find a heading for Chapter {chapter_number}. "
            "Try rag_tool with the chapter's topic instead."
        )

    total_parts = max(1, math.ceil(len(best) / CHAPTER_PART_SIZE))
    part = min(max(part, 1), total_parts)
    start = (part - 1) * CHAPTER_PART_SIZE
    piece = best[start: start + CHAPTER_PART_SIZE]

    result = f"Chapter {chapter_number} (part {part} of {total_parts}):\n\n{piece}"
    if part < total_parts:
        result += (
            f"\n\n[More content remains. "
            f"Call get_chapter again with part={part + 1}.]"
        )
    return result


search_tool = TavilySearch(max_results=5, topic="general")


# --- Safe calculator (no eval) ---

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "factorial": math.factorial,
}

_CONSTANTS = {"pi": math.pi, "e": math.e}


def _calc(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _calc(node.left), _calc(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_calc(node.operand))

    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_calc(arg) for arg in node.args])

    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """
    Do math calculations.
    Supports + - * / // % **, parentheses, pi, e and these functions:
    sqrt, abs, round, min, max, log, log10, sin, cos, tan, factorial.
    Example: 2 + 2, sqrt(16), 10 * 5
    """
    try:
        expression = expression.replace("math.", "").strip()
        if len(expression) > 200:
            return "Calculation error: expression is too long."
        tree = ast.parse(expression, mode="eval")
        return str(_calc(tree.body))
    except Exception as error:
        return f"Calculation error: {error}"


# Common company names -> ticker symbols (so "google" works, not just "GOOGL")
_COMPANY_TICKERS = {
    "google": "GOOGL", "alphabet": "GOOGL", "apple": "AAPL",
    "microsoft": "MSFT", "amazon": "AMZN", "tesla": "TSLA",
    "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "netflix": "NFLX", "reliance": "RELIANCE.NS", "tcs": "TCS.NS",
    "infosys": "INFY.NS", "hdfc bank": "HDFCBANK.NS",
}


def _clean_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    return _COMPANY_TICKERS.get(symbol.lower(), symbol.upper())


def _price_from_yahoo(symbol: str) -> str | None:
    """Free, no API key. Returns a result string, or None if it fails."""
    try:
        import yfinance as yf

        history = yf.Ticker(symbol).history(period="5d")
        if history.empty:
            return None

        last = history.iloc[-1]
        price = float(last["Close"])
        volume = int(last["Volume"])
        day = history.index[-1].strftime("%Y-%m-%d")

        change_text = "change n/a"
        if len(history) > 1:
            previous = float(history.iloc[-2]["Close"])
            change = price - previous
            percent = change / previous * 100
            change_text = f"change {change:+.2f} ({percent:+.2f}%)"

        return (
            f"{symbol}: price {price:.2f}, {change_text}, "
            f"volume {volume:,}, last trading day {day} "
            "(source: Yahoo Finance, may be delayed up to 15 minutes)"
        )
    except Exception:
        return None


def _price_from_alphavantage(symbol: str) -> str | None:
    """Backup. Only runs if ALPHAVANTAGE_API_KEY is set."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    quote = data.get("Global Quote") or {}
    if not quote:
        return None

    return (
        f"{symbol}: price {quote.get('05. price')}, "
        f"change {quote.get('09. change')} ({quote.get('10. change percent')}), "
        f"volume {quote.get('06. volume')}, "
        f"last trading day {quote.get('07. latest trading day')} "
        "(source: Alpha Vantage)"
    )


@tool
def get_stock_price(symbol: str) -> str:
    """
    Get the latest stock price for a ticker symbol or company name,
    such as 'AAPL', 'TSLA', 'GOOGL' or 'google'.
    For Indian stocks use the .NS suffix, for example 'RELIANCE.NS'.
    """
    symbol = _clean_symbol(symbol)

    # Try Yahoo first (no key needed), then Alpha Vantage if a key exists
    for source in (_price_from_yahoo, _price_from_alphavantage):
        result = source(symbol)
        if result:
            return result

    return (
        f"Could not get a price for '{symbol}'. "
        "The symbol may be wrong, or the price services are not reachable right now."
    )


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }




@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a city or location.

    Args:
        location: City name, for example "Agra", "London, UK" or "New York, US".
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()
        if not locations:
            return f"Could not find the location: {location}"

        place = locations[0]

        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": place["lat"],
                "lon": place["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        data = weather_response.json()

        visibility_m = data.get("visibility")
        visibility_km = round(visibility_m / 1000, 1) if visibility_m is not None else "N/A"

        display_location = ", ".join(
            part
            for part in [place.get("name", location), place.get("state", ""), place.get("country", "")]
            if part
        )

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {data['weather'][0]['description'].title()}\n"
            f"- Temperature: {data['main']['temp']}°C\n"
            f"- Feels like: {data['main']['feels_like']}°C\n"
            f"- Humidity: {data['main']['humidity']}%\n"
            f"- Pressure: {data['main']['pressure']} hPa\n"
            f"- Wind speed: {data.get('wind', {}).get('speed', 'N/A')} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        if status == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"


tools = [
    search_tool, calculator, get_stock_price, purchase_stock,
    get_current_weather, rag_tool, get_chapter,
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# STATE AND NODES
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node: answers directly or calls a tool."""

    thread_id = _thread_id_from(config)
    doc_name = get_document_name(thread_id) if thread_id else None

    if doc_name:
        doc_status = (
            f"A PDF named '{doc_name}' IS uploaded and ready in this chat. "
            "For ANY question about the document, its chapters, pages or code, "
            "you MUST call a document tool first. "
            "Never ask the user to upload the PDF again."
        )
    else:
        doc_status = (
            "No PDF is uploaded in this chat. "
            "If the user asks about a PDF, ask them to upload one."
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            f"Document status: {doc_status}\n\n"
            "Tool rules:\n"
            "- `get_chapter`: use when the user asks about a whole chapter "
            "(summary, code, explanation). If the result says more parts remain, "
            "call it again with the next part (up to 3 parts) before answering.\n"
            "- `rag_tool`: use for specific questions about the PDF.\n"
            "- `search_tool`: use for current events or internet search.\n"
            "- `calculator`: use for math.\n"
            "- `get_stock_price`: use for stock prices. Pass the ticker symbol.\n"
            "- `purchase_stock`: use only when the user clearly asks to buy shares. "
            "It asks the human for approval first.\n"
            "- `get_current_weather`: use for current weather.\n\n"
            "Answer general questions directly. Never invent document content. "
            "If a document tool finds nothing, say so. "
            "When showing code from the document, keep it in code blocks and "
            "tell the user if the indentation may be broken by PDF extraction. "
            "After a tool result, give a clear final answer."
        )
    )

    # Send only the recent messages so long chats don't get slow or too big
    history = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=MAX_HISTORY_MESSAGES,
        start_on="human",
        include_system=False,
    )
    if not history:
        history = state["messages"]

    response = llm_with_tools.invoke([system_message, *history])
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ============================================================
# CHECKPOINTER AND GRAPH
# ============================================================

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


# ============================================================
# HELPERS FOR THE STREAMLIT FRONTEND
# ============================================================

def get_all_threads():
    """All thread ids, oldest first (the frontend reverses this to show newest first)."""
    ordered = []
    seen = set()

    # SqliteSaver lists the newest checkpoints first
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.add(thread_id)
            ordered.append(thread_id)

    return ordered[::-1]