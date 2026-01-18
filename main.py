import os
import uvicorn
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION & API KEYS ---

# 1. Load variables from .env file
load_dotenv()

# 2. Verify API Key is present
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY is missing from .env file or environment variables.")

# 3. Define the Model
# "llama3-8b-8192" is DEPRECATED. 
# We use "llama-3.1-8b-instant" which is the current fast/efficient standard.
# Other options: "llama-3.3-70b-versatile" (slower but smarter)
LLM_MODEL = "llama-3.1-8b-instant"

# --- PART 1: LANGGRAPH AGENT SETUP ---

# NEW: Define Task Types
class AgentState(TypedDict):
    url: str
    # Added "smart_extract" and "simplify"
    task_type: Literal["summarize", "smart_extract", "simplify"] 
    clean_content: Optional[str]
    final_output: Optional[str]
    error: Optional[str]

def scraper_node(state: AgentState):
    """Node 1: Scrapes and cleans the website."""
    url = state["url"]
    try:
        # Use a generic user agent to avoid basic blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove junk elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe', 'ads']):
            tag.decompose()
            
        text_content = soup.get_text(separator="\n\n", strip=True)
        
        # --- FIX IS HERE ---
        # Reduced from 25,000 to 15,000 to stay under Groq's 6,000 TPM limit
        # 15,000 chars is roughly 3,750 tokens, which is safe.
        if len(text_content) > 15000:
            text_content = text_content[:15000] + "\n...[Content Truncated]"

        return {"clean_content": text_content}

    except Exception as e:
        return {"error": str(e), "clean_content": None}

def extractor_node(state: AgentState):
    """Smart Clutter Free: Extracts only main article content as Markdown."""
    content = state.get("clean_content")
    if not content: return {"error": "No content"}
    
    llm = ChatGroq(temperature=0, model=LLM_MODEL, api_key=os.getenv("GROQ_API_KEY"))
    
    # Instructions to reconstruct the article cleanly
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert content extractor. Extract only the main article text."),
        ("user", "Convert the following chaotic web text into clean, formatted Markdown. Remove ads, navigation, sidebars, and promotional fluff. Keep the original meaning intact.\n\nTEXT:\n{text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"text": content})
    return {"final_output": response.content}

def simplifier_node(state: AgentState):
    """Smart Reader: Rewrites text to be simpler and highlights key points."""
    content = state.get("clean_content")
    if not content: return {"error": "No content"}

    llm = ChatGroq(temperature=0, model=LLM_MODEL, api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cognitive accessibility assistant."),
        ("user", "Rewrite the following text to be easier to read (Grade 8 level). Use short paragraphs. **Bold** the most important key phrase in every paragraph for emphasis.\n\nTEXT:\n{text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"text": content})
    return {"final_output": response.content}

def summarizer_node(state: AgentState):
    """Node 2: Summarizes the content using Groq."""
    content = state.get("clean_content")
    
    # Fail fast if previous step failed
    if not content or state.get("error"):
        return {"final_output": "Error: Could not retrieve content to summarize."}

    try:
        # Initialize Groq LLM
        # Note: We use 'model' instead of 'model_name' for better compatibility with newer LangChain versions
        llm = ChatGroq(
            temperature=0, 
            model=LLM_MODEL,
            api_key=os.getenv("GROQ_API_KEY") # Explicitly passing key ensures clarity
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes web content."),
            ("user", "Analyze the following text and provide a summary in 3-5 concise bullet points. Focus on the main insights.\n\nTEXT:\n{text}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"text": content})
        
        return {"final_output": response.content}

    except Exception as e:
        return {"final_output": f"LLM Processing Error: {str(e)}"}

def route_next_step(state: AgentState):
    if state.get("error"): return END
    # Routing logic based on task_type
    if state["task_type"] == "smart_extract": return "extractor"
    if state["task_type"] == "simplify": return "simplifier"
    return "summarizer"

# UPDATE GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("scraper", scraper_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("extractor", extractor_node)   # NEW
workflow.add_node("simplifier", simplifier_node) # NEW

workflow.set_entry_point("scraper")

workflow.add_conditional_edges(
    "scraper",
    route_next_step,
    {
        "extractor": "extractor",
        "simplifier": "simplifier",
        "summarizer": "summarizer",
        END: END
    }
)

workflow.add_edge("extractor", END)
workflow.add_edge("simplifier", END)
workflow.add_edge("summarizer", END)

agent_app = workflow.compile()

# --- PART 2: FASTAPI SERVER ---

app = FastAPI(title="NeuroFlow Web Agent")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the Input Model
class RequestBody(BaseModel):
    url: str
    task_type: Literal["summarize", "smart_extract", "simplify"]

@app.post("/process")
async def process_url(request: RequestBody):
    """
    Endpoint to process a URL.
    Payload: { "url": "https://example.com", "task_type": "summarize" }
    """
    try:
        results = agent_app.invoke({
            "url": request.url, 
            "task_type": request.task_type
        })
        
        if results.get("error"):
            # If the scraping failed, we return a 400 error
            raise HTTPException(status_code=400, detail=results["error"])

        response_data = {
            "status": "success",
            "task": request.task_type,
        }

        # Format the output based on task
        if request.task_type == "clutter_free":
            response_data["data"] = results.get("clean_content", "No content found")
        else:
            response_data["data"] = results.get("final_output", "No summary generated")

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port} using model {LLM_MODEL}...")
    uvicorn.run(app, host="0.0.0.0", port=port)