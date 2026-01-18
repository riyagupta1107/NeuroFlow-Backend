import os
import uvicorn
import requests
from bs4 import BeautifulSoup
from typing import Literal, Optional
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# AI Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Import from your local models.py file
from models import (
    CorrectionRequest, CorrectionResponse,
    TaskRequest, TaskResponse,
    PanicResponse, PanicResponse,
    RequestBody, AgentState
)

load_dotenv()

app = FastAPI(title="NeuroFlow AI Engine", version="1.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PART 1: GEMINI CHAINS (NeuroFlow)
# ==========================================

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Updated to faster/cheaper model
    temperature=0.1,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 1. Correction Chain
correction_parser = JsonOutputParser(pydantic_object=CorrectionResponse)
correction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistive writing AI. Return JSON matching: {format_instructions}"),
    ("user", "{text}")
])
correction_chain = correction_prompt.partial(format_instructions=correction_parser.get_format_instructions()) | gemini_llm | correction_parser

# 2. Task Chain
task_parser = JsonOutputParser(pydantic_object=TaskResponse)
task_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Executive Function Coach. Output STRICT JSON with keys: zone, advice, suggested_tasks. Schema: {format_instructions}"),
    ("user", "Energy Level: {energy_level}. Tasks: {tasks}")
])
task_chain = task_prompt.partial(format_instructions=task_parser.get_format_instructions()) | gemini_llm | task_parser

# 3. Panic Chain
panic_parser = JsonOutputParser(pydantic_object=PanicResponse)
panic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calming anxiety-relief assistant. Return valid JSON: {format_instructions}"),
    ("user", "I am overwhelmed.")
])
panic_chain = panic_prompt.partial(format_instructions=panic_parser.get_format_instructions()) | gemini_llm | panic_parser

# ==========================================
# PART 2: GROQ AGENT (Web Scraper)
# ==========================================

LLM_MODEL = "llama-3.1-8b-instant"

def scraper_node(state: AgentState):
    url = state["url"]
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'ads']):
            tag.decompose()
        text = soup.get_text(separator="\n\n", strip=True)
        return {"clean_content": text[:15000]}
    except Exception as e:
        return {"error": str(e)}

def summarizer_node(state: AgentState):
    content = state.get("clean_content")
    if not content: return {"error": "No content"}
    llm = ChatGroq(temperature=0, model=LLM_MODEL, api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this text in 3-5 bullet points."),
        ("user", "{text}")
    ])
    res = (prompt | llm).invoke({"text": content})
    return {"final_output": res.content}

# ... (Add extractor_node and simplifier_node here similarly if needed) ...

# Simple Graph Setup for the example
workflow = StateGraph(AgentState)
workflow.add_node("scraper", scraper_node)
workflow.add_node("summarizer", summarizer_node)
workflow.set_entry_point("scraper")
workflow.add_edge("scraper", "summarizer")
workflow.add_edge("summarizer", END)
agent_app = workflow.compile()

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def root():
    return {"status": "NeuroFlow AI is Online"}

@app.post("/api/v1/correct", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest):
    return correction_chain.invoke({"text": request.text})

@app.post("/api/v1/suggest-tasks", response_model=TaskResponse)
async def suggest_tasks(request: TaskRequest):
    return task_chain.invoke({
        "energy_level": request.energy_level,
        "mood": request.current_mood,
        "tasks": ", ".join(request.user_context)
    })

@app.post("/api/v1/panic-reset", response_model=PanicResponse)
async def panic_reset():
    return panic_chain.invoke({})

@app.post("/process")
async def process_url(request: RequestBody):
    # This invokes the LangGraph agent
    result = agent_app.invoke({"url": request.url, "task_type": request.task_type})
    return {"status": "success", "data": result.get("final_output")}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)