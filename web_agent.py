from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from agent import agent_app   # your LangGraph logic

router = APIRouter()

class RequestBody(BaseModel):
    url: str
    task_type: Literal["summarize", "smart_extract", "simplify"]

@router.post("/process")
async def process_url(request: RequestBody):
    results = agent_app.invoke({
        "url": request.url,
        "task_type": request.task_type
    })

    if results.get("error"):
        raise HTTPException(status_code=400, detail=results["error"])

    return {
        "status": "success",
        "task": request.task_type,
        "data": results.get("final_output")
    }
