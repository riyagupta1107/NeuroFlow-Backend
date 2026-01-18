from fastapi import APIRouter, HTTPException
from models import *
from chains import correction_chain, task_chain, panic_chain

router = APIRouter()

@router.post("/correct", response_model=CorrectionResponse)
async def correct_text(req: CorrectionRequest):
    return correction_chain.invoke({"text": req.text})

@router.post("/suggest-tasks", response_model=TaskResponse)
async def suggest_tasks(req: TaskRequest):
    return task_chain.invoke({
        "energy_level": req.energy_level,
        "mood": req.current_mood,
        "tasks": ", ".join(req.user_context)
    })

@router.post("/panic-reset", response_model=PanicResponse)
async def panic_reset():
    return panic_chain.invoke({})
