from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from chains import correction_chain, task_chain, panic_chain
from models import CorrectionRequest, TaskRequest, CorrectionResponse, TaskResponse, PanicResponse

app = FastAPI(title="NeuroFlow AI Engine", version="1.0")

# Allow CORS for your Chrome Extension (important!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Extension ID here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Common Prefix
API_PREFIX = "/api/v1"

@app.get("/")
async def root():
    return {"status": "NeuroFlow AI is Online"}

# --- Endpoint 1: Smart Correction ---
@app.post(f"{API_PREFIX}/correct", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest):
    try:
        # Invoke LangChain
        response = correction_chain.invoke({"text": request.text})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint 2: Energy Task Suggestions ---
@app.post(f"{API_PREFIX}/suggest-tasks", response_model=TaskResponse)
async def suggest_tasks(request: TaskRequest):
    try:
        # Invoke LangChain with dynamic inputs
        response = task_chain.invoke({
            "energy_level": request.energy_level,
            "mood": request.current_mood,
            "tasks": ", ".join(request.user_context)
        })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint 3: Panic Reset ---
@app.post(f"{API_PREFIX}/panic-reset", response_model=PanicResponse)
async def panic_reset():
    try:
        response = panic_chain.invoke({})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)