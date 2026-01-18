from pydantic import BaseModel, Field
from typing import List, Optional

# --- Phonetic Correction Models ---
class CorrectionRequest(BaseModel):
    text: str = Field(..., description="The user's current raw text input")

class CorrectionResponse(BaseModel):
    original: str
    suggestion: str
    confidence: float

# --- Mood Task Models ---
class TaskRequest(BaseModel):
    energy_level: int = Field(..., ge=1, le=10)
    current_mood: Optional[str] = "neutral"
    # We pass a simple list of strings as context
    user_context: Optional[List[str]] = ["Study History", "Write Math Paper", "Email Professor", "Clean Desk"]

class TaskResponse(BaseModel):
    zone: str = Field(description="The energy zone name (e.g., 'High Energy')")
    advice: str = Field(description="A short encouraging message")
    suggested_tasks: List[str] = Field(description="A simple list of task names appropriate for the energy level")

# --- Panic Mode Models ---
class PanicResponse(BaseModel):
    calming_message: str
    micro_step: str