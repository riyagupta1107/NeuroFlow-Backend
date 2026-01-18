import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models import CorrectionResponse, TaskResponse, PanicResponse

load_dotenv()

# Initialize Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.1,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------------------------------------------------------
# Feature 1: Phonetic Auto-Complete
# ---------------------------------------------------------
correction_parser = JsonOutputParser(pydantic_object=CorrectionResponse)

correction_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an assistive writing AI. 
    Analyze the text for phonetic spelling errors.
    If the text is cut off, predict the completion.
    
    Return valid JSON matching this schema:
    {format_instructions}
    
    Example Input: "nefasary"
    Example Output: {{ "original": "nefasary", "suggestion": "necessary", "confidence": 0.9 }}
    """),
    ("user", "{text}")
])

correction_chain = correction_prompt.partial(format_instructions=correction_parser.get_format_instructions()) | llm | correction_parser

# ---------------------------------------------------------
# Feature 2: Energy-Based Task Planner (FIXED PROMPT)
# ---------------------------------------------------------
task_parser = JsonOutputParser(pydantic_object=TaskResponse)

task_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an Executive Function Coach.
    Based on the energy level (1-10), select the best tasks from the list provided.
    
    RULES:
    1. Low Energy (1-4): Pick simple, low-friction tasks.
    2. High Energy (7-10): Pick complex, deep-work tasks.
    3. Output STRICT JSON with exactly these keys: "zone", "advice", "suggested_tasks".
    4. "suggested_tasks" must be a flat list of strings, NOT a dictionary.
    
    Schema instructions:
    {format_instructions}
    """),
    ("user", "Energy Level: {energy_level}. Tasks: {tasks}")
])

task_chain = task_prompt.partial(format_instructions=task_parser.get_format_instructions()) | llm | task_parser

# ---------------------------------------------------------
# Feature 3: Panic Reset
# ---------------------------------------------------------
panic_parser = JsonOutputParser(pydantic_object=PanicResponse)

panic_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a calming anxiety-relief assistant.
    Generate a soothing message and ONE tiny micro-step.
    
    Return valid JSON:
    {format_instructions}
    """),
    ("user", "I am overwhelmed.")
])

panic_chain = panic_prompt.partial(format_instructions=panic_parser.get_format_instructions()) | llm | panic_parser