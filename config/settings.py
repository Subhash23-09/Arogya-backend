import os
from dotenv import load_dotenv

load_dotenv()

#API_KEY = os.getenv("API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")


if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment")
