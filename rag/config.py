import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env
load_dotenv()

# Google GenAI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Check your environment configuration.")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "strategies"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
