"""Check available Groq models"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    models = client.models.list()
    print("Available Groq Models:")
    print("=" * 60)
    for model in models.data:
        print(f"  • {model.id}")
    print("=" * 60)
except Exception as e:
    print(f"Error: {e}")
