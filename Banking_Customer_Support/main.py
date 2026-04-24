import os
from dotenv import load_dotenv

# LangChain (UPDATED IMPORTS)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLMs
from groq import Groq
from transformers import pipeline

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------
# INIT GROQ CLIENT
# -------------------------
groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# LOAD FALLBACK MODEL
# -------------------------
print("🔄 Loading fallback model...")

fallback_llm = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    max_new_tokens=150,
    do_sample=False
)

# -------------------------
# LOAD PDF
# -------------------------
print("📄 Loading PDF...")
loader = PyPDFLoader("banking_customer_support_qa.pdf")
documents = loader.load()

# -------------------------
# SPLIT TEXT
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# -------------------------
# EMBEDDINGS
# -------------------------
print("🧠 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# VECTOR STORE
# -------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 🔥 SMART HITL LOGIC (UPDATED)
# -------------------------
def needs_human(query):
    query = query.lower()

    # Strong triggers
    if any(word in query for word in ["fraud", "unauthorized", "stolen", "hack"]):
        return True

    # Combination triggers
    if "card" in query and any(word in query for word in ["lost", "block"]):
        return True

    return False


def log_human_request(query):
    with open("human_requests.txt", "a") as f:
        f.write(query + "\n")

# -------------------------
# QUERY FUNCTION
# -------------------------
def ask_question(query):

    # 🔴 HITL CHECK
    if needs_human(query):
        log_human_request(query)
        return "🚨 This query has been escalated to a human agent. Please wait for assistance."

    docs = retriever.invoke(query)

    if not docs:
        return "I don't know."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a banking customer support assistant.

Answer the question clearly in 2-3 lines.
DO NOT generate multiple questions.
DO NOT repeat the context.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("⚠️ Groq failed, using fallback:", str(e))
        result = fallback_llm(prompt)[0]["generated_text"]
        return result.replace(prompt, "").strip()


# -------------------------
# TEST
# -------------------------
if __name__ == "__main__":
    print("\n✅ RAG Ready (Groq + HITL)")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        print("Bot:", ask_question(q))