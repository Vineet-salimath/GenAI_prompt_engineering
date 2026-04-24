import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

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
# FALLBACK MODEL
# -------------------------
print("Loading fallback model...")

fallback_llm = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    max_new_tokens=150,
    do_sample=False
)

# -------------------------
# LOAD PDF
# -------------------------
print("Loading PDF...")

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
print("Creating embeddings...")

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
# HUMAN ESCALATION LOGIC
# -------------------------
def needs_human(query):
    query = query.lower()

    if any(word in query for word in ["fraud", "unauthorized", "stolen", "hack"]):
        return True

    if "card" in query and any(word in query for word in ["lost", "block"]):
        return True

    return False


def log_human_request(query):
    with open("human_requests.txt", "a") as f:
        f.write(query + "\n")

# -------------------------
# MAIN QUERY FUNCTION
# -------------------------
def ask_question(query):

    if needs_human(query):
        log_human_request(query)
        return "This query has been escalated to a human agent. Please wait for assistance."

    docs = retriever.invoke(query)

    if not docs:
        return "I don't know."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a banking customer support assistant.

Answer in 2-3 lines.
Do not generate multiple questions.
Do not repeat context.

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
        print("Groq failed, using fallback:", str(e))

        result = fallback_llm(prompt)[0]["generated_text"]
        return result.replace(prompt, "").strip()


# -------------------------
# RUN TEST LOOP
# -------------------------
if __name__ == "__main__":
    print("RAG system ready")

    while True:
        q = input("You: ")

        if q.lower() == "exit":
            break

        print("Bot:", ask_question(q))
