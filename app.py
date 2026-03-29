from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
import uvicorn

from src.helper import download_embeddings
from src.prompt import *

from langchain_pinecone import PineconeVectorStore
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------- INIT --------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

load_dotenv()

# Env vars
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY 
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY 
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------- VECTOR DB --------------------
embedding = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -------------------- LLM --------------------
# chatModel = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.3,
#     google_api_key=GEMINI_API_KEY
# )
chatModel = ChatGroq(
    model="llama-3.3-70b-Versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -------------------- ROUTES --------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", context={"request": request}
)


@app.post("/get")
async def chat(request: Request):
    try:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            payload = await request.json()
            user_message = str((payload or {}).get("msg", "")).strip()
        else:
            form_data = await request.form()
            user_message = str(form_data.get("msg", "")).strip()

        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Message is required."})

        print("User:", user_message)

        response = rag_chain.invoke({"input": user_message})

        answer = response.get("answer", "No response")
        print("Bot:", answer)

        return {"answer": answer}

    except Exception as e:
        return JSONResponse(status_code=500,content={"error": str(e)})


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)