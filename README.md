# Medical Chatbot (Medical_BOT)

A sophisticated medical chatbot application powered by LLMs and vector embeddings. This application uses LangChain, Pinecone vector database, and Groq LLM to answer medical questions based on a knowledge base of PDF documents.

## 🎯 Overview

Medical_BOT is a Retrieval Augmented Generation (RAG) application that:
- Processes medical documents (PDFs) and stores them in a vector database
- Uses semantic search to retrieve relevant medical information
- Generates accurate, concise medical answers using advanced LLMs
- Provides an interactive web interface for users to ask medical questions
- Delivers responses limited to 3 sentences to ensure clarity and conciseness

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Web UI)                       │
│              (HTML/CSS/JavaScript - Bootstrap)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─ /get (POST) - Chat endpoint
                       └─ / (GET) - Main interface
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  FastAPI Backend (app.py)                    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RAG Chain (Retrieval Augmented Generation)         │   │
│  │  ┌──────────────┐    ┌──────────────┐              │   │
│  │  │  Retriever   │    │  LLM Chain   │              │   │
│  │  │  (Top 3      │───▶│  (Groq       │              │   │
│  │  │   similar    │    │   llama-3.3) │              │   │
│  │  │  chunks)     │    │              │              │   │
│  │  └──────────────┘    └──────────────┘              │   │
│  └─────────────────────────────────────────────────────┘   │
│                       │                                       │
└───────────────┬───────▼───────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│              Pinecone Vector Database                         │
│              (Index: medical-chatbot)                         │
│         Stores embeddings of medical documents                │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│         Initial Data Pipeline (store_index.py)               │
│                                                               │
│  PDF Files → Extraction → Text Splitting → Embeddings       │
│  (data/) → (PyPDF) → (Chunks: 500 chars) → (Sentence-Trans.)│
└────────────────────────────────────────────────────────────┘
```

## 🚀 Features

- **RAG-Based Question Answering**: Retrieves relevant medical documents and generates contextual answers
- **Vector Embeddings**: Uses sentence-transformers for document embeddings (all-MiniLM-L6-v2 model)
- **Advanced LLM**: Powered by Groq's llama-3.3-70b-Versatile for high-quality responses
- **Vector Database**: Pinecone for efficient similarity search over medical documents
- **Responsive Web UI**: Clean, intuitive chat interface built with Bootstrap
- **REST API**: FastAPI backend with JSON endpoints
- **Health Check**: Built-in health endpoint for monitoring
- **Error Handling**: Robust error handling for API requests
- **Modular Code Structure**: Organized helper functions and prompts for maintainability

## 📦 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | HTML5, CSS, JavaScript (jQuery), Bootstrap 4 |
| **Backend Framework** | FastAPI |
| **Server** | Uvicorn |
| **LLM** | Groq (llama-3.3-70b-Versatile) |
| **Vector Database** | Pinecone |
| **Embeddings** | Sentence Transformers (HuggingFace) |
| **LLM Framework** | LangChain |
| **Document Loading** | PyPDF |
| **Templating** | Jinja2 |

## 📁 Project Structure

```
Medical_BOT/
├── app.py                 # Main FastAPI application
├── store_index.py        # Data pipeline: PDF processing → Vector DB
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup configuration
├── template.sh           # Shell template (utility)
├── README.md             # This file
│
├── data/                 # Medical PDF documents directory
│   └── *.pdf            # Source medical documents
│
├── src/                  # Source code modules
│   ├── __init__.py       # Package initialization
│   ├── helper.py         # Helper functions for PDF loading, text splitting, embeddings
│   └── prompt.py         # System prompts and prompt templates
│
├── templates/            # HTML templates
│   └── index.html        # Main chat interface UI
│
└── static/               # Static files
    └── style.css         # Custom styling for chat interface
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository
```bash

git clone https://github.com/cheerlaminusri/Medical_BOT.git


```

### Step 2: Create Virtual Environment (Recommended)
```bash

python -m venv venv
venv\Scripts\activate

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root directory with the following variables:

```env
# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key_here

# Groq LLM API
GROQ_API_KEY=your_groq_api_key_here
```

### Get API Keys

1. **Pinecone API Key**:
   - Go to [Pinecone.io](https://www.pinecone.io/)
   - Sign up for a free account
   - Create a project and get your API key
   - Create a Serverless index named `medical-chatbot`

2. **Groq API Key**:
   - Visit [Groq Console](https://console.groq.com/)
   - Sign up and create an API key
   - No usage limits on free tier for llama models

## 📚 Usage

### Step 1: Prepare Medical Documents
Place your medical PDF documents in the `data/` directory:
```bash
cp your_medical_documents.pdf data/
```

### Step 2: Build the Vector Database Index
Run the indexing script to process PDFs and store embeddings in Pinecone:
```bash
python store_index.py
```

**What this does:**
- Loads all PDF files from `data/` directory
- Extracts text from PDFs
- Splits text into chunks (500 characters with 20-character overlap)
- Generates embeddings using Sentence Transformers
- Stores embeddings in Pinecone vector database at index `medical-chatbot`

### Step 3: Start the Application

**Option 1: Simple Start**
```bash
python app.py
```

**Option 2: Development Mode (with auto-reload)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

The server will start at `http://localhost:8080`

*Note: Use `--reload` flag during development for automatic server restart when code changes are detected.*

### Step 4: Access the Web Interface
Open your browser and navigate to:
```
http://localhost:8080
```

Ask your medical questions in the chat interface and receive AI-powered medical information.

## 🔌 API Endpoints

### GET `/`
- **Description**: Serves the main chat interface
- **Response**: HTML page with chat UI

### POST `/get`
- **Description**: Processes user medical questions
- **Content-Type**: `application/json` or `multipart/form-data`
- **Request Body**:
  ```json
  {
    "msg": "What are the symptoms of diabetes?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Diabetes symptoms include increased thirst, frequent urination, and fatigue. You should consult a healthcare provider if you experience these signs. Early detection and management can help prevent complications."
  }
  ```
- **Error Response**:
  ```json
  {
    "error": "Error message describing what went wrong"
  }
  ```

### GET `/health`
- **Description**: Health check endpoint
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

## 🔄 How It Works

### Data Processing Pipeline (`store_index.py`)
1. **PDF Loading**: Uses PyPDF to extract text from medical documents
2. **Filtering**: Keeps only essential metadata (source file path)
3. **Text Chunking**: Splits documents into 500-character chunks with 20-character overlap
4. **Embeddings Generation**: Converts text chunks to vector embeddings using `all-MiniLM-L6-v2`
5. **Vector Storage**: Stores embeddings in Pinecone for fast similarity search

### Query Processing (`app.py`)
1. **User Input**: User submits a medical question via the web interface
2. **Embedding**: The query is converted to an embedding using the same model
3. **Retrieval**: Pinecone retrieves the top 3 most similar document chunks
4. **Prompt Construction**: Selected chunks are added to the system prompt as context
5. **LLM Generation**: Groq's llama-3.3-70b generates a concise answer (max 3 sentences)
6. **Response**: Answer is returned to the user and displayed in the chat
