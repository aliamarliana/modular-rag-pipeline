# 🧠 Modular RAG Pipeline

A **plug-and-play Retrieval-Augmented Generation (RAG) pipeline** — built with modular design so companies and developers can easily load data, generate embeddings, store vectors, and query using LLMs.

---

## ⚙️ Overview

This project provides a **common RAG pipeline** that can be reused or adapted across different company use cases.  
You can modify the data source, embedding model, vector store, and LLM integration — all through modular files.

---

## 🪜 Setup Guide

### 1️⃣ Create and Activate Virtual Environment

It’s best to isolate your dependencies using a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2️⃣ Install Dependencies

Once your environment is activated, install all required libraries:

```bash
pip install -r requirements.txt
```

### 3️⃣ Create a Data Folder

Before running the app, you need to create a data/ folder in your project root.
This is where you’ll place all your company or project documents.

```bash
mkdir data
```

Then, copy your files (PDF, TXT, CSV, DOCX, JSON, Excel, etc.) into that folder:

```kotlin
data/
├── company_policy.pdf
├── sales_data.xlsx
├── notes.txt
└── reports/
    └── 2024_summary.docx
```

⚠️ The pipeline will automatically read all supported files inside this folder.

### 5️⃣ Run the Pipeline

Once dependencies are installed and your documents are inside the data/ folder, run:
```bash
python app.py
```

---
## 🧩 Modular Design

The project is fully modular — each file handles one logical part of the RAG pipeline.

| Module | Description |
|---------|-------------|
| `data_loader.py` | Loads data from PDF, TXT, CSV, DOCX, JSON, and Excel files. |
| `embedding.py` | Splits documents into chunks and generates embeddings using SentenceTransformers. |
| `vector_store.py` | Builds and stores vector embeddings using FAISS. |
| `search.py` | Retrieves top relevant documents and summarizes results using Groq LLM. |
| `app.py` | Orchestrates everything — can be customized for company-specific workflows. |

---

## 🎥 Code Reference
This project was **inspired by the tutorial by Krish Naik**:  
🔗 [Complete RAG Crash Course With Langchain In 2 Hours](https://www.youtube.com/watch?v=o126p1QN_RI&t=3971s)
