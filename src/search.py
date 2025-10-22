"""
📘 RAG Search & Summarization Module
-----------------------------------
This module performs:
1️⃣ Retrieval: Finds relevant document chunks from FAISS vector store.
2️⃣ Generation: Summarizes or answers queries using an LLM (Groq, OpenAI, etc.).

✅ Purpose:
- Combines stored knowledge (retrieval) with reasoning (generation).
- Acts as the "Q&A brain" of your modular RAG system.

⚙️ Customizable sections are clearly marked with:
    # 🔧 COMPANY-SPECIFIC CONFIG
"""

import os
from dotenv import load_dotenv
from src.vector_store import FaissVectorStore
from langchain_groq import ChatGroq  # LLM from Groq (can replace with OpenAI, Anthropic, etc.)

load_dotenv()


class RAGSearch:
    """
    RAGSearch orchestrates the retrieval (FAISS) and generation (LLM) process.
    """

    def __init__(
        self,
        persist_dir: str = "faiss_store",              # 🔧 Path to saved FAISS index
        embedding_model: str = "all-MiniLM-L6-v2",     # 🔧 Embedding model (must match the one used in vector_store)
        llm_model: str = "gemma2-9b-it"                # 🔧 LLM for summarization or answering
    ):
        # Initialize vector store
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build FAISS index if not found
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents  # Import lazily to avoid circular dependencies
            print("[INFO] FAISS index not found — building new one...")
            docs = load_all_documents("data")               # 🔧 Change data path as needed
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # Initialize Groq LLM (uses API key from environment)
        groq_api_key = os.getenv("GROQ_API_KEY", "")        # 🔧 Set this in your .env file
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    # -------------------------------------------------------------------------
    # 1️⃣ SEARCH & SUMMARIZE
    # -------------------------------------------------------------------------
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve top_k relevant chunks from FAISS and summarize using the LLM.
        """
        print(f"[INFO] Searching and summarizing for query: '{query}'")

        # Step 1: Retrieve relevant documents
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        # Step 2: Build prompt for LLM summarization or answer generation
        prompt = f"""
        You are an expert assistant. Based on the following context, answer or summarize for the query: 
        '{query}'

        Context:
        {context}

        Summary:
        """.strip()

        # Step 3: Generate response
        response = self.llm.invoke([prompt])
        return response.content


# -------------------------------------------------------------------------
# Example usage — for testing standalone RAG search pipeline
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#    rag_search = RAGSearch()

#    query = "What is attention mechanism?"
#    summary = rag_search.search_and_summarize(query, top_k=3)

#    print("\n🧠 QUERY:", query)
#    print("💬 SUMMARY RESULT:\n", summary)
