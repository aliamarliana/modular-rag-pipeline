from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch
import os

def main():
    """
    Main entry point for running the modular RAG pipeline.
    Steps:
    1. Load or build the FAISS vector store.
    2. Initialize RAG search using Groq LLM.
    3. Perform a query and summarize retrieved context.

    ⚙️ Customizable sections are clearly marked with:
        # 🔧 COMPANY-SPECIFIC CONFIG
    """

    # 🔧 COMPANY-SPECIFIC CONFIG
    # These are the only lines users typically change to fit their own setup.
    persist_dir = "faiss_store"          # Folder to store FAISS index & metadata
    data_dir = "data"                    # Folder containing your company's documents
    query = "What is attention mechanism?"  # Example query (replace with any question)
    # ================================================================

    print("\n[INFO] Starting RAG pipeline...")

    # Step 1: Initialize Vector Store (handles embeddings + FAISS)
    store = FaissVectorStore(persist_dir)

    # Step 2: Build or Load FAISS index
    faiss_index_path = f"{persist_dir}/faiss.index"
    meta_path = f"{persist_dir}/metadata.pkl"

    if not (os.path.exists(faiss_index_path) and os.path.exists(meta_path)):
        print("[INFO] Building new FAISS index...")
        docs = load_all_documents(data_dir)
        store.build_from_documents(docs)
    else:
        print("[INFO] Loading existing FAISS index...")
        store.load()

    # Step 3: Initialize RAG Search (connects FAISS + LLM)
    rag_search = RAGSearch(persist_dir=persist_dir)

    # Step 4: Query the system
    print(f"\n[INFO] Running query: {query}")
    summary = rag_search.search_and_summarize(query, top_k=3)

    # Step 5: Display Result
    print("\n[RESULT] Summary:\n", summary)


if __name__ == "__main__":
    main()