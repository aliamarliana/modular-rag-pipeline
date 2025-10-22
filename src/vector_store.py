"""
📘 RAG Common Vector Store Utility (FAISS)
------------------------------------------
This module manages storing, saving, loading, and searching embeddings using FAISS.

✅ Purpose:
- Build and persist a FAISS index from document embeddings.
- Enable fast similarity search for retrieval during RAG queries.
- Modular and easily replaceable (e.g., switch FAISS to Chroma or Pinecone later).

⚙️ Customizable sections are clearly marked with:
    # 🔧 COMPANY-SPECIFIC CONFIG
"""

import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    """
    Handles all vector storage and retrieval operations using FAISS.
    """

    def __init__(
        self,
        persist_dir: str = "faiss_store",           # 🔧 Where FAISS files are stored
        embedding_model: str = "all-MiniLM-L6-v2",  # 🔧 Default embedding model
        chunk_size: int = 1000,                     # 🔧 Same chunk size as embedding stage
        chunk_overlap: int = 200                    # 🔧 Overlap between chunks
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Initialized FAISS store with model: {embedding_model}")

    # -------------------------------------------------------------------------
    # 1️⃣ BUILD VECTOR STORE FROM DOCUMENTS
    # -------------------------------------------------------------------------
    def build_from_documents(self, documents: List[Any]):
        """
        Converts raw documents → chunks → embeddings → FAISS index.
        """
        print(f"[INFO] Building FAISS vector store from {len(documents)} documents...")

        # Step 1: Create embeddings
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        # Step 2: Add text metadata for traceability
        metadatas = [{"text": chunk.page_content} for chunk in chunks]

        # Step 3: Store embeddings in FAISS
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] ✅ FAISS vector store built and saved at '{self.persist_dir}'")

    # -------------------------------------------------------------------------
    # 2️⃣ ADD EMBEDDINGS TO FAISS
    # -------------------------------------------------------------------------
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """
        Adds embeddings and metadata to FAISS index.
        """
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)  # 🔧 Change distance metric if needed
        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index.")

    # -------------------------------------------------------------------------
    # 3️⃣ SAVE & LOAD
    # -------------------------------------------------------------------------
    def save(self):
        """
        Saves FAISS index and metadata to disk.
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index and metadata to '{self.persist_dir}'")

    def load(self):
        """
        Loads FAISS index and metadata from disk.
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded FAISS index and metadata from '{self.persist_dir}'")

    # -------------------------------------------------------------------------
    # 4️⃣ SEARCH / QUERY
    # -------------------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search for the top_k most similar embeddings.
        """
        D, I = self.index.search(query_embedding, top_k)
        results = []

        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({
                "index": idx,
                "distance": dist,
                "metadata": meta
            })
        return results

    def query(self, query_text: str, top_k: int = 5):
        """
        Converts a text query into embeddings and performs FAISS similarity search.
        """
        print(f"[INFO] Querying FAISS store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# -------------------------------------------------------------------------
# Example usage — remove or modify when integrating into your main RAG pipeline
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#     from src.data_loader import load_all_documents

#     # Step 1: Load data
#     docs = load_all_documents("data")

#     # Step 2: Initialize FAISS store
#     store = FaissVectorStore("faiss_store")

#     # Step 3: Build FAISS index
#     store.build_from_documents(docs)

#     # Step 4: Load and test query
#     store.load()
#     results = store.query("What is attention mechanism?", top_k=3)

#     # Step 5: Display results
#     for res in results:
#         print(f"→ Score: {res['distance']:.4f}\n{res['metadata']['text'][:200]}...\n")