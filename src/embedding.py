"""
📘 RAG Common Embedding Utility
-------------------------------
This module handles:
1️⃣ Splitting documents into smaller chunks.
2️⃣ Creating numerical embeddings using a SentenceTransformer model.

✅ Purpose:
- Converts raw text data into embeddings for vector databases (e.g., FAISS, Pinecone, Chroma).
- Modular: easy to adjust chunk size, overlap, or model depending on company use case.

⚙️ Customizable sections are clearly marked with:
    # 🔧 COMPANY-SPECIFIC CONFIG
"""

from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# Import from your data loading module
from src.data_loader import load_all_documents


class EmbeddingPipeline:
    """
    Embedding pipeline for splitting documents and generating embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # 🔧 Default embedding model
        chunk_size: int = 1000,                # 🔧 Adjust chunk size based on document length or context window
        chunk_overlap: int = 200               # 🔧 Overlap between chunks for context continuity
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load the sentence-transformer model
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split long documents into smaller overlapping chunks.
        This ensures embeddings capture context better and fit model limits.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Convert text chunks into numerical embeddings (vectors).
        These embeddings can be stored in a vector database for retrieval.
        """
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        # Generate embeddings using the SentenceTransformer model
        embeddings = self.model.encode(texts, show_progress_bar=True)

        print(f"[INFO] Embeddings generated with shape: {embeddings.shape}")
        return embeddings
    

# Example usage — you can remove this part when integrating into your main pipeline
# if __name__ == "__main__":
#     # 1️⃣ Load documents
#     docs = load_all_documents("data")

#     # 2️⃣ Initialize embedding pipeline
#     emb_pipe = EmbeddingPipeline()

#     # 3️⃣ Split into chunks
#     chunks = emb_pipe.chunk_documents(docs)

#     # 4️⃣ Generate embeddings
#     embeddings = emb_pipe.embed_chunks(chunks)

#     # 5️⃣ Display example result
#     if len(embeddings) > 0:
#         print("[INFO] Example embedding vector:", embeddings[0])