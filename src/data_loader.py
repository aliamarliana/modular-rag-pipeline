"""
📘 RAG Common Loader Utility
----------------------------
This script loads documents of various formats (PDF, TXT, CSV, Excel, Word, JSON)
into LangChain Document objects.

✅ Purpose:
- Acts as a universal document ingestion pipeline.
- Easy to plug into a RAG system (e.g., for indexing or vectorization).
- Modular design — you can add/remove supported file types as needed.

⚙️ Customizable sections are clearly marked with:
    # 🔧 COMPANY-SPECIFIC CONFIG
"""

from pathlib import Path
from typing import List, Any

# LangChain community document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Loads all supported files from the given directory and its subfolders.

    Supported file types:
        - PDF (.pdf)
        - Text (.txt)
        - CSV (.csv)
        - Excel (.xlsx)
        - Word (.docx)
        - JSON (.json)
    """
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Loading documents from: {data_path}")

    documents = []

    # 🔧 COMPANY-SPECIFIC CONFIG
    # If you want to add/remove file types, update this mapping.
    loaders = {
        "*.pdf": PyPDFLoader,
        "*.txt": TextLoader,
        "*.csv": CSVLoader,
        "*.xlsx": UnstructuredExcelLoader,
        "*.docx": Docx2txtLoader,
        "*.json": JSONLoader,
    }

    # Iterate over all supported file types
    for pattern, LoaderClass in loaders.items():
        files = list(data_path.glob(f"**/{pattern}"))
        print(f"[DEBUG] Found {len(files)} files for pattern '{pattern}'")

        for file_path in files:
            try:
                print(f"[INFO] Loading: {file_path}")
                loader = LoaderClass(str(file_path))
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"[SUCCESS] Loaded {len(loaded_docs)} docs from {file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")

    print(f"[INFO] ✅ Total documents loaded: {len(documents)}")
    return documents