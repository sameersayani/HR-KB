# ============================================================
# PHASE 3 - STEP 3: Document Manager
# ============================================================
# What you'll learn:
#   - A clean class to manage all document CRUD operations
#   - Add: ingest new docs safely (no duplicates)
#   - List: see what's stored per country
#   - Delete: remove a document and all its chunks
#   - Stats: storage summary across all countries
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")
SUPPORTED_FORMATS = {".txt", ".pdf", ".docx"}
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 150

# ============================================================
# THE DOCUMENT MANAGER CLASS
# ============================================================

class DocumentManager:
    """
    Manages the HR knowledge base — adding, listing, and
    deleting documents per country in the Chroma vector store.
    """

    def __init__(self, vectorstore: Chroma):
        self.vs       = vectorstore
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    # ----------------------------------------------------------
    # PRIVATE: get stored metadata
    # ----------------------------------------------------------
    def _get_all_records(self) -> list[dict]:
        """Fetch all records with ids + metadata from Chroma."""
        data = self.vs._collection.get(
            include=["metadatas"]
        )
        return [
            {"id": id_, "metadata": meta}
            for id_, meta in zip(data["ids"], data["metadatas"])
        ]

    # ----------------------------------------------------------
    # LIST documents
    # ----------------------------------------------------------
    def list_documents(self, country: str = None) -> list[dict]:
        """
        List unique documents in the store.
        Optionally filter by country.
        Returns: list of {filename, country, chunk_count}
        """
        records = self._get_all_records()

        # Group chunks by filename
        doc_map = defaultdict(lambda: {"country": "", "chunk_count": 0})
        for rec in records:
            fname   = rec["metadata"].get("filename", "unknown")
            ctry    = rec["metadata"].get("country",  "unknown")
            if country and ctry != country.lower():
                continue
            doc_map[fname]["country"]     = ctry
            doc_map[fname]["chunk_count"] += 1

        return [
            {"filename": fname, **info}
            for fname, info in sorted(doc_map.items())
        ]

    # ----------------------------------------------------------
    # ADD a document
    # ----------------------------------------------------------
    def add_document(self, file_path: str, country: str) -> dict:
        """
        Add a document to the knowledge base.
        Skips if already stored (no duplicates).
        Returns a status dict.
        """
        path     = Path(file_path)
        filename = path.name
        ext      = path.suffix.lower()

        # Check format
        if ext not in SUPPORTED_FORMATS:
            return {
                "status"  : "error",
                "filename": filename,
                "message" : f"Unsupported format: {ext}. "
                            f"Supported: {SUPPORTED_FORMATS}",
            }

        # Check for duplicate
        existing = {d["filename"] for d in self.list_documents()}
        if filename in existing:
            return {
                "status"  : "skipped",
                "filename": filename,
                "message" : "Already exists in knowledge base.",
            }

        # Load the document
        try:
            if ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)

            docs = loader.load()
        except Exception as e:
            return {"status": "error", "filename": filename,
                    "message": str(e)}

        # Enrich metadata
        for doc in docs:
            doc.metadata.update({
                "country"    : country.lower(),
                "filename"   : filename,
                "doc_type"   : "hr_policy",
                "uploaded_at": datetime.now().isoformat(),
            })

        # Chunk and store
        chunks = self.splitter.split_documents(docs)
        self.vs.add_documents(chunks)

        return {
            "status"      : "added",
            "filename"    : filename,
            "country"     : country,
            "chunks_added": len(chunks),
            "message"     : f"Successfully added {len(chunks)} chunks.",
        }

    # ----------------------------------------------------------
    # DELETE a document
    # ----------------------------------------------------------
    def delete_document(self, filename: str) -> dict:
        """
        Delete all chunks belonging to a given filename.
        Returns a status dict.
        """
        records = self._get_all_records()

        # Find IDs belonging to this file
        ids_to_delete = [
            rec["id"]
            for rec in records
            if rec["metadata"].get("filename") == filename
        ]

        if not ids_to_delete:
            return {
                "status"  : "not_found",
                "filename": filename,
                "message" : "No chunks found for this file.",
            }

        self.vs._collection.delete(ids=ids_to_delete)

        return {
            "status"         : "deleted",
            "filename"       : filename,
            "chunks_deleted" : len(ids_to_delete),
            "message"        : f"Deleted {len(ids_to_delete)} chunks.",
        }

    # ----------------------------------------------------------
    # STATS
    # ----------------------------------------------------------
    def get_stats(self) -> dict:
        """Return a summary of the knowledge base."""
        records  = self._get_all_records()
        total    = len(records)

        country_chunks = defaultdict(int)
        country_files  = defaultdict(set)
        for rec in records:
            ctry  = rec["metadata"].get("country",  "unknown")
            fname = rec["metadata"].get("filename", "unknown")
            country_chunks[ctry] += 1
            country_files[ctry].add(fname)

        return {
            "total_chunks"    : total,
            "total_countries" : len(country_chunks),
            "by_country"      : {
                ctry: {
                    "chunks"   : country_chunks[ctry],
                    "files"    : sorted(country_files[ctry]),
                    "num_files": len(country_files[ctry]),
                }
                for ctry in sorted(country_chunks)
            },
        }

# ============================================================
# DEMO / TEST
# ============================================================
if __name__ == "__main__":
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory  = VECTOR_STORE_PATH,
        embedding_function = embeddings,
        collection_name    = "hr_policies",
    )
    dm = DocumentManager(vectorstore)

    # 1. Stats
    print("\n" + "="*55)
    print("KNOWLEDGE BASE STATS")
    print("="*55)
    stats = dm.get_stats()
    print(f"Total chunks    : {stats['total_chunks']}")
    print(f"Total countries : {stats['total_countries']}")
    for country, info in stats["by_country"].items():
        print(f"\n  {country.upper()}")
        print(f"    Files  : {info['num_files']}")
        print(f"    Chunks : {info['chunks']}")
        for f in info["files"]:
            print(f"    • {f}")

    # 2. List all docs
    print("\n" + "="*55)
    print("ALL DOCUMENTS")
    print("="*55)
    docs = dm.list_documents()
    print(f"{'Filename':<35} {'Country':<8} {'Chunks':>6}")
    print("-"*52)
    for d in docs:
        print(f"{d['filename']:<35} {d['country']:<8} {d['chunk_count']:>6}")

    # 3. Try adding duplicate (should be skipped)
    print("\n" + "="*55)
    print("DUPLICATE INGESTION TEST")
    print("="*55)
    result = dm.add_document(
        "data/uploads/india/hr_policy_india.txt", "india"
    )
    print(f"Status : {result['status'].upper()}")
    print(f"Message: {result['message']}")

    # 4. List India-only docs
    print("\n" + "="*55)
    print("INDIA DOCUMENTS ONLY")
    print("="*55)
    india_docs = dm.list_documents(country="india")
    for d in india_docs:
        print(f"  • {d['filename']} ({d['chunk_count']} chunks)")
