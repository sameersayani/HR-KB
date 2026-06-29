# ============================================================
# PHASE 3 - STEP 1: Multi-Tenant Store Management
# ============================================================
# What you'll learn:
#   - What "multi-tenant" means for our HR system
#   - Two isolation strategies: metadata filter vs collection per country
#   - Why we use metadata filtering (simpler, scalable)
#   - How to safely add NEW documents to an existing store
#   - How to list what's already stored per country
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from collections import Counter

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")

# -----------------------------------------------------------
# WHAT IS MULTI-TENANCY?
# -----------------------------------------------------------
# Multi-tenant = one system serving multiple isolated groups.
#
# In our HR system:
#   - India employees should ONLY get India policy answers
#   - USA employees should ONLY get USA policy answers
#   - An India question must NEVER return a UK document
#
# STRATEGY A: Separate Chroma collection per country
#   ✅ Complete isolation
#   ❌ Hard to manage, N collections to maintain
#
# STRATEGY B: One collection, metadata filter per query  ← WE USE THIS
#   ✅ Simple, single store, easy to add countries
#   ✅ Metadata filter enforces isolation at query time
#   ❌ Relies on always passing the filter (easy to enforce)
#
# With strategy B, every chunk has country="india" / "usa" / "uk"
# and we ALWAYS pass filter={"country": country} when querying.

# -----------------------------------------------------------
# 1. Load existing store and inspect it
# -----------------------------------------------------------
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

total = vectorstore._collection.count()
print(f"\nTotal vectors in store: {total}")

# -----------------------------------------------------------
# 2. Inspect what countries are stored
# -----------------------------------------------------------
print("\n" + "="*55)
print("WHAT IS CURRENTLY IN THE STORE?")
print("="*55)

# Fetch all records with their metadata
all_records = vectorstore._collection.get(include=["metadatas"])
metadatas   = all_records["metadatas"]

country_counts  = Counter(m.get("country", "unknown") for m in metadatas)
filename_counts = Counter(m.get("filename", "unknown") for m in metadatas)

print("\nChunks per country:")
for country, count in sorted(country_counts.items()):
    bar = "█" * (count // 2)
    print(f"  {country:8} → {count:3d} chunks  {bar}")

print("\nChunks per file:")
for filename, count in sorted(filename_counts.items()):
    print(f"  {filename:35} → {count:3d} chunks")

# -----------------------------------------------------------
# 3. Safe incremental ingestion
# -----------------------------------------------------------
# Problem: running from_documents() again DUPLICATES everything.
# Solution: check what's already stored before adding.

def get_stored_filenames(vectorstore: Chroma) -> set:
    """Return set of filenames already in the vector store."""
    records = vectorstore._collection.get(include=["metadatas"])
    return {m.get("filename") for m in records["metadatas"] if m.get("filename")}


def ingest_document(file_path: str, country: str,
                    vectorstore: Chroma, splitter) -> dict:
    """
    Safely ingest a document — skips if already stored.
    Returns a status dict.
    """
    path     = Path(file_path)
    filename = path.name
    existing = get_stored_filenames(vectorstore)

    if filename in existing:
        return {"status": "skipped", "filename": filename,
                "reason": "already in store"}

    loader = TextLoader(file_path, encoding="utf-8")
    docs   = loader.load()
    for doc in docs:
        doc.metadata.update({
            "country" : country.lower(),
            "filename": filename,
            "doc_type": "hr_policy",
        })

    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)      # ← adds WITHOUT duplicating

    return {"status": "added", "filename": filename,
            "chunks": len(chunks), "country": country}


print("\n" + "="*55)
print("INCREMENTAL INGESTION TEST")
print("="*55 + "\n")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# Try to add India doc again — should be skipped
result = ingest_document(
    "data/uploads/india/hr_policy_india.txt", "india", vectorstore, splitter
)
print(f"India policy  → {result['status'].upper()}: {result.get('reason', '')}")

# -----------------------------------------------------------
# 4. Verify country isolation
# -----------------------------------------------------------
print("\n" + "="*55)
print("VERIFYING COUNTRY ISOLATION")
print("="*55 + "\n")

query = "maternity leave"

for country in ["india", "usa", "uk"]:
    results = vectorstore.similarity_search(
        query, k=1, filter={"country": country}
    )
    if results:
        doc = results[0]
        print(f"{country.upper():6}: [{doc.metadata['filename']}]")
        print(f"        {doc.page_content[:120].strip()}...")
    print()

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# One Chroma collection + metadata filter = clean multi-tenancy.
# Always use get_stored_filenames() before ingesting to avoid
# duplicates. add_documents() is safe for incremental updates —
# from_documents() wipes and recreates (never call it twice!).
# -----------------------------------------------------------
