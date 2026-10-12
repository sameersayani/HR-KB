# ============================================================
# PHASE 2 - STEP 4: Vector Store (ChromaDB)
# ============================================================
# What you'll learn:
#   - What a vector store is and why we need it
#   - Creating a Chroma collection from documents
#   - Persisting the vector store to disk (so we don't re-embed)
#   - Similarity search — the core retrieval operation
#   - Metadata filtering — country-specific search
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# -----------------------------------------------------------
# WHAT IS A VECTOR STORE?
# -----------------------------------------------------------
# A vector store is a database optimised for storing and
# searching vectors (embeddings).
#
# When you do similarity_search("maternity leave"), it:
#   1. Embeds your query into a vector
#   2. Compares it against ALL stored document vectors
#   3. Returns the k most similar chunks
#
# ChromaDB stores everything locally on disk — no server needed.
# Perfect for learning and small-to-medium HR knowledge bases.

# -----------------------------------------------------------
# 1. Setup: embeddings model
# -----------------------------------------------------------
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Model ready.\n")

# -----------------------------------------------------------
# 2. Load and chunk all documents
# -----------------------------------------------------------
def load_and_chunk(file_path: str, country: str) -> list:
    path   = Path(file_path)
    loader = TextLoader(file_path, encoding="utf-8")
    docs   = loader.load()
    for doc in docs:
        doc.metadata["country"]  = country
        doc.metadata["filename"] = path.name
        doc.metadata["doc_type"] = "hr_policy"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    )
    return splitter.split_documents(docs)

print("Loading and chunking documents...")
all_chunks = []
all_chunks += load_and_chunk("data/uploads/india/hr_policy_india.txt", "india")
all_chunks += load_and_chunk("data/uploads/usa/hr_policy_usa.txt",     "usa")
all_chunks += load_and_chunk("data/uploads/uk/hr_policy_uk.txt",       "uk")
print(f"Total chunks ready to embed: {len(all_chunks)}\n")

# -----------------------------------------------------------
# 3. Create the vector store
# -----------------------------------------------------------
# Chroma.from_documents():
#   - Takes your chunks + embedding model
#   - Embeds every chunk (calls the HuggingFace model)
#   - Stores vectors + metadata + text in persist_directory
#   - First run: slow (embedding takes time)
#   - Subsequent runs: load from disk (fast)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")

print(f"Creating vector store at: {VECTOR_STORE_PATH}")
print("Embedding documents (this takes 30-60 seconds on first run)...\n")

vectorstore = Chroma.from_documents(
    documents        = all_chunks,
    embedding        = embeddings,
    persist_directory= VECTOR_STORE_PATH,
    collection_name  = "hr_policies",
)

print(f"Vector store created!")
print(f"Total vectors stored: {vectorstore._collection.count()}\n")

# -----------------------------------------------------------
# 4. Basic similarity search
# -----------------------------------------------------------
print("="*55)
print("BASIC SIMILARITY SEARCH")
print("="*55)

query   = "How many days of maternity leave am I entitled to?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nQuery: '{query}'")
print(f"Top {len(results)} results:\n")

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Country  : {doc.metadata['country']}")
    print(f"File     : {doc.metadata['filename']}")
    print(f"Content  : {doc.page_content[:300]}...")
    print()

# -----------------------------------------------------------
# 5. Similarity search WITH SCORE
# -----------------------------------------------------------
print("="*55)
print("SEARCH WITH RELEVANCE SCORES")
print("="*55)

results_with_scores = vectorstore.similarity_search_with_score(query, k=4)

print(f"\nQuery: '{query}'\n")
for doc, score in results_with_scores:
    # Lower score = more similar (distance metric)
    relevance = "HIGH" if score < 0.5 else "MEDIUM" if score < 0.8 else "LOW"
    print(f"[{relevance}] Score: {score:.4f} | Country: {doc.metadata['country']}")
    print(f"  {doc.page_content[:150]}...")
    print()

# -----------------------------------------------------------
# 6. COUNTRY FILTERING — the key feature for our HR system
# -----------------------------------------------------------
print("="*55)
print("COUNTRY-FILTERED SEARCH")
print("="*55)

for country in ["india", "usa", "uk"]:
    results = vectorstore.similarity_search(
        query  = "maternity leave entitlement",
        k      = 1,
        filter = {"country": country},   # ← filters by metadata
    )
    print(f"\n{country.upper()} result:")
    if results:
        print(f"  {results[0].page_content[:200]}...")
    else:
        print("  No results found.")

# -----------------------------------------------------------
# 7. Loading an existing vector store (for next steps)
# -----------------------------------------------------------
print("\n" + "="*55)
print("RELOADING EXISTING VECTOR STORE (no re-embedding!)")
print("="*55)

# This is how you'll load it in step5 and step6 — fast!
loaded_store = Chroma(
    persist_directory = VECTOR_STORE_PATH,
    embedding_function= embeddings,
    collection_name   = "hr_policies",
)
print(f"Loaded store with {loaded_store._collection.count()} vectors")
print("No re-embedding needed — loaded from disk instantly!")

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# ChromaDB stores vectors + text + metadata on disk.
# similarity_search() finds semantically relevant chunks fast.
# filter={"country": "india"} restricts results to one country.
# Load with Chroma(...) after creation — no re-embedding needed.
# This is the heart of our RAG system.
# -----------------------------------------------------------
