# ============================================================
# PHASE 2 - STEP 5: Retrieval
# ============================================================
# What you'll learn:
#   - Turning a vector store into a LangChain retriever
#   - The retriever interface (.invoke() for queries)
#   - Retrieval strategies: similarity, mmr, threshold
#   - Building a country-aware retriever for our HR system
#   - Scoring and deciding "did we find a good answer?"
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")

# -----------------------------------------------------------
# 1. Load the vector store (already created in step 4)
# -----------------------------------------------------------
print("Loading vector store from disk...")
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
print(f"Loaded {vectorstore._collection.count()} vectors.\n")

# -----------------------------------------------------------
# 2. What is a Retriever?
# -----------------------------------------------------------
# A retriever is a thin wrapper around the vector store that
# exposes a clean .invoke(query) → List[Document] interface.
# It integrates natively with LangChain chains and LangGraph.

# Basic retriever (top 4 most similar chunks)
retriever = vectorstore.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 4},
)

results = retriever.invoke("What is the annual leave entitlement?")
print("="*55)
print(f"Basic retriever — top 4 results for 'annual leave':")
print("="*55)
for doc in results:
    print(f"  [{doc.metadata['country']:6}] {doc.page_content[:120]}...")
print()

# -----------------------------------------------------------
# 3. Country-aware retriever
# -----------------------------------------------------------
# filter limits results to a specific country's documents.
# This is the retriever we'll use in our actual HR chatbot.

def get_country_retriever(country: str, k: int = 4):
    """Return a retriever scoped to a single country."""
    return vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {
            "k"     : k,
            "filter": {"country": country.lower()},
        },
    )

print("="*55)
print("COUNTRY-AWARE RETRIEVERS")
print("="*55)

query = "How many days of sick leave am I entitled to?"
print(f"\nQuery: '{query}'\n")

for country in ["india", "usa", "uk"]:
    retriever_c = get_country_retriever(country, k=1)
    docs        = retriever_c.invoke(query)
    if docs:
        print(f"{country.upper():5}: {docs[0].page_content[:200]}...")
    else:
        print(f"{country.upper():5}: No results found.")
    print()

# -----------------------------------------------------------
# 4. MMR (Maximal Marginal Relevance) — avoiding repetition
# -----------------------------------------------------------
# Problem with pure similarity: top results can be near-duplicates.
# MMR balances relevance AND diversity — useful for long answers.

mmr_retriever = vectorstore.as_retriever(
    search_type   = "mmr",
    search_kwargs = {
        "k"              : 4,
        "fetch_k"        : 20,     # candidates to consider
        "lambda_mult"    : 0.7,    # 1.0=pure similarity, 0.0=pure diversity
    },
)

print("="*55)
print("MMR RETRIEVER (diversity-aware)")
print("="*55)
mmr_results = mmr_retriever.invoke("leave policies and entitlements")
print(f"\nMMR top 4 results (more diverse than pure similarity):")
for doc in mmr_results:
    print(f"  [{doc.metadata['country']:6}] {doc.page_content[:100]}...")
print()

# -----------------------------------------------------------
# 5. Relevance scoring — knowing when RAG "didn't find it"
# -----------------------------------------------------------
# This is critical for our system: if score is too low,
# we fall back to LLM or web search (that's the LangGraph logic).

def retrieve_with_confidence(query: str, country: str, threshold: float = 0.6):
    """
    Retrieve documents and return a confidence signal.
    Chroma returns L2 distance — lower = more similar.
    We convert to a 0-1 confidence score.
    """
    results = vectorstore.similarity_search_with_score(
        query  = query,
        k      = 3,
        filter = {"country": country.lower()},
    )

    if not results:
        return [], "not_found", 0.0

    # Chroma L2 distance: 0=identical, higher=less similar
    # Convert: confidence = 1 / (1 + distance)
    best_score    = results[0][1]
    confidence_val = round(1 / (1 + best_score), 3)
    docs          = [r[0] for r in results]

    if confidence_val >= threshold:
        confidence_label = "HIGH — answer from RAG"
    elif confidence_val >= 0.3:
        confidence_label = "MEDIUM — check LLM too"
    else:
        confidence_label = "LOW — fallback to web search"

    return docs, confidence_label, confidence_val

print("="*55)
print("CONFIDENCE-BASED RETRIEVAL (our routing logic!)")
print("="*55 + "\n")

test_queries = [
    ("How many days of maternity leave in India?", "india"),
    ("What is the gratuity calculation formula?",  "india"),
    ("What is quantum computing?",                 "india"),  # not in HR docs
]

for query, country in test_queries:
    docs, label, score = retrieve_with_confidence(query, country)
    print(f"Query  : {query}")
    print(f"Country: {country.upper()}")
    print(f"Result : {label} (score: {score})")
    if docs:
        print(f"Top doc: {docs[0].page_content[:120]}...")
    print()

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# retriever.invoke(query) → List[Document] (LangChain native)
# filter={"country": "india"} scopes search to one country
# Confidence scoring tells us WHEN to trust RAG vs fall back
# This confidence signal is exactly what LangGraph will use
# to route between RAG → LLM → Web Search in Phase 4.
# -----------------------------------------------------------
