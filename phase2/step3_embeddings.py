# ============================================================
# PHASE 2 - STEP 3: Embeddings
# ============================================================
# What you'll learn:
#   - What embeddings ARE (conceptually and practically)
#   - HuggingFaceEmbeddings — free, local, no API key required
#   - How to embed a single text and a list of texts
#   - Why similar texts have similar vectors (cosine similarity)
#   - How this powers semantic search in our RAG system
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# -----------------------------------------------------------
# WHAT IS AN EMBEDDING?
# -----------------------------------------------------------
# An embedding is a list of numbers (a vector) that represents
# the MEANING of a piece of text in mathematical space.
#
# Texts with similar meanings → vectors that are close together
# Texts with different meanings → vectors that are far apart
#
# Example:
#   "maternity leave policy"   → [0.12, -0.45, 0.89, ...]
#   "parental leave for women" → [0.11, -0.47, 0.91, ...]  ← very close!
#   "company car policy"       → [-0.78, 0.23, -0.12, ...]  ← far away
#
# This is how RAG finds relevant chunks WITHOUT keyword matching.
# It finds chunks that are semantically close to the question.

# -----------------------------------------------------------
# 1. Initialise HuggingFace embeddings
# -----------------------------------------------------------
# First run: downloads ~90MB model to ~/.cache/huggingface/
# Subsequent runs: loads from cache (fast)
#
# "all-MiniLM-L6-v2" is the most popular embedding model:
#   - 384-dimensional vectors
#   - Very fast (runs on CPU)
#   - Great accuracy for semantic similarity

print("Loading embedding model (downloads ~90MB on first run)...")
print("This may take 1-2 minutes the first time.\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},   # use "cuda" if you have GPU
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding model loaded!\n")

# -----------------------------------------------------------
# 2. Embed a single piece of text
# -----------------------------------------------------------
sample_text = "How many days of maternity leave is an employee entitled to?"
vector = embeddings.embed_query(sample_text)

print(f"Text   : {sample_text}")
print(f"Vector dimensions : {len(vector)}")
print(f"First 5 numbers   : {[round(v, 4) for v in vector[:5]]}")
print(f"Last  5 numbers   : {[round(v, 4) for v in vector[-5:]]}")

# -----------------------------------------------------------
# 3. Embed multiple documents (batch)
# -----------------------------------------------------------
# embed_documents() is optimised for batches (more efficient)
hr_texts = [
    "Employees are entitled to 26 weeks of maternity leave in India.",
    "The probation period is 6 months from the date of joining.",
    "Annual leave accrues at 1.75 days per month.",
    "Overtime is paid at 2x the normal rate for hours beyond 45 per week.",
]

doc_vectors = embeddings.embed_documents(hr_texts)
print(f"\nEmbedded {len(doc_vectors)} documents")
print(f"Each vector has {len(doc_vectors[0])} dimensions")

# -----------------------------------------------------------
# 4. Semantic similarity — the magic of embeddings
# -----------------------------------------------------------
# Cosine similarity measures how "close" two vectors are.
# Range: -1 (opposite) to 1 (identical)
# For normalized vectors: closer to 1 = more similar

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

print("\n" + "="*55)
print("SEMANTIC SIMILARITY DEMO")
print("="*55)

query = "What is the maternity leave entitlement?"
query_vec = embeddings.embed_query(query)

print(f"\nQuery: '{query}'\n")
print("Similarity scores against HR texts:")
for text, vec in zip(hr_texts, doc_vectors):
    score = cosine_similarity(query_vec, vec)
    bar   = "█" * int(score * 30)
    print(f"  {score:.4f} {bar}")
    print(f"          {text[:70]}")
    print()

# -----------------------------------------------------------
# 5. Keyword vs Semantic search
# -----------------------------------------------------------
print("="*55)
print("KEYWORD vs SEMANTIC SEARCH COMPARISON")
print("="*55)

semantic_tests = [
    ("maternity leave",         "parental benefit for mothers"),
    ("probation period",        "trial period for new joiners"),
    ("annual leave",            "paid vacation days per year"),
    ("maternity leave",         "company car benefit"),       # should be LOW
]

for text_a, text_b in semantic_tests:
    vec_a = embeddings.embed_query(text_a)
    vec_b = embeddings.embed_query(text_b)
    score = cosine_similarity(vec_a, vec_b)
    match = "HIGH ✓" if score > 0.5 else "LOW  ✗"
    print(f"  {match} ({score:.3f})  '{text_a}' vs '{text_b}'")

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# Embeddings turn text into vectors that capture MEANING.
# Similar meanings = similar vectors = high cosine similarity.
# This is why RAG can answer "parental benefit for mothers"
# even if the document only says "maternity leave" — they're
# semantically close in vector space.
#
# Next step: store these vectors in ChromaDB so we can
# search them efficiently at query time.
# -----------------------------------------------------------
