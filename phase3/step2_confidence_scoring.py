# ============================================================
# PHASE 3 - STEP 2: Confidence Scoring Engine
# ============================================================
# What you'll learn:
#   - Why a single RAG answer isn't enough — we need a signal
#   - Converting Chroma distance scores to 0-1 confidence
#   - Three-tier routing: HIGH / MEDIUM / LOW
#   - How this confidence signal drives LangGraph routing (Phase 4)
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")

print("Loading vector store...")
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
print("Ready.\n")

# -----------------------------------------------------------
# WHY CONFIDENCE SCORING?
# -----------------------------------------------------------
# RAG always returns SOMETHING — even for irrelevant queries.
# Without scoring, "What is quantum computing?" would return
# the least-irrelevant HR chunk and the LLM would hallucinate.
#
# We need to KNOW when the retrieved content is trustworthy
# so we can decide: use it, supplement it, or ignore it.

# -----------------------------------------------------------
# 1. Understanding Chroma distance scores
# -----------------------------------------------------------
# Chroma returns L2 (Euclidean) distance:
#   0.0  = identical vectors (perfect match)
#   0.5  = moderate similarity
#   1.0+ = low similarity (different topics)
#
# We convert to a 0-1 confidence scale:
#   confidence = 1 / (1 + distance)
#   distance=0.0 → confidence=1.00  (perfect)
#   distance=0.3 → confidence=0.77  (strong)
#   distance=0.6 → confidence=0.63  (moderate)
#   distance=1.0 → confidence=0.50  (weak)
#   distance=2.0 → confidence=0.33  (poor)

def distance_to_confidence(distance: float) -> float:
    return round(1 / (1 + distance), 3)

print("="*55)
print("DISTANCE → CONFIDENCE CONVERSION")
print("="*55)
for d in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
    c   = distance_to_confidence(d)
    bar = "█" * int(c * 20)
    print(f"  dist={d:.1f} → conf={c:.2f}  {bar}")

# -----------------------------------------------------------
# 2. Three-tier routing thresholds
# -----------------------------------------------------------
# HIGH   (≥0.70) → RAG answer is reliable, use it directly
# MEDIUM (≥0.45) → Partial match, supplement with LLM knowledge
# LOW    (<0.45)  → No relevant content, fall back to web search

HIGH_THRESHOLD   = 0.70
MEDIUM_THRESHOLD = 0.45

def get_confidence_tier(confidence: float) -> str:
    if confidence >= HIGH_THRESHOLD:
        return "HIGH"
    elif confidence >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"

# -----------------------------------------------------------
# 3. Full confidence scoring function
# -----------------------------------------------------------
def score_retrieval(query: str, country: str, k: int = 4) -> dict:
    """
    Retrieve documents and compute a confidence score.

    Returns:
        docs        → retrieved Document objects
        scores      → raw distance scores
        confidence  → 0-1 float (best chunk)
        tier        → HIGH / MEDIUM / LOW
        route       → what the system should do next
    """
    results = vectorstore.similarity_search_with_score(
        query  = query,
        k      = k,
        filter = {"country": country.lower()},
    )

    if not results:
        return {
            "docs"      : [],
            "scores"    : [],
            "confidence": 0.0,
            "tier"      : "LOW",
            "route"     : "web_search",
        }

    docs    = [r[0] for r in results]
    scores  = [r[1] for r in results]

    # Best (lowest) distance drives the confidence
    best_distance  = min(scores)
    confidence     = distance_to_confidence(best_distance)
    tier           = get_confidence_tier(confidence)

    route_map = {
        "HIGH"  : "rag_answer",      # trust RAG, answer directly
        "MEDIUM": "llm_supplement",  # use RAG + ask LLM to expand
        "LOW"   : "web_search",      # ignore RAG, search web
    }

    return {
        "docs"      : docs,
        "scores"    : scores,
        "confidence": confidence,
        "tier"      : tier,
        "route"     : route_map[tier],
    }

# -----------------------------------------------------------
# 4. Test with a variety of queries
# -----------------------------------------------------------
test_cases = [
    # (question, country, expected tier)
    ("How many days of annual leave do I get?",     "india", "HIGH"),
    ("What is the maternity leave policy?",         "usa",   "HIGH"),
    ("What is the notice period in UK?",            "uk",    "HIGH"),
    ("Can I work remotely from another country?",   "india", "MEDIUM"),
    ("What are the visa sponsorship rules?",        "usa",   "LOW"),
    ("Explain quantum entanglement",                "uk",    "LOW"),
]

print("\n" + "="*55)
print("CONFIDENCE SCORING — ROUTING DECISIONS")
print("="*55 + "\n")
print(f"{'Query':<45} {'Country':<7} {'Conf':>5}  {'Tier':>6}  {'Route'}")
print("-"*85)

for question, country, expected in test_cases:
    result = score_retrieval(question, country)
    match  = "✓" if result["tier"] == expected else "?"
    print(f"{question:<45} {country:<7} {result['confidence']:>5.2f}  "
          f"{result['tier']:>6}  {result['route']}  {match}")

# -----------------------------------------------------------
# 5. Drill into a HIGH confidence result
# -----------------------------------------------------------
print("\n" + "="*55)
print("DRILL-DOWN: HIGH CONFIDENCE RESULT")
print("="*55 + "\n")

result = score_retrieval("maternity leave entitlement weeks", "india")
print(f"Query      : maternity leave entitlement weeks")
print(f"Country    : INDIA")
print(f"Confidence : {result['confidence']} ({result['tier']})")
print(f"Route      : {result['route']}")
print(f"\nTop chunk  :\n{result['docs'][0].page_content[:400]}")
print(f"\nAll chunk distances: {[round(s,3) for s in result['scores']]}")

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# confidence score + tier + route is the output that
# LangGraph nodes will read in Phase 4 to decide:
#
#   HIGH   → go to rag_answer_node
#   MEDIUM → go to llm_supplement_node
#   LOW    → go to web_search_node
#
# This one function is the brain of the routing logic.
# -----------------------------------------------------------
