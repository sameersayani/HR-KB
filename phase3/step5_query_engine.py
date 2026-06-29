# ============================================================
# PHASE 3 - STEP 5: Smart Query Engine
# ============================================================
# What you'll learn:
#   - Combining confidence scoring + RAG into one clean engine
#   - The three routing paths (RAG / LLM supplement / signal fallback)
#   - Structured response object (answer, source, tier, route)
#   - This is the exact interface LangGraph will plug into
# ============================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")
HIGH_THRESHOLD    = 0.70
MEDIUM_THRESHOLD  = 0.45

# ============================================================
# SETUP
# ============================================================
print("Loading components...")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

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

# ============================================================
# PROMPTS
# ============================================================

# HIGH confidence → answer directly from docs
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional HR assistant for {country} employees.
Answer the question using ONLY the context provided.
Be specific — cite exact numbers, durations, and percentages from the context.
If the answer is partially in the context, share what you know clearly.

Context:
{context}"""),
    ("human", "{question}"),
])

# MEDIUM confidence → use docs but supplement with general knowledge
SUPPLEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional HR assistant for {country} employees.
The following context was retrieved from uploaded HR documents, but may not
fully answer the question. Use it as a starting point and supplement with
your general knowledge of {country} employment law where needed.
Clearly indicate which parts come from company policy vs general knowledge.

Context from company HR documents:
{context}"""),
    ("human", "{question}"),
])

# ============================================================
# QUERY ENGINE
# ============================================================

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def distance_to_confidence(distance: float) -> float:
    return round(1 / (1 + distance), 3)

def smart_query(question: str, country: str) -> dict:
    """
    The central query engine for Phase 3.

    Routing logic:
        HIGH   → answer from RAG docs directly
        MEDIUM → answer from RAG docs + LLM supplements
        LOW    → signal to caller: needs web search (Phase 4)

    Returns a structured dict that LangGraph will consume.
    """

    # --- Step 1: Retrieve with scores ---
    results = vectorstore.similarity_search_with_score(
        query  = question,
        k      = 4,
        filter = {"country": country.lower()},
    )

    if not results:
        return _build_response(
            question   = question,
            country    = country,
            answer     = None,
            docs       = [],
            confidence = 0.0,
            tier       = "LOW",
            route      = "web_search",
            sources    = [],
        )

    docs          = [r[0] for r in results]
    best_distance = min(r[1] for r in results)
    confidence    = distance_to_confidence(best_distance)

    # --- Step 2: Determine tier + route ---
    if confidence >= HIGH_THRESHOLD:
        tier  = "HIGH"
        route = "rag_answer"
    elif confidence >= MEDIUM_THRESHOLD:
        tier  = "MEDIUM"
        route = "llm_supplement"
    else:
        tier  = "LOW"
        route = "web_search"

    # --- Step 3: Generate answer (not for LOW tier) ---
    answer  = None
    sources = list({d.metadata.get("filename", "unknown") for d in docs})

    if tier == "HIGH":
        chain  = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context" : format_docs(docs),
            "question": question,
            "country" : country.upper(),
        })

    elif tier == "MEDIUM":
        chain  = SUPPLEMENT_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context" : format_docs(docs),
            "question": question,
            "country" : country.upper(),
        })

    # LOW tier: no answer generated — caller handles fallback

    return _build_response(
        question   = question,
        country    = country,
        answer     = answer,
        docs       = docs,
        confidence = confidence,
        tier       = tier,
        route      = route,
        sources    = sources,
    )


def _build_response(question, country, answer, docs,
                    confidence, tier, route, sources) -> dict:
    return {
        "question"  : question,
        "country"   : country,
        "answer"    : answer,
        "docs"      : docs,
        "confidence": confidence,
        "tier"      : tier,
        "route"     : route,
        "sources"   : sources,
    }

# ============================================================
# DISPLAY HELPER
# ============================================================
def display_result(result: dict):
    tier_icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    icon       = tier_icons.get(result["tier"], "⚪")

    print(f"\n{'='*60}")
    print(f"Q       : {result['question']}")
    print(f"Country : {result['country'].upper()}")
    print(f"Conf    : {result['confidence']:.2f}  Tier: {icon} {result['tier']}")
    print(f"Route   : {result['route']}")
    print(f"Sources : {', '.join(result['sources']) or 'none'}")
    print(f"{'='*60}")

    if result["answer"]:
        print(f"\n{result['answer']}")
    else:
        print("\n[No answer generated — Phase 4 will route to web search]")

# ============================================================
# TEST THE QUERY ENGINE
# ============================================================
if __name__ == "__main__":

    test_queries = [
        # HIGH confidence — should answer from docs
        ("How many days annual leave in India?",    "india"),
        ("What is the 401k matching policy?",       "usa"),
        ("How does UK shared parental leave work?", "uk"),

        # MEDIUM confidence — supplemented answer
        ("Can I take leave without prior approval?","india"),

        # LOW confidence — should route to web search
        ("What are the visa sponsorship options?",  "india"),
        ("Explain cryptocurrency tax in the UK",    "uk"),
    ]

    for question, country in test_queries:
        result = smart_query(question, country)
        display_result(result)

    # -----------------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------------
    print("\n" + "="*60)
    print("ROUTING SUMMARY")
    print("="*60)
    print(f"\n{'Question':<44} {'Tier':>6}  {'Route'}")
    print("-"*70)
    for question, country in test_queries:
        r = smart_query(question, country)
        print(f"{question:<44} {r['tier']:>6}  {r['route']}")

    # -----------------------------------------------------------
    # KEY TAKEAWAY
    # -----------------------------------------------------------
    # smart_query() returns a dict with:
    #   tier   → HIGH / MEDIUM / LOW
    #   route  → rag_answer / llm_supplement / web_search
    #   answer → filled for HIGH+MEDIUM, None for LOW
    #
    # In Phase 4, LangGraph nodes will call smart_query() and
    # use result["route"] to decide the next node to visit.
    # LOW → web_search_node → fetch live results → final answer
    # -----------------------------------------------------------
