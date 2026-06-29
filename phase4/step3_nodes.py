# ============================================================
# PHASE 4 - STEP 3: Graph Nodes
# ============================================================
# What you'll learn:
#   - Converting Phase 1-3 logic into LangGraph nodes
#   - Each node is a plain Python function (state → dict)
#   - Nodes are stateless — they read state, return updates
#   - Five nodes: extract_intent, rag_search, rag_answer,
#     llm_supplement, web_search, format_response
# ============================================================

import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import operator

load_dotenv()

# ── shared state type (from step 2) ──────────────────────────
from typing import TypedDict

class HRGraphState(TypedDict):
    question        : str
    country         : str
    retrieved_docs  : list
    retrieval_score : float
    confidence      : float
    confidence_tier : str
    route           : str
    answer          : str
    sources         : list
    source_type     : str
    node_log        : Annotated[list[str], operator.add]
    web_results     : list

# ── setup shared resources ────────────────────────────────────
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")
HIGH_THRESHOLD    = 0.40
MEDIUM_THRESHOLD  = 0.25

print("Loading shared resources...")
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
print("Resources ready.\n")


# ============================================================
# NODE 1: extract_intent_node
# ============================================================
# Reads: question, country
# Writes: country (validated/corrected), node_log

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an HR system router. Extract the country from the question.
Return ONLY a JSON object with one key: "country"
Use lowercase country names: india, usa, uk, germany, australia, canada, singapore, uae, france, japan
If no country is mentioned, return the provided default.
Example: {{"country": "india"}}"""),
    ("human", "Question: {question}\nDefault country if none mentioned: {default_country}"),
])

def extract_intent_node(state: HRGraphState) -> dict:
    """Validate and confirm the country from the question."""
    from langchain_core.output_parsers import JsonOutputParser

    chain  = INTENT_PROMPT | llm | JsonOutputParser()
    result = chain.invoke({
        "question"      : state["question"],
        "default_country": state["country"],
    })
    country = result.get("country", state["country"]).lower()

    print(f"  [extract_intent] country={country}")
    return {
        "country" : country,
        "node_log": [f"extract_intent → country={country}"],
    }


# ============================================================
# NODE 2: rag_search_node
# ============================================================
# Reads: question, country
# Writes: retrieved_docs, retrieval_score, confidence,
#         confidence_tier, route

def rag_search_node(state: HRGraphState) -> dict:
    """Search ChromaDB and compute confidence score."""
    results = vectorstore.similarity_search_with_score(
        query  = state["question"],
        k      = 4,
        filter = {"country": state["country"]},
    )

    if not results:
        return {
            "retrieved_docs" : [],
            "retrieval_score": 999.0,
            "confidence"     : 0.0,
            "confidence_tier": "LOW",
            "route"          : "web_search",
            "node_log"       : ["rag_search → no results → LOW"],
        }

    docs         = [r[0] for r in results]
    best_dist    = min(r[1] for r in results)
    confidence   = round(1 / (1 + best_dist), 3)

    if confidence >= HIGH_THRESHOLD:
        tier  = "HIGH"
        route = "rag_answer"
    elif confidence >= MEDIUM_THRESHOLD:
        tier  = "MEDIUM"
        route = "llm_supplement"
    else:
        tier  = "LOW"
        route = "web_search"

    print(f"  [rag_search] dist={best_dist:.3f} conf={confidence:.3f} tier={tier}")
    return {
        "retrieved_docs" : docs,
        "retrieval_score": best_dist,
        "confidence"     : confidence,
        "confidence_tier": tier,
        "route"          : route,
        "node_log"       : [f"rag_search → conf={confidence:.3f} tier={tier}"],
    }


# ============================================================
# NODE 3: rag_answer_node  (HIGH confidence path)
# ============================================================
# Reads: question, country, retrieved_docs
# Writes: answer, sources, source_type

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional HR assistant for {country} employees.
Answer using ONLY the context below. Be specific — cite exact numbers and durations.

Context:
{context}"""),
    ("human", "{question}"),
])

def rag_answer_node(state: HRGraphState) -> dict:
    """Generate answer directly from retrieved HR documents."""
    docs    = state["retrieved_docs"]
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("filename", "unknown") for d in docs})

    chain  = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context" : context,
        "question": state["question"],
        "country" : state["country"].upper(),
    })

    print(f"  [rag_answer] answered from {len(docs)} doc chunks")
    return {
        "answer"     : answer,
        "sources"    : sources,
        "source_type": "hr_document",
        "node_log"   : [f"rag_answer → sources={sources}"],
    }


# ============================================================
# NODE 4: llm_supplement_node  (MEDIUM confidence path)
# ============================================================
# Reads: question, country, retrieved_docs
# Writes: answer, sources, source_type

SUPPLEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional HR assistant for {country} employees.
The context below is from company HR documents but may not fully answer the question.
Use it as a base and supplement with your knowledge of {country} employment law.
Clearly indicate which parts are from company policy vs general law.

Company HR context:
{context}"""),
    ("human", "{question}"),
])

def llm_supplement_node(state: HRGraphState) -> dict:
    """Supplement partial RAG results with LLM general knowledge."""
    docs    = state["retrieved_docs"]
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "No relevant documents found."
    sources = list({d.metadata.get("filename", "unknown") for d in docs})

    chain  = SUPPLEMENT_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context" : context,
        "question": state["question"],
        "country" : state["country"].upper(),
    })

    print(f"  [llm_supplement] answered with docs + LLM knowledge")
    return {
        "answer"     : answer,
        "sources"    : sources + ["LLM general knowledge"],
        "source_type": "llm_knowledge",
        "node_log"   : ["llm_supplement → docs + LLM"],
    }


# ============================================================
# NODE 5: web_search_node  (LOW confidence path)
# ============================================================
# Reads: question, country
# Writes: answer, sources, source_type, web_results

def web_search_node(state: HRGraphState) -> dict:
    """Search the web using Tavily for questions not in HR docs."""
    try:
        from langchain_tavily import TavilySearch
        tool    = TavilySearch(max_results=3)
        query   = f"{state['question']} {state['country']} employment law HR policy"
        results = tool.invoke({"query": query})

        # results is a list of dicts with 'url', 'content', 'title'
        if not results:
            raise ValueError("No results returned")

        # Build a context string from web results
        context_parts = []
        sources       = []
        for r in results:
            context_parts.append(
                f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}"
            )
            sources.append(r.get("url", "unknown"))

        context = "\n\n---\n\n".join(context_parts)

        WEB_PROMPT = ChatPromptTemplate.from_messages([
            ("system", """You are an HR assistant. Answer based on the web search results below.
Always cite the source URLs in your answer.

Web search results:
{context}"""),
            ("human", "{question}"),
        ])
        chain  = WEB_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": state["question"]})

        print(f"  [web_search] found {len(results)} web results")
        return {
            "answer"     : answer,
            "sources"    : sources,
            "source_type": "web_search",
            "web_results": results,
            "node_log"   : [f"web_search → {len(results)} results"],
        }

    except Exception as e:
        # Fallback if Tavily not configured yet
        print(f"  [web_search] Tavily not available: {e}")
        fallback_answer = (
            f"I searched for information about '{state['question']}' "
            f"but could not find it in the {state['country'].upper()} HR documents. "
            f"Please contact your HR team directly for the most accurate information."
        )
        return {
            "answer"     : fallback_answer,
            "sources"    : ["web_search_unavailable"],
            "source_type": "web_search",
            "web_results": [],
            "node_log"   : ["web_search → tavily unavailable → fallback"],
        }


# ============================================================
# NODE 6: format_response_node
# ============================================================
# Reads: answer, sources, source_type, confidence_tier
# Writes: answer (formatted with badge info)

def format_response_node(state: HRGraphState) -> dict:
    """Add source attribution and confidence badge to the answer."""
    tier        = state.get("confidence_tier", "LOW")
    source_type = state.get("source_type", "unknown")
    sources     = state.get("sources", [])

    badge_map = {
        "HIGH"  : "🟢 Answered from company HR documents",
        "MEDIUM": "🟡 Answered from HR docs + general knowledge",
        "LOW"   : "🔴 Answered from web search",
    }
    badge = badge_map.get(tier, "⚪ Source unknown")

    source_line = ""
    if sources:
        cleaned = [s for s in sources if s not in ("unknown", "web_search_unavailable")]
        if cleaned:
            source_line = f"\n\n📄 Sources: {', '.join(cleaned)}"

    formatted_answer = f"{state['answer']}{source_line}"

    print(f"  [format_response] badge={badge}")
    return {
        "answer"  : formatted_answer,
        "node_log": [f"format_response → {badge}"],
    }


# ============================================================
# QUICK TEST: run each node manually
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("TESTING EACH NODE INDEPENDENTLY")
    print("="*55)

    base_state: HRGraphState = {
        "question"       : "How many days of maternity leave in India?",
        "country"        : "india",
        "retrieved_docs" : [],
        "retrieval_score": 0.0,
        "confidence"     : 0.0,
        "confidence_tier": "",
        "route"          : "",
        "answer"         : "",
        "sources"        : [],
        "source_type"    : "",
        "node_log"       : [],
        "web_results"    : [],
    }

    print("\n1. rag_search_node:")
    search_result = rag_search_node(base_state)
    print(f"   tier={search_result['confidence_tier']} | route={search_result['route']}")

    print("\n2. rag_answer_node (using search results):")
    full_state = {**base_state, **search_result}
    answer_result = rag_answer_node(full_state)
    print(f"   answer preview: {answer_result['answer'][:100]}...")

    print("\n3. format_response_node:")
    full_state2 = {**full_state, **answer_result}
    final = format_response_node(full_state2)
    print(f"   final: {final['answer'][:150]}...")
