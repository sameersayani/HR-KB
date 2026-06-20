# ============================================================
# PHASE 4 - STEP 2: Graph State Design
# ============================================================
# What you'll learn:
#   - Designing the state that flows through our HR graph
#   - Every field in the state and why it exists
#   - How state evolves as it passes through each node
#   - Annotated fields (for lists that append vs overwrite)
# ============================================================

from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
import operator

# ============================================================
# THE HR GRAPH STATE
# ============================================================
# This is the single most important design decision in LangGraph.
# Every node reads from this state and writes back to it.
# Get this right and the rest flows naturally.

class HRGraphState(TypedDict):

    # ── INPUT ─────────────────────────────────────────────
    question    : str           # the user's raw question
    country     : str           # "india" / "usa" / "uk" etc.

    # ── RETRIEVAL ─────────────────────────────────────────
    retrieved_docs  : list      # Document objects from ChromaDB
    retrieval_score : float     # best similarity score (0-1)
    confidence      : float     # converted confidence (0-1)
    confidence_tier : str       # "HIGH" / "MEDIUM" / "LOW"

    # ── ROUTING ───────────────────────────────────────────
    route       : str           # "rag_answer" / "llm_supplement" / "web_search"

    # ── ANSWER ────────────────────────────────────────────
    answer      : str           # final answer text
    sources     : list[str]     # list of source filenames / URLs
    source_type : str           # "hr_document" / "llm_knowledge" / "web_search"

    # ── METADATA ──────────────────────────────────────────
    # Annotated[list, operator.add] means each node APPENDS
    # to this list instead of replacing it — perfect for logs
    node_log    : Annotated[list[str], operator.add]
    web_results : list          # raw Tavily results (for Phase 4)


# ============================================================
# WHY EACH FIELD EXISTS
# ============================================================
print("=" * 55)
print("STATE FIELD WALKTHROUGH")
print("=" * 55)

field_docs = {
    "question"       : "Raw question from the user. Set at START, never changed.",
    "country"        : "Which country's HR docs to search. Set at START.",
    "retrieved_docs" : "Chunks returned by ChromaDB similarity search.",
    "retrieval_score": "Raw L2 distance from Chroma (lower = more similar).",
    "confidence"     : "1/(1+distance). 0=no match, 1=perfect match.",
    "confidence_tier": "HIGH/MEDIUM/LOW. Drives conditional edge routing.",
    "route"          : "Which answer node to visit next.",
    "answer"         : "The final answer shown to the user.",
    "sources"        : "Where the answer came from (file names or URLs).",
    "source_type"    : "hr_document / llm_knowledge / web_search.",
    "node_log"       : "List of nodes that ran. Appends (never overwrites).",
    "web_results"    : "Raw Tavily JSON results for LOW confidence queries.",
}

for field, desc in field_docs.items():
    print(f"\n  {field}")
    print(f"    → {desc}")

# ============================================================
# HOW STATE EVOLVES THROUGH THE GRAPH
# ============================================================
print("\n\n" + "=" * 55)
print("STATE EVOLUTION — STEP BY STEP")
print("=" * 55)

# We'll simulate what the state looks like after each node
# This is NOT running a graph — just showing the concept.

print("""
INITIAL STATE (set by user):
  question         = "How many days maternity leave in India?"
  country          = "india"
  retrieved_docs   = []
  retrieval_score  = 0.0
  confidence       = 0.0
  confidence_tier  = ""
  route            = ""
  answer           = ""
  sources          = []
  source_type      = ""
  node_log         = []
  web_results      = []

AFTER extract_intent_node:
  question         = "How many days maternity leave in India?"
  country          = "india"          ← confirmed/corrected
  node_log         = ["extract_intent_node"]

AFTER rag_search_node:
  retrieved_docs   = [Document(...), Document(...), ...]  ← top 4 chunks
  retrieval_score  = 0.42             ← L2 distance
  confidence       = 0.70             ← 1/(1+0.42)
  confidence_tier  = "HIGH"
  route            = "rag_answer"     ← set here
  node_log         = ["extract_intent_node", "rag_search_node"]

AFTER rag_answer_node (routed via conditional edge):
  answer           = "Employees in India are entitled to 26 weeks..."
  sources          = ["hr_policy_india.txt"]
  source_type      = "hr_document"
  node_log         = [..., "rag_answer_node"]

AFTER format_response_node:
  answer           = (same, formatted for display)
  node_log         = [..., "format_response_node"]

→ END: return final state to user
""")

# ============================================================
# DEMO: Build a tiny graph just to prove Annotated append works
# ============================================================
print("=" * 55)
print("ANNOTATED LIST DEMO (append vs overwrite)")
print("=" * 55)

class DemoState(TypedDict):
    value   : str
    log     : Annotated[list[str], operator.add]  # appends!

def node_a(state: DemoState) -> dict:
    return {"value": "set by A", "log": ["node_a ran"]}

def node_b(state: DemoState) -> dict:
    return {"value": "set by B", "log": ["node_b ran"]}

def node_c(state: DemoState) -> dict:
    return {"log": ["node_c ran"]}  # only updates log

g = StateGraph(DemoState)
g.add_node("a", node_a)
g.add_node("b", node_b)
g.add_node("c", node_c)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", "c")
g.add_edge("c", END)

result = g.compile().invoke({"value": "", "log": []})
print(f"\nFinal value : {result['value']}")
print(f"Final log   : {result['log']}")
print("""
Notice: log has ALL three entries — it APPENDED each time.
Without Annotated[list, operator.add], only the last
node's log entry would survive (it would overwrite).
This is how node_log works in our HR graph.
""")

# ============================================================
# KEY TAKEAWAY
# ============================================================
# HRGraphState is the backbone of Phase 4.
# Every node reads from it and writes partial updates back.
# The confidence_tier field is what drives conditional routing:
#   "HIGH"   → rag_answer_node
#   "MEDIUM" → llm_supplement_node
#   "LOW"    → web_search_node
#
# Annotated[list, operator.add] is essential for any field
# where multiple nodes contribute (logs, sources, results).
# ============================================================
