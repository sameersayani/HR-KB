# ============================================================
# PHASE 4 - STEP 4: Graph Assembly
# ============================================================
# What you'll learn:
#   - Assembling all nodes into a complete LangGraph
#   - Wiring conditional edges for the three routing paths
#   - Compiling, visualising, and running the full graph
#   - Streaming output node by node
# ============================================================

import os, sys
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4.step3_nodes import (
    HRGraphState,
    extract_intent_node,
    rag_search_node,
    rag_answer_node,
    llm_supplement_node,
    web_search_node,
    format_response_node,
)

# ============================================================
# ROUTING FUNCTION
# ============================================================
# Called by LangGraph after rag_search_node runs.
# Reads state["route"] and returns the next node name.

def route_after_search(state: HRGraphState) -> Literal["rag_answer", "llm_supplement", "web_search"]:
    """Conditional edge — reads route from state."""
    route = state.get("route", "web_search")
    print(f"  [router] → {route}")
    return route

# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_hr_graph() -> StateGraph:
    """Assemble and compile the full HR LangGraph."""

    g = StateGraph(HRGraphState)

    # ── Add all nodes ─────────────────────────────────────
    g.add_node("extract_intent",  extract_intent_node)
    g.add_node("rag_search",      rag_search_node)
    g.add_node("rag_answer",      rag_answer_node)
    g.add_node("llm_supplement",  llm_supplement_node)
    g.add_node("web_search",      web_search_node)
    g.add_node("format_response", format_response_node)

    # ── Fixed edges ───────────────────────────────────────
    g.add_edge(START,             "extract_intent")
    g.add_edge("extract_intent",  "rag_search")

    # ── Conditional edge (the routing brain) ─────────────
    g.add_conditional_edges(
        "rag_search",          # source node
        route_after_search,    # routing function
        {                      # mapping: return value → next node
            "rag_answer"    : "rag_answer",
            "llm_supplement": "llm_supplement",
            "web_search"    : "web_search",
        }
    )

    # ── All paths converge to format_response then END ───
    g.add_edge("rag_answer",     "format_response")
    g.add_edge("llm_supplement", "format_response")
    g.add_edge("web_search",     "format_response")
    g.add_edge("format_response", END)

    return g.compile()


# ============================================================
# HELPER: build initial state
# ============================================================

def make_initial_state(question: str, country: str) -> HRGraphState:
    return {
        "question"       : question,
        "country"        : country.lower(),
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


# ============================================================
# DISPLAY HELPER
# ============================================================

def display_result(result: HRGraphState):
    tier_icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    icon       = tier_icons.get(result.get("confidence_tier", ""), "⚪")

    print(f"\n{'='*60}")
    print(f"QUESTION  : {result['question']}")
    print(f"COUNTRY   : {result['country'].upper()}")
    print(f"CONFIDENCE: {result.get('confidence', 0):.3f}  {icon} {result.get('confidence_tier', '')}")
    print(f"SOURCE    : {result.get('source_type', '')}")
    print(f"NODE PATH : {' → '.join(result.get('node_log', []))}")
    print(f"{'='*60}")
    print(f"\n{result.get('answer', 'No answer generated.')}")


# ============================================================
# MAIN: run the graph on test queries
# ============================================================

if __name__ == "__main__":
    print("Building HR LangGraph...")
    graph = build_hr_graph()
    print("Graph compiled successfully.\n")

    # ── Test queries ─────────────────────────────────────
    test_cases = [
        # HIGH confidence — should come from HR docs
        ("How many days of annual leave do I get?",     "india"),
        ("What is the maternity leave policy?",         "usa"),
        ("What is the notice period in the UK?",        "uk"),

        # MEDIUM confidence — supplemented
        ("Can I carry forward unused leave?",           "india"),

        # LOW confidence — web search
        ("What are the visa sponsorship options?",      "india"),
    ]

    for question, country in test_cases:
        print(f"\nRunning graph: '{question}' [{country.upper()}]")
        print("-" * 55)

        state   = make_initial_state(question, country)
        result  = graph.invoke(state)
        display_result(result)

    # ── Streaming mode ────────────────────────────────────
    print("\n\n" + "="*60)
    print("STREAMING MODE — see nodes execute one by one")
    print("="*60)

    stream_state = make_initial_state(
        "How is gratuity calculated in India?", "india"
    )

    print("\nStreaming graph execution:\n")
    for step in graph.stream(stream_state):
        for node_name, node_output in step.items():
            print(f"  ✓ {node_name} completed")
            if "answer" in node_output and node_output["answer"]:
                print(f"    preview: {str(node_output['answer'])[:80]}...")
