# ============================================================
# PHASE 5 - STEP 1: Multi-Agent Concepts
# ============================================================
# What you'll learn:
#   - How "multi-agent" differs from the conditional routing in Phase 4
#   - The Supervisor pattern
#   - Fan-out / fan-in: calling multiple agents in PARALLEL
#   - LangGraph's Send() API for dynamic parallel dispatch
#   - Why a Summarizer agent is needed when multiple agents run
# ============================================================

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

# ------------------------------------------------------------
# PHASE 4 vs PHASE 5 — THE KEY DIFFERENCE
# ------------------------------------------------------------
# Phase 4 (conditional routing):
#   One question → confidence score → EXACTLY ONE path
#   HIGH   → rag_answer       (only this runs)
#   MEDIUM → llm_supplement   (only this runs)
#   LOW    → web_search       (only this runs)
#
# Phase 5 (multi-agent):
#   One question → Supervisor decides → MULTIPLE agents may run
#   "What is maternity leave and is it different from EU law?"
#     → RAG Agent     (company docs)       ✅ runs
#     → Policy Agent  (general law)        ✅ runs
#     → Web Agent     (live search)        ❌ not needed
#   → Summarizer Agent combines BOTH outputs into one answer
#
# This is genuine multi-agent collaboration: agents work in
# PARALLEL and their outputs get MERGED, not just routed.

print("=" * 60)
print("CONCEPT: Send() for dynamic fan-out")
print("=" * 60)
print("""
Normally, add_conditional_edges() picks ONE next node.

Send() lets a single node dispatch to MULTIPLE nodes at once,
each with its OWN custom input. This is how the Supervisor
triggers 1, 2, or 3 agents in parallel based on the question.

    return [
        Send("rag_agent",    {"question": q, "country": c}),
        Send("policy_agent", {"question": q, "country": c}),
    ]

LangGraph runs both nodes CONCURRENTLY, then waits for both
to finish before continuing — this is "fan-out, then fan-in".
""")

# ------------------------------------------------------------
# MINI DEMO: Supervisor dispatches to 2 of 3 possible agents
# ------------------------------------------------------------

class DemoState(TypedDict):
    question      : str
    needed_agents : list[str]
    # Annotated + operator.add → every agent's result APPENDS
    # to this list instead of overwriting each other.
    agent_results : Annotated[list[dict], operator.add]
    final_answer  : str


def supervisor_node(state: DemoState) -> dict:
    """Decide which agents are needed (simplified rule-based demo)."""
    q = state["question"].lower()
    needed = []
    if "policy" in q or "leave" in q:
        needed.append("rag_agent")
    if "law" in q or "compare" in q or "eu" in q:
        needed.append("policy_agent")
    if "latest" in q or "2026" in q:
        needed.append("web_agent")
    if not needed:
        needed = ["rag_agent"]   # sensible default

    print(f"  [supervisor] needed_agents = {needed}")
    return {"needed_agents": needed}


def dispatch_to_agents(state: DemoState):
    """
    This is a ROUTING FUNCTION used with add_conditional_edges,
    but instead of returning a single node name, it returns a
    LIST of Send() objects — one per agent to invoke.
    """
    return [
        Send(agent_name, {"question": state["question"]})
        for agent_name in state["needed_agents"]
    ]


def rag_agent_demo(state: dict) -> dict:
    print(f"  [rag_agent] running on: {state['question']}")
    return {"agent_results": [{"agent": "rag_agent", "answer": "From HR docs..."}]}

def policy_agent_demo(state: dict) -> dict:
    print(f"  [policy_agent] running on: {state['question']}")
    return {"agent_results": [{"agent": "policy_agent", "answer": "From legal knowledge..."}]}

def web_agent_demo(state: dict) -> dict:
    print(f"  [web_agent] running on: {state['question']}")
    return {"agent_results": [{"agent": "web_agent", "answer": "From web search..."}]}

def summarizer_demo(state: DemoState) -> dict:
    parts = [f"[{r['agent']}]: {r['answer']}" for r in state["agent_results"]]
    combined = " | ".join(parts)
    print(f"  [summarizer] combining {len(state['agent_results'])} agent outputs")
    return {"final_answer": combined}


# ------------------------------------------------------------
# BUILD THE DEMO GRAPH
# ------------------------------------------------------------
g = StateGraph(DemoState)
g.add_node("supervisor",    supervisor_node)
g.add_node("rag_agent",     rag_agent_demo)
g.add_node("policy_agent",  policy_agent_demo)
g.add_node("web_agent",     web_agent_demo)
g.add_node("summarizer",    summarizer_demo)

g.add_edge(START, "supervisor")

# Conditional edge that returns Send() objects = fan-out
g.add_conditional_edges(
    "supervisor",
    dispatch_to_agents,
    ["rag_agent", "policy_agent", "web_agent"],   # possible destinations
)

# Fan-in: ALL agents converge to summarizer
g.add_edge("rag_agent",    "summarizer")
g.add_edge("policy_agent", "summarizer")
g.add_edge("web_agent",    "summarizer")
g.add_edge("summarizer",   END)

graph = g.compile()

# ------------------------------------------------------------
# TEST: different questions trigger different agent combos
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST RUNS — Watch which agents get triggered")
print("=" * 60)

test_questions = [
    "What is the leave policy?",                       # → rag only
    "Compare our leave policy with EU labour law",      # → rag + policy
    "What's the latest 2026 visa rule and our leave policy?",  # → rag + web
]

for q in test_questions:
    print(f"\nQuestion: '{q}'")
    result = graph.invoke({
        "question": q, "needed_agents": [], "agent_results": [], "final_answer": "",
    })
    print(f"Agents triggered : {[r['agent'] for r in result['agent_results']]}")
    print(f"Final answer     : {result['final_answer']}")

# ------------------------------------------------------------
# KEY TAKEAWAY
# ------------------------------------------------------------
# Send(node_name, custom_input) → dynamic parallel dispatch
# add_conditional_edges(source, fn_returning_list_of_Send, [...possible nodes])
# Annotated[list, operator.add] → safely collect parallel results
# A Summarizer node is REQUIRED whenever more than one agent
# might run — someone has to merge their outputs into one answer.
# ------------------------------------------------------------
