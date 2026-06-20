# ============================================================
# PHASE 4 - STEP 1: LangGraph Core Concepts
# ============================================================
# What you'll learn:
#   - What StateGraph is and how it differs from a LangChain chain
#   - TypedDict state — the shared memory of the graph
#   - Nodes: plain Python functions that read/write state
#   - Edges: how nodes connect to each other
#   - Conditional edges: routing based on state values
#   - Compiling and running a graph
# ============================================================

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# ============================================================
# CONCEPT 1: STATE
# ============================================================
# State is a TypedDict — a typed Python dictionary that is
# passed between every node in the graph.
#
# Each node READS from state and RETURNS a partial update.
# LangGraph merges updates automatically — you never replace
# the whole state, only the keys you care about.

class SimpleState(TypedDict):
    question : str          # the input question
    country  : str          # detected country
    answer   : str          # generated answer
    step_log : list[str]    # track which nodes ran

# ============================================================
# CONCEPT 2: NODES
# ============================================================
# A node is any Python function that:
#   - Takes state (dict) as its only argument
#   - Returns a dict with ONLY the keys it wants to update
#
# LangGraph merges the returned dict into the current state.
# You don't need to return every key — just the ones you changed.

def greet_node(state: SimpleState) -> dict:
    """Node 1: Log that we received the question."""
    print(f"  [greet_node] Question received: '{state['question']}'")
    return {
        "step_log": state["step_log"] + ["greet_node ran"],
    }

def detect_country_node(state: SimpleState) -> dict:
    """Node 2: Detect country from question (simplified)."""
    question = state["question"].lower()
    if "india" in question or "pf" in question or "gratuity" in question:
        country = "india"
    elif "usa" in question or "401k" in question or "fmla" in question:
        country = "usa"
    elif "uk" in question or "redundancy" in question or "paye" in question:
        country = "uk"
    else:
        country = "general"

    print(f"  [detect_country_node] Detected country: {country}")
    return {
        "country" : country,
        "step_log": state["step_log"] + [f"detect_country_node → {country}"],
    }

def answer_node(state: SimpleState) -> dict:
    """Node 3: Generate a simple answer."""
    answer = (f"This is a {state['country'].upper()} HR question. "
              f"A full answer would come from RAG + LLM.")
    print(f"  [answer_node] Generated answer.")
    return {
        "answer"  : answer,
        "step_log": state["step_log"] + ["answer_node ran"],
    }

# ============================================================
# CONCEPT 3: BUILDING THE GRAPH
# ============================================================
# 1. Create a StateGraph with your state type
# 2. Add nodes (name → function)
# 3. Add edges (name → name)
# 4. Compile — validates the graph and returns a runnable

print("=" * 55)
print("BUILDING A SIMPLE 3-NODE GRAPH")
print("=" * 55)

graph_builder = StateGraph(SimpleState)

# Add nodes
graph_builder.add_node("greet",          greet_node)
graph_builder.add_node("detect_country", detect_country_node)
graph_builder.add_node("generate_answer", answer_node)

# Add edges — define the flow
graph_builder.add_edge(START,            "greet")
graph_builder.add_edge("greet",          "detect_country")
graph_builder.add_edge("detect_country", "generate_answer")
graph_builder.add_edge("generate_answer",         END)

# Compile the graph
graph = graph_builder.compile()
print("Graph compiled successfully.\n")

# ============================================================
# CONCEPT 4: RUNNING THE GRAPH
# ============================================================
# .invoke() runs the full graph from START to END.
# You pass the initial state as a dict.
# It returns the final state after all nodes have run.

print("Running graph with: 'What is the PF contribution in India?'\n")
result = graph.invoke({
    "question" : "What is the PF contribution in India?",
    "country"  : "",
    "answer"   : "",
    "step_log" : [],
})

print(f"\nFinal state:")
print(f"  question : {result['question']}")
print(f"  country  : {result['country']}")
print(f"  answer   : {result['answer']}")
print(f"  step_log : {result['step_log']}")

# ============================================================
# CONCEPT 5: CONDITIONAL EDGES (routing)
# ============================================================
# This is the key LangGraph feature — routing to different
# nodes based on a value in the state.

class RoutedState(TypedDict):
    question  : str
    country   : str
    confidence: str     # "high", "medium", "low"
    answer    : str

def mock_rag_node(state: RoutedState) -> dict:
    """Simulate RAG search — returns a confidence level."""
    q = state["question"].lower()
    if any(w in q for w in ["maternity", "annual leave", "probation", "gratuity"]):
        confidence = "high"
    elif any(w in q for w in ["remote", "laptop", "allowance"]):
        confidence = "medium"
    else:
        confidence = "low"
    print(f"  [mock_rag_node] confidence → {confidence}")
    return {"confidence": confidence}

def rag_answer_node(state: RoutedState) -> dict:
    print(f"  [rag_answer_node] Answering from HR documents.")
    return {"answer": "Answer from company HR documents. (HIGH confidence)"}

def llm_supplement_node(state: RoutedState) -> dict:
    print(f"  [llm_supplement_node] Supplementing with LLM knowledge.")
    return {"answer": "Partial answer from docs + LLM knowledge. (MEDIUM)"}

def web_search_node(state: RoutedState) -> dict:
    print(f"  [web_search_node] Searching the web via Tavily.")
    return {"answer": "Answer from web search with source links. (LOW)"}

# The routing FUNCTION — reads state, returns a node name string
def route_on_confidence(state: RoutedState) -> Literal["rag_answer", "llm_supplement", "web_search"]:
    """Decides which node to go to based on confidence."""
    mapping = {
        "high"  : "rag_answer",
        "medium": "llm_supplement",
        "low"   : "web_search",
    }
    return mapping[state["confidence"]]

print("\n" + "=" * 55)
print("CONDITIONAL EDGE GRAPH")
print("=" * 55)

g2 = StateGraph(RoutedState)
g2.add_node("rag_search",     mock_rag_node)
g2.add_node("rag_answer",     rag_answer_node)
g2.add_node("llm_supplement", llm_supplement_node)
g2.add_node("web_search",     web_search_node)

g2.add_edge(START, "rag_search")

# add_conditional_edges(source, routing_function, mapping)
g2.add_conditional_edges(
    "rag_search",
    route_on_confidence,
    {
        "rag_answer"    : "rag_answer",
        "llm_supplement": "llm_supplement",
        "web_search"    : "web_search",
    }
)
g2.add_edge("rag_answer",     END)
g2.add_edge("llm_supplement", END)
g2.add_edge("web_search",     END)

graph2 = g2.compile()

test_questions = [
    ("What is maternity leave in India?", "india"),
    ("Can I work from home every day?",   "india"),
    ("What is the visa sponsorship?",     "india"),
]

for question, country in test_questions:
    print(f"\nQ: {question}")
    result = graph2.invoke({
        "question"  : question,
        "country"   : country,
        "confidence": "",
        "answer"    : "",
    })
    print(f"   Route taken → {result['confidence']} confidence")
    print(f"   Answer: {result['answer']}")

# ============================================================
# KEY TAKEAWAY
# ============================================================
# LangGraph = StateGraph + Nodes + Edges
#
# StateGraph  → holds your TypedDict state class
# Nodes       → plain Python functions (state in, dict out)
# Edges       → fixed connections between nodes
# Conditional → routing function decides the next node
# .compile()  → validates graph, returns runnable
# .invoke()   → runs graph start to end, returns final state
#
# Everything from Phase 1-3 (LLM calls, RAG, confidence
# scoring) will become NODES in our full graph in Step 4.
# ============================================================
