# ============================================================
# PHASE 5 - STEP 5: Multi-Agent Graph Assembly
# ============================================================
# What you'll learn:
#   - Wiring Supervisor → [parallel agents] → Summarizer
#   - Using Send() for dynamic fan-out based on supervisor decision
#   - Fan-in: all triggered agents converge to one Summarizer
#   - Running and inspecting the full multi-agent system
# ============================================================

import os, sys
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase5.step2_subagents import (
    MultiAgentState,
    rag_agent_node,
    web_agent_node,
    policy_agent_node,
)
from phase5.step3_supervisor import supervisor_node
from phase5.step4_summarizer import summarizer_node


# ============================================================
# DISPATCH FUNCTION — fan-out via Send()
# ============================================================
# This replaces a normal conditional edge's "return one string"
# with "return a LIST of Send() objects" — one per agent that
# the supervisor decided is needed. LangGraph runs all of them
# in PARALLEL and waits for every one to finish.

def dispatch_to_agents(state: MultiAgentState) -> list[Send]:
    """Fan-out to every agent the supervisor flagged as needed."""
    agent_input = {
        "question": state["question"],
        "country" : state["country"],
    }
    return [Send(agent_name, agent_input) for agent_name in state["needed_agents"]]


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_multiagent_graph():
    g = StateGraph(MultiAgentState)

    # ── Nodes ─────────────────────────────────────────────
    g.add_node("supervisor",    supervisor_node)
    g.add_node("rag_agent",     rag_agent_node)
    g.add_node("web_agent",     web_agent_node)
    g.add_node("policy_agent",  policy_agent_node)
    g.add_node("summarizer",    summarizer_node)

    # ── Entry point ───────────────────────────────────────
    g.add_edge(START, "supervisor")

    # ── Fan-out: supervisor → 1, 2, or 3 agents in parallel ─
    g.add_conditional_edges(
        "supervisor",
        dispatch_to_agents,
        ["rag_agent", "web_agent", "policy_agent"],
    )

    # ── Fan-in: every agent converges to summarizer ────────
    g.add_edge("rag_agent",    "summarizer")
    g.add_edge("web_agent",    "summarizer")
    g.add_edge("policy_agent", "summarizer")

    g.add_edge("summarizer", END)

    return g.compile()


def make_initial_state(question: str, country: str) -> MultiAgentState:
    return {
        "question"     : question,
        "country"      : country.lower(),
        "needed_agents": [],
        "agent_results": [],
        "final_answer" : "",
        "node_log"     : [],
    }


def display_result(result: MultiAgentState):
    agents_used = [r["agent"] for r in result["agent_results"]]
    print(f"\n{'='*65}")
    print(f"QUESTION       : {result['question']}")
    print(f"COUNTRY        : {result['country'].upper()}")
    print(f"AGENTS INVOKED : {agents_used}")
    print(f"NODE PATH      : {' → '.join(result['node_log'])}")
    print(f"{'='*65}")
    print(f"\n{result['final_answer']}")

    print(f"\n--- Individual agent confidence ---")
    for r in result["agent_results"]:
        print(f"  {r['agent']:14} conf={r['confidence']:.2f}  sources={r['sources']}")


# ============================================================
# RUN THE FULL MULTI-AGENT SYSTEM
# ============================================================
if __name__ == "__main__":
    print("Building multi-agent graph...")
    graph = build_multiagent_graph()
    print("Graph compiled.\n")

    test_cases = [
        # Should trigger rag_agent only
        ("How many days of annual leave do I get?", "india"),

        # Should trigger rag_agent + policy_agent (comparison question)
        ("How does our maternity leave compare to EU minimum standards?", "uk"),

        # Should trigger rag_agent + web_agent (needs current info)
        ("What is our notice period, and are there any 2026 changes to UK redundancy law?", "uk"),

        # Should trigger policy_agent only (no company-specific element)
        ("What is the general legal minimum notice period under German labour law?", "germany"),
    ]

    for question, country in test_cases:
        print(f"\n\nRunning: '{question}' [{country.upper()}]")
        print("-" * 65)
        state  = make_initial_state(question, country)
        result = graph.invoke(state)
        display_result(result)

    # ------------------------------------------------------------
    # SUMMARY TABLE — see which agents were triggered per question
    # ------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("SUMMARY — Agent Selection per Question")
    print("=" * 65)
    print(f"\n{'Question':<60} {'Agents Invoked'}")
    print("-" * 90)
    for question, country in test_cases:
        result = graph.invoke(make_initial_state(question, country))
        agents = [r["agent"] for r in result["agent_results"]]
        print(f"{question[:58]:<60} {agents}")
