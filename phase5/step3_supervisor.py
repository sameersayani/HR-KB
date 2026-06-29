# ============================================================
# PHASE 5 - STEP 3: Supervisor Agent
# ============================================================
# What you'll learn:
#   - Using the LLM itself to decide which specialist agents
#     are needed for a given question (structured output)
#   - Multi-label classification (a question can need 1, 2, or 3 agents)
#   - Why this is smarter than the fixed if/else routing in Phase 4
# ============================================================

import os, sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)

# ------------------------------------------------------------
# WHY AN LLM-BASED SUPERVISOR (not rule-based if/else)?
# ------------------------------------------------------------
# Phase 4's routing used a numeric confidence score from ONE
# retrieval call — simple, but binary in spirit (pick one path).
#
# Phase 5's Supervisor reads the QUESTION ITSELF and reasons
# about what KIND of question it is:
#   - Purely company-specific?            → RAG Agent only
#   - Needs legal context/comparison?     → + Policy Agent
#   - Needs current/external information? → + Web Agent
#
# This handles compound questions naturally, e.g.:
#   "What's our maternity leave and how does UK statutory
#    leave compare, and are there any 2026 law changes?"
#   → needs ALL THREE agents.

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Supervisor Agent routing HR questions to specialist agents.

Available agents:
- rag_agent    : Searches the COMPANY's own HR policy documents. Use for ANY
                 question that mentions "our", "company", "we", or asks about
                 a specific benefit/leave/policy that the company would define.
- policy_agent : Pure employment-law reasoning (no documents). Use when the
                 question asks to COMPARE to legal minimums, explain general
                 labour law, or interpret legal concepts/compliance.
- web_agent    : Live web search. Use when the question needs CURRENT, recent,
                 or external information (visa rules, "latest", specific years,
                 news, or anything outside company docs).

CRITICAL RULE: Comparison questions ("compare", "vs", "how does ours compare",
"is our X compliant with Y") ALWAYS need BOTH rag_agent AND policy_agent —
you need the company's actual policy (rag_agent) AND the legal benchmark
(policy_agent) to make a comparison. Never answer a comparison with only one.

A question can need ONE, TWO, or ALL THREE agents. Always include at least one.

EXAMPLES:
Q: "How many days of annual leave do I get?"
A: {{"agents": ["rag_agent"], "reasoning": "purely company-specific, no comparison"}}

Q: "How does our maternity leave compare to EU minimum standards?"
A: {{"agents": ["rag_agent", "policy_agent"], "reasoning": "needs company policy AND legal benchmark to compare"}}

Q: "Is our probation period legally compliant in Germany?"
A: {{"agents": ["rag_agent", "policy_agent"], "reasoning": "compliance check needs company policy AND legal standard"}}

Q: "What is the general legal minimum notice period under German labour law?"
A: {{"agents": ["policy_agent"], "reasoning": "purely general law, no company reference"}}

Q: "What is our notice period, and are there any 2026 changes to redundancy law?"
A: {{"agents": ["rag_agent", "web_agent"], "reasoning": "company policy plus current/recent law changes"}}

Respond ONLY with JSON in the exact format shown above."""),
    ("human", "Question: {question}\nCountry: {country}"),
])

# Keywords that strongly imply a comparison/compliance question —
# these REQUIRE both rag_agent and policy_agent as a safety net,
# since smaller LLMs sometimes collapse comparisons to one label.
COMPARISON_KEYWORDS = [
    "compare", "compared", "comparison", "vs", "versus",
    "compliant", "compliance", "legally", "minimum standard",
    "benchmark", "how does our", "is our",
]

def supervisor_node(state: dict) -> dict:
    """LLM-based supervisor — decides which specialist agents to invoke."""
    chain  = SUPERVISOR_PROMPT | llm | JsonOutputParser()
    result = chain.invoke({
        "question": state["question"],
        "country" : state["country"],
    })

    agents = result.get("agents", ["rag_agent"])
    valid_agents = {"rag_agent", "policy_agent", "web_agent"}
    agents = [a for a in agents if a in valid_agents]
    if not agents:
        agents = ["rag_agent"]

    # ── Safety net: force both agents for comparison-style questions ──
    question_lower = state["question"].lower()
    if any(kw in question_lower for kw in COMPARISON_KEYWORDS):
        if "rag_agent" not in agents:
            agents.append("rag_agent")
        if "policy_agent" not in agents:
            agents.append("policy_agent")

    agents = list(dict.fromkeys(agents))  # dedupe, preserve order

    print(f"  [supervisor] agents={agents}")
    print(f"  [supervisor] reasoning: {result.get('reasoning', 'n/a')}")

    return {
        "needed_agents": agents,
        "node_log": [f"supervisor → {agents} ({result.get('reasoning', '')})"],
    }


# ============================================================
# TEST: Watch the supervisor reason about different questions
# ============================================================
if __name__ == "__main__":
    test_questions = [
        ("How many days of annual leave do I get?",                          "india"),
        ("Compare our maternity leave to the EU statutory minimum",          "uk"),
        ("What's the latest H1B visa rule and our remote work policy?",      "usa"),
        ("Is our probation period legally compliant in Germany?",           "germany"),
        ("What is the company notice period?",                              "uk"),
    ]

    print("=" * 65)
    print("SUPERVISOR ROUTING DECISIONS")
    print("=" * 65)

    for question, country in test_questions:
        print(f"\nQ: {question}")
        print(f"Country: {country.upper()}")
        result = supervisor_node({"question": question, "country": country})
        print(f"→ Agents selected: {result['needed_agents']}")

    print("\n" + "=" * 65)
    print("KEY TAKEAWAY")
    print("=" * 65)
    print("""
The Supervisor reads the QUESTION TEXT and decides agent(s) needed
using the LLM's own reasoning — not a fixed numeric threshold.
This naturally handles compound questions that need multiple
specialists, which Phase 4's single-path routing could not do.
    """)