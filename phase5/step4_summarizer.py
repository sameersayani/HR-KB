# ============================================================
# PHASE 5 - STEP 4: Summarizer Agent
# ============================================================
# What you'll learn:
#   - Merging 1-3 agent outputs into ONE coherent final answer
#   - Weighting by each agent's self-reported confidence
#   - Deduplicating and combining sources from multiple agents
#   - Why this node is REQUIRED whenever more than one agent runs
# ============================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

# ------------------------------------------------------------
# WHY A SEPARATE SUMMARIZER (vs just picking the best agent)?
# ------------------------------------------------------------
# If only 1 agent ran, summarizing is trivial — pass it through.
# If 2-3 agents ran, their answers may:
#   - Overlap (both mention the same numbers)
#   - Complement (one gives company policy, other gives legal context)
#   - Conflict (rare, but should be flagged)
#
# The Summarizer's job is to weave these into ONE answer that
# reads naturally — not just concatenate them with line breaks.

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Summarizer Agent. You receive answers from one or
more specialist agents who each researched the same HR question independently.
Combine their findings into ONE clear, well-organized answer for the employee.

Guidelines:
- If only one agent answered, simply present their answer clearly.
- If multiple agents answered, weave them together logically — e.g. start with
  company policy (rag_agent), then add legal context (policy_agent), then
  current external info (web_agent) if relevant.
- Clearly attribute which parts come from company documents vs general
  knowledge vs web search, so the employee knows how reliable each part is.
- Do not repeat the same fact twice. Do not say "Agent 1 said... Agent 2 said...".
  Write as a single unified HR assistant voice.
- If agents conflict, note the discrepancy briefly and recommend confirming with HR.

Specialist agent findings:
{agent_findings}"""),
    ("human", "Original question: {question}"),
])


def format_agent_findings(agent_results: list[dict]) -> str:
    """Build a readable block of all agent outputs for the summarizer prompt."""
    parts = []
    label_map = {
        "rag_agent"   : "RAG Agent (Company HR Documents)",
        "policy_agent": "Policy Agent (General Employment Law)",
        "web_agent"   : "Web Agent (Live Web Search)",
    }
    for r in agent_results:
        label = label_map.get(r["agent"], r["agent"])
        parts.append(
            f"--- {label} (confidence: {r['confidence']:.2f}) ---\n{r['answer']}"
        )
    return "\n\n".join(parts)


def summarizer_node(state: dict) -> dict:
    """Merge all agent_results into one final coherent answer."""
    agent_results = state["agent_results"]

    if not agent_results:
        return {
            "final_answer": "No agents were able to answer this question.",
            "node_log": ["summarizer → no agent results"],
        }

    # Single-agent shortcut: still pass through the summarizer for
    # consistent formatting, but it's a much cheaper/simpler call.
    findings = format_agent_findings(agent_results)

    chain  = SUMMARIZER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "agent_findings": findings,
        "question"      : state["question"],
    })

    # Merge + dedupe sources across all agents that ran
    all_sources = []
    for r in agent_results:
        for s in r.get("sources", []):
            if s and s not in all_sources:
                all_sources.append(s)

    agents_used = [r["agent"] for r in agent_results]
    print(f"  [summarizer] merged {len(agent_results)} agent(s): {agents_used}")

    return {
        "final_answer": answer,
        "node_log": [f"summarizer → merged {agents_used}"],
    }


# ============================================================
# TEST: Summarize different combinations of agent outputs
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Single agent result (rag_agent only)")
    print("=" * 60)

    single_state = {
        "question": "How many days of annual leave do I get in India?",
        "agent_results": [
            {
                "agent": "rag_agent",
                "answer": "Employees in India get 21 days of earned leave per year, "
                          "accruing at 1.75 days/month, with up to 30 days carry-forward.",
                "sources": ["hr_policy_india.txt"],
                "confidence": 0.85,
            }
        ],
    }
    result = summarizer_node(single_state)
    print(f"\nFinal answer:\n{result['final_answer']}")

    print("\n\n" + "=" * 60)
    print("TEST 2: Two agents (rag_agent + policy_agent)")
    print("=" * 60)

    multi_state = {
        "question": "How does our maternity leave compare to EU minimum standards?",
        "agent_results": [
            {
                "agent": "rag_agent",
                "answer": "Company policy (India) provides 26 weeks of paid maternity leave "
                          "for the first two children.",
                "sources": ["hr_policy_india.txt"],
                "confidence": 0.80,
            },
            {
                "agent": "policy_agent",
                "answer": "EU minimum standards require at least 14 weeks of maternity leave "
                          "under the EU Pregnant Workers Directive, though many EU countries "
                          "exceed this minimum significantly.",
                "sources": ["General employment law knowledge"],
                "confidence": 0.55,
            },
        ],
    }
    result = summarizer_node(multi_state)
    print(f"\nFinal answer:\n{result['final_answer']}")

    print("\n\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
The Summarizer reads ALL agent outputs + their confidence scores
and produces ONE unified answer — properly attributing which
parts are company policy vs general legal knowledge vs web search.
This is what makes multi-agent output feel coherent rather than
like three separate paragraphs stitched together.
    """)
