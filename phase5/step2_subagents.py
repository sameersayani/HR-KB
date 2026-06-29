# ============================================================
# PHASE 5 - STEP 2: Specialist Sub-Agents
# ============================================================
# What you'll learn:
#   - Building 3 independent specialist agents
#   - RAG Agent    → answers from company HR documents
#   - Web Agent    → answers from live Tavily search
#   - Policy Agent → answers from general employment law
#                    reasoning (no docs, no web — pure LLM)
#   - Each agent returns the SAME shape of result dict so the
#     Summarizer (step 4) can merge them uniformly
# ============================================================

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import operator

load_dotenv()

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")

# ============================================================
# SHARED STATE (used across all of Phase 5)
# ============================================================

class AgentResult(TypedDict):
    agent      : str    # "rag_agent" / "web_agent" / "policy_agent"
    answer     : str
    sources    : list[str]
    confidence : float  # agent's own confidence in its answer (0-1)

class MultiAgentState(TypedDict):
    question       : str
    country        : str
    needed_agents  : list[str]
    agent_results  : Annotated[list[AgentResult], operator.add]
    final_answer   : str
    node_log       : Annotated[list[str], operator.add]

# ============================================================
# SHARED RESOURCES
# ============================================================
print("Loading shared resources for agents...")

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
# AGENT 1: RAG Agent — answers from company HR documents
# ============================================================

RAG_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the RAG Agent — an HR specialist who ONLY uses the
company HR documents provided below to answer. Be specific and cite exact
numbers/durations. If the context doesn't answer the question, say so plainly.

Company HR Documents ({country}):
{context}"""),
    ("human", "{question}"),
])

def rag_agent_node(state: dict) -> dict:
    """Specialist agent: answers strictly from company HR documents."""
    question = state["question"]
    country  = state["country"]

    results = vectorstore.similarity_search_with_score(
        query=question, k=4, filter={"country": country.lower()}
    )

    if not results:
        return {
            "agent_results": [{
                "agent": "rag_agent", "answer": "No relevant company documents found.",
                "sources": [], "confidence": 0.0,
            }],
            "node_log": ["rag_agent → no docs found"],
        }

    docs       = [r[0] for r in results]
    best_dist  = min(r[1] for r in results)
    confidence = round(1 / (1 + best_dist), 3)
    context    = "\n\n---\n\n".join(d.page_content for d in docs)
    sources    = list({d.metadata.get("filename", "unknown") for d in docs})

    chain  = RAG_AGENT_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question, "country": country.upper()})

    print(f"  [rag_agent] confidence={confidence:.2f} sources={sources}")
    return {
        "agent_results": [{
            "agent": "rag_agent", "answer": answer,
            "sources": sources, "confidence": confidence,
        }],
        "node_log": [f"rag_agent → conf={confidence:.2f}"],
    }


# ============================================================
# AGENT 2: Web Agent — answers from live web search
# ============================================================

WEB_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Web Agent — an HR specialist who answers using
ONLY the live web search results below. Always cite source URLs clearly.

Web Search Results:
{context}"""),
    ("human", "{question}"),
])

def web_agent_node(state: dict) -> dict:
    """Specialist agent: answers from live Tavily web search."""
    question = state["question"]
    country  = state["country"]

    try:
        from langchain_tavily import TavilySearch
        tool     = TavilySearch(max_results=3)
        query    = f"{question} {country} employment law HR policy"
        response = tool.invoke({"query": query})

        # TavilySearch returns a dict — extract the actual results list
        results = response.get("results", []) if isinstance(response, dict) else response

        if not results:
            raise ValueError("No web results returned")

        context_parts = [f"Source: {r.get('url','unknown')}\n{r.get('content','')}" for r in results]
        sources        = [r.get("url", "unknown") for r in results]
        context        = "\n\n---\n\n".join(context_parts)

        chain  = WEB_AGENT_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        print(f"  [web_agent] found {len(results)} results")
        return {
            "agent_results": [{
                "agent": "web_agent", "answer": answer,
                "sources": sources, "confidence": 0.6,  # web search = moderate trust
            }],
            "node_log": [f"web_agent → {len(results)} results"],
        }

    except Exception as e:
        print(f"  [web_agent] unavailable: {e}")
        return {
            "agent_results": [{
                "agent": "web_agent",
                "answer": "Web search unavailable.",
                "sources": [], "confidence": 0.0,
            }],
            "node_log": ["web_agent → unavailable"],
        }


# ============================================================
# AGENT 3: Policy Agent — pure LLM employment-law reasoning
# ============================================================
# No documents, no web search — this agent relies on the LLM's
# own training knowledge of general employment law to add
# context, comparisons, or explanations the other two can't.

POLICY_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Policy Agent — an employment law expert for {country}.
Answer using your general knowledge of {country} labour law, statutory minimums,
and common employment practices. Do NOT reference any specific company documents
(you don't have access to them). Be clear that this is general legal knowledge,
not company-specific policy. Keep it concise and practical."""),
    ("human", "{question}"),
])

def policy_agent_node(state: dict) -> dict:
    """Specialist agent: pure LLM reasoning on general employment law."""
    question = state["question"]
    country  = state["country"]

    chain  = POLICY_AGENT_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "country": country.upper()})

    print(f"  [policy_agent] answered from general legal knowledge")
    return {
        "agent_results": [{
            "agent": "policy_agent", "answer": answer,
            "sources": ["General employment law knowledge"], "confidence": 0.5,
        }],
        "node_log": ["policy_agent → general knowledge"],
    }


# ============================================================
# QUICK TEST: run each agent independently
# ============================================================
if __name__ == "__main__":
    test_state = {
        "question": "How many days of maternity leave am I entitled to, and how does it compare to EU minimum standards?",
        "country" : "india",
    }

    print("=" * 60)
    print("TESTING EACH AGENT INDEPENDENTLY")
    print("=" * 60)

    print("\n1. RAG Agent:")
    r1 = rag_agent_node(test_state)
    print(f"   {r1['agent_results'][0]['answer'][:200]}...")

    print("\n2. Policy Agent:")
    r2 = policy_agent_node(test_state)
    print(f"   {r2['agent_results'][0]['answer'][:200]}...")

    print("\n3. Web Agent:")
    r3 = web_agent_node(test_state)
    print(f"   {r3['agent_results'][0]['answer'][:200]}...")

    print("\n" + "=" * 60)
    print("All three agents answered the SAME question independently.")
    print("In step 3, a Supervisor will decide which ones to actually call.")
    print("In step 4, a Summarizer will merge whichever ones ran.")
    print("=" * 60)
