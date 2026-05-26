# ============================================================
# PHASE 2 - STEP 6: Full RAG Chain
# ============================================================
# What you'll learn:
#   - Combining retriever + LLM into a complete RAG pipeline
#   - create_retrieval_chain — LangChain's RAG shortcut
#   - How context is injected into the prompt
#   - Citing sources (which document answered the question)
#   - The confidence-based fallback pattern
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

VECTOR_STORE_PATH = "vectorstore/hr_chroma_db"

# -----------------------------------------------------------
# 1. Setup LLM + embeddings + vector store
# -----------------------------------------------------------
print("Initialising components...")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

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
print("All components ready.\n")

# -----------------------------------------------------------
# 2. The RAG prompt — how context is injected
# -----------------------------------------------------------
# {context} → filled with retrieved document chunks
# {question} → the user's question
# The LLM uses context to answer; if not in context, says so.

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful HR assistant for {country} employees.
    
Answer the question using ONLY the information provided in the context below.
If the answer is not in the context, respond with exactly:
"NOT_FOUND_IN_DOCS"

Context:
{context}

Rules:
- Be specific and cite exact numbers/durations from the context.
- Keep the answer clear and professional.
- Do not make up information not present in the context."""),
    ("human", "{question}"),
])

# -----------------------------------------------------------
# 3. Build the RAG chain manually with LCEL
# -----------------------------------------------------------
# We build it manually (not create_retrieval_chain) so we have
# full control over the context formatting and country filter.

def format_docs(docs):
    """Join retrieved chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(country: str):
    """Build a country-specific RAG chain."""
    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": 4, "filter": {"country": country.lower()}},
    )

    chain = (
        {
            "context" : retriever | format_docs,
            "question": RunnablePassthrough(),
            "country" : lambda _: country.upper(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# -----------------------------------------------------------
# 4. Full RAG answer with source attribution
# -----------------------------------------------------------
def ask_hr(question: str, country: str):
    """
    Ask an HR question and get an answer with sources.
    Falls back to LLM general knowledge if not found in docs.
    """
    chain, retriever = build_rag_chain(country)

    # Get retrieved docs for citation
    retrieved_docs = retriever.invoke(question)

    # Get the answer
    answer = chain.invoke(question)

    print(f"\n{'='*55}")
    print(f"Q: {question}")
    print(f"Country: {country.upper()}")
    print(f"{'='*55}")

    if answer.strip() == "NOT_FOUND_IN_DOCS":
        print("Source : NOT FOUND in HR documents")
        print("Answer : This topic is not covered in the uploaded HR documents.")
        print("         → System will fall back to LLM/web search (Phase 4)")
    else:
        print(f"Source : HR Policy Document ({country.upper()})")
        print(f"\nAnswer :\n{answer}")

        # Show which document chunks were used
        print(f"\nRetrieved from {len(retrieved_docs)} document chunk(s):")
        for i, doc in enumerate(retrieved_docs[:2]):
            print(f"  [{i+1}] {doc.metadata['filename']} — "
                  f"{doc.page_content[:80]}...")

    return answer

# -----------------------------------------------------------
# 5. Test with various queries
# -----------------------------------------------------------

# Questions that SHOULD be found in docs
print("\n" + "="*55)
print("QUESTIONS FOUND IN HR DOCUMENTS")
print("="*55)

ask_hr("How many days of annual leave am I entitled to?", "india")
ask_hr("What is the maternity leave policy?",             "usa")
ask_hr("What is the notice period after confirmation?",   "uk")

# Questions that should NOT be found (triggers fallback)
print("\n" + "="*55)
print("QUESTIONS NOT IN HR DOCUMENTS (fallback trigger)")
print("="*55)

ask_hr("What is the company's stock options policy?",     "india")
ask_hr("How do I apply for a visa sponsorship?",          "usa")

# -----------------------------------------------------------
# 6. Side-by-side country comparison
# -----------------------------------------------------------
print("\n" + "="*55)
print("SAME QUESTION — DIFFERENT COUNTRIES")
print("="*55)

question = "What is the probation period for new employees?"
print(f"\nQuestion: '{question}'\n")

for country in ["india", "usa", "uk"]:
    chain, _ = build_rag_chain(country)
    answer   = chain.invoke(question)
    print(f"{country.upper():5}: {answer[:200]}...")
    print()

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# Full RAG pipeline:
#   question → retriever (get docs) → format context
#            → prompt (inject context) → LLM → answer
#
# "NOT_FOUND_IN_DOCS" sentinel tells us when to fall back.
# Country filter ensures India employees get India answers.
# In Phase 4 (LangGraph), the fallback logic becomes a graph
# node that routes to LLM knowledge or web search.
# -----------------------------------------------------------
