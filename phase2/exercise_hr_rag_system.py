# ============================================================
# PHASE 2 - EXERCISE (Capstone): Complete HR RAG System
# ============================================================
# This ties everything from Phase 2 together:
#   ✅ Step 1: Document loading with metadata
#   ✅ Step 2: Chunking with RecursiveCharacterTextSplitter
#   ✅ Step 3: HuggingFace embeddings
#   ✅ Step 4: ChromaDB vector store
#   ✅ Step 5: Country-aware retrieval with confidence scoring
#   ✅ Step 6: Full RAG chain with fallback detection
#
# WHAT THIS BUILDS:
#   An interactive HR chatbot where:
#   1. User selects their country
#   2. Asks HR questions in a chat loop
#   3. System searches country docs first
#   4. If not found, tells the user (Phase 4 will auto-fallback)
#   5. Shows source document for transparency
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")
UPLOADS_PATH      = "data/uploads"
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 150
RETRIEVAL_K       = 4
SUPPORTED_FORMATS = {".txt", ".pdf", ".docx"}

# ============================================================
# PART 1: INGESTION PIPELINE
# ============================================================

def load_document(file_path: str, country: str) -> list:
    """Load any supported document type."""
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        print(f"  Skipping unsupported format: {path.name}")
        return []

    docs = loader.load()
    for doc in docs:
        doc.metadata.update({
            "country" : country.lower(),
            "filename": path.name,
            "doc_type": "hr_policy",
        })
    return docs


def build_vectorstore(embeddings) -> Chroma:
    """
    Scan the uploads folder, load all documents for every country,
    chunk them, and build/update the ChromaDB vector store.
    """
    uploads_dir = Path(UPLOADS_PATH)
    splitter    = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks  = []

    print("\nScanning uploads folder...")
    for country_dir in sorted(uploads_dir.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        files   = [f for f in country_dir.iterdir()
                   if f.suffix.lower() in SUPPORTED_FORMATS]

        if not files:
            print(f"  {country:10} → No documents found")
            continue

        for file_path in files:
            docs   = load_document(str(file_path), country)
            chunks = splitter.split_documents(docs)
            all_chunks += chunks
            print(f"  {country:10} → {file_path.name} ({len(chunks)} chunks)")

    if not all_chunks:
        raise ValueError("No documents found in uploads folder!")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    print("Embedding and storing in ChromaDB (may take a minute)...")

    vectorstore = Chroma.from_documents(
        documents         = all_chunks,
        embedding         = embeddings,
        persist_directory = VECTOR_STORE_PATH,
        collection_name   = "hr_policies",
    )
    print(f"Vector store ready: {vectorstore._collection.count()} vectors\n")
    return vectorstore


def load_existing_vectorstore(embeddings) -> Chroma:
    """Load vector store from disk (fast, no re-embedding)."""
    return Chroma(
        persist_directory  = VECTOR_STORE_PATH,
        embedding_function = embeddings,
        collection_name    = "hr_policies",
    )

# ============================================================
# PART 2: RAG ENGINE
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional HR assistant specialised in {country} employment law and company policies.

Answer the employee's question using ONLY the context provided below.
If the answer cannot be found in the context, respond with exactly: NOT_FOUND_IN_DOCS

Context from HR Policy Documents:
{context}

Guidelines:
- Cite specific numbers, durations, and percentages from the documents.
- Be clear, professional, and empathetic.
- If partial information is found, share what you know and flag the gap.
- Do not invent information not present in the context."""),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class HRAssistant:
    def __init__(self, vectorstore: Chroma, country: str):
        self.country     = country.lower()
        self.history     = []
        self.llm         = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
        self.retriever   = vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {
                "k"     : RETRIEVAL_K,
                "filter": {"country": self.country},
            },
        )
        self.chain = (
            {
                "context" : self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "country" : lambda _: self.country.upper(),
                "history" : lambda _: self.history[:-1] if self.history else [],
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> dict:
        """Ask a question and return answer + metadata."""
        self.history.append(HumanMessage(content=question))
        retrieved_docs = self.retriever.invoke(question)
        answer         = self.chain.invoke(question)
        self.history.append(AIMessage(content=answer))

        not_found = answer.strip() == "NOT_FOUND_IN_DOCS"
        return {
            "answer"       : answer,
            "not_found"    : not_found,
            "sources"      : [d.metadata["filename"] for d in retrieved_docs],
            "num_chunks"   : len(retrieved_docs),
        }

# ============================================================
# PART 3: INTERACTIVE CHAT LOOP
# ============================================================

def run_chat(vectorstore: Chroma):
    """Run the interactive HR chatbot."""
    print("\n" + "="*55)
    print("  HR KNOWLEDGE BASE — INTERACTIVE ASSISTANT")
    print("="*55)

    # Country selection
    available = sorted({
        Path(p).name
        for p in Path(UPLOADS_PATH).iterdir()
        if p.is_dir() and any(f.suffix in SUPPORTED_FORMATS
                               for f in p.iterdir())
    })

    print(f"\nAvailable countries: {', '.join(c.upper() for c in available)}")
    country = input("Enter your country: ").strip().lower()

    if country not in available:
        print(f"No documents found for '{country}'. Available: {available}")
        return

    assistant = HRAssistant(vectorstore, country)
    print(f"\nHR Assistant ready for {country.upper()} employees.")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = assistant.ask(question)
        print()

        if result["not_found"]:
            print("Assistant: I couldn't find this information in the uploaded")
            print("           HR documents. [In Phase 4, I'll automatically")
            print("           search the LLM and web to answer this.]\n")
        else:
            print(f"Assistant: {result['answer']}")
            print(f"\n[Sources: {', '.join(set(result['sources']))}]")
            print(f"[Retrieved from {result['num_chunks']} document chunk(s)]\n")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Initialising HR Knowledge Base...\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs_path = Path(VECTOR_STORE_PATH)
    if vs_path.exists() and any(vs_path.iterdir()):
        print("Found existing vector store — loading from disk (fast)...")
        vectorstore = load_existing_vectorstore(embeddings)
        print(f"Loaded {vectorstore._collection.count()} vectors.\n")
    else:
        print("No vector store found — building from documents...")
        vectorstore = build_vectorstore(embeddings)

    run_chat(vectorstore)
