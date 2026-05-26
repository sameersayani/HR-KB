# ============================================================
# PHASE 2 - STEP 1: Document Loading
# ============================================================
# What you'll learn:
#   - How to load TXT, PDF, and DOCX files using LangChain loaders
#   - What a "Document" object is (the core data unit in RAG)
#   - How metadata works (country, source, filename)
#   - Why metadata is critical for our country-based HR system
# ============================================================

from langchain_community.document_loaders import (
    TextLoader,          # for .txt files
    PyPDFLoader,         # for .pdf files
    Docx2txtLoader,      # for .docx files
)
from pathlib import Path

# -----------------------------------------------------------
# 1. What is a LangChain "Document"?
# -----------------------------------------------------------
# Every loader returns a list of Document objects.
# Each Document has two fields:
#
#   document.page_content  → the raw text content
#   document.metadata      → a dict of info about the document
#                            (source path, page number, etc.)
#
# This is the fundamental data unit that flows through the
# entire RAG pipeline — from loading to chunking to embedding
# to retrieval. Get comfortable with this object!

# -----------------------------------------------------------
# 2. Load a TXT file
# -----------------------------------------------------------
print("=" * 55)
print("LOADING TXT FILE")
print("=" * 55)

txt_path = "data/uploads/india/hr_policy_india.txt"
txt_loader = TextLoader(txt_path, encoding="utf-8")
txt_docs = txt_loader.load()

print(f"Number of documents loaded : {len(txt_docs)}")
print(f"Type of first doc          : {type(txt_docs[0])}")
print(f"Metadata                   : {txt_docs[0].metadata}")
print(f"Content preview (200 chars): {txt_docs[0].page_content[:200]}")

# -----------------------------------------------------------
# 3. Load a PDF file (each page = one Document)
# -----------------------------------------------------------
# NOTE: If you don't have a PDF yet, skip this section.
# We'll create a proper PDF ingestion pipeline in the exercise.

print("\n" + "=" * 55)
print("PDF LOADING (how it works)")
print("=" * 55)
print("""
PyPDFLoader splits a PDF into one Document per page.
Each Document's metadata includes the page number:

    loader = PyPDFLoader("path/to/file.pdf")
    docs   = loader.load()
    # docs[0].metadata = {'source': 'path/to/file.pdf', 'page': 0}
    # docs[1].metadata = {'source': 'path/to/file.pdf', 'page': 1}

This is useful because you can cite the exact page
in your answer: "According to page 3 of India HR Policy..."
""")

# -----------------------------------------------------------
# 4. Load a DOCX file
# -----------------------------------------------------------
print("=" * 55)
print("DOCX LOADING (how it works)")
print("=" * 55)
print("""
Docx2txtLoader loads the entire DOCX as a single Document.

    loader = Docx2txtLoader("path/to/file.docx")
    docs   = loader.load()
    # Returns 1 Document with all text combined
""")

# -----------------------------------------------------------
# 5. CRITICAL: Add custom metadata (country tagging)
# -----------------------------------------------------------
# By default, metadata only contains the file path.
# For our HR system, we MUST add "country" to metadata
# so we can filter by country when searching.

print("=" * 55)
print("ADDING COUNTRY METADATA (Key concept!)")
print("=" * 55)

# Reload and add country metadata
docs_with_metadata = txt_loader.load()

for doc in docs_with_metadata:
    # Add custom metadata fields
    doc.metadata["country"]  = "india"
    doc.metadata["doc_type"] = "hr_policy"
    doc.metadata["filename"] = Path(txt_path).name

print("Metadata after enrichment:")
for key, value in docs_with_metadata[0].metadata.items():
    print(f"  {key:12} → {value}")

# -----------------------------------------------------------
# 6. Universal loader function (handles all 3 formats)
# -----------------------------------------------------------
def load_document(file_path: str, country: str) -> list:
    """
    Load any supported document (TXT, PDF, DOCX) and
    enrich each Document with country metadata.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    docs = loader.load()

    # Enrich every page/chunk with country info
    for doc in docs:
        doc.metadata["country"]  = country.lower()
        doc.metadata["filename"] = path.name
        doc.metadata["doc_type"] = "hr_policy"

    print(f"Loaded: {path.name} ({len(docs)} document(s)) | Country: {country}")
    return docs

# -----------------------------------------------------------
# 7. Load all three country policies
# -----------------------------------------------------------
print("\n" + "=" * 55)
print("LOADING ALL COUNTRY DOCUMENTS")
print("=" * 55 + "\n")

all_docs = []
all_docs += load_document("data/uploads/india/hr_policy_india.txt", "india")
all_docs += load_document("data/uploads/usa/hr_policy_usa.txt",     "usa")
all_docs += load_document("data/uploads/uk/hr_policy_uk.txt",       "uk")

print(f"\nTotal documents loaded: {len(all_docs)}")
print("\nCountry distribution:")
from collections import Counter
countries = [doc.metadata["country"] for doc in all_docs]
for country, count in Counter(countries).items():
    print(f"  {country:10} → {count} document(s)")

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# - Every loader returns List[Document] (page_content + metadata)
# - PDF gives 1 Document per page; TXT/DOCX gives 1 Document total
# - Always add custom metadata (country) immediately after loading
# - The metadata is what allows filtering in Phase 3 (country RAG)
# -----------------------------------------------------------
