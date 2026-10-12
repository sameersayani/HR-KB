# ============================================================
# PHASE 2 - STEP 2: Text Splitting (Chunking)
# ============================================================
# What you'll learn:
#   - Why you can't embed an entire document at once
#   - RecursiveCharacterTextSplitter — the go-to splitter
#   - chunk_size and chunk_overlap and how to tune them
#   - How metadata is preserved through splitting
#   - What good vs bad chunks look like
# ============================================================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# -----------------------------------------------------------
# WHY CHUNKING?
# -----------------------------------------------------------
# Problem 1: Embedding models have a token limit (~512 tokens).
#            You can't embed a 10-page document as one vector.
#
# Problem 2: Even if you could, a huge document vector is
#            "blurry" — it captures everything and nothing.
#            A chunk about "maternity leave" gives a sharp,
#            focused vector that matches relevant queries well.
#
# Solution: Split documents into small, overlapping chunks
#           before embedding. Each chunk becomes one vector.

# -----------------------------------------------------------
# 1. Load the India policy
# -----------------------------------------------------------
loader = TextLoader("data/uploads/india/hr_policy_india.txt", encoding="utf-8")
docs   = loader.load()

# Add metadata
docs[0].metadata["country"]  = "india"
docs[0].metadata["filename"] = "hr_policy_india.txt"

print(f"Original document length: {len(docs[0].page_content)} characters")
print(f"Approx tokens           : {len(docs[0].page_content) // 4}")

# -----------------------------------------------------------
# 2. RecursiveCharacterTextSplitter — how it works
# -----------------------------------------------------------
# It tries to split on these separators IN ORDER:
#   ["\n\n", "\n", " ", ""]
#
# It first tries to split on double newlines (paragraphs).
# If a paragraph is still too big, it splits on single newlines.
# If still too big, it splits on spaces (words).
# Last resort: splits on individual characters.
#
# This "recursive" approach keeps semantically related text
# together as much as possible.
#
# chunk_size    → max characters per chunk (aim for ~500-1000)
# chunk_overlap → characters shared between adjacent chunks
#                 (prevents cutting a sentence in half at boundary)

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 800,   # ~200 tokens — good for HR docs
    chunk_overlap = 150,   # overlap so context isn't lost at edges
    length_function = len,
)

chunks = splitter.split_documents(docs)

print(f"\nAfter splitting:")
print(f"  Number of chunks : {len(chunks)}")
print(f"  Avg chunk size   : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print(f"  Smallest chunk   : {min(len(c.page_content) for c in chunks)} chars")
print(f"  Largest chunk    : {max(len(c.page_content) for c in chunks)} chars")

# -----------------------------------------------------------
# 3. Inspect a few chunks
# -----------------------------------------------------------
print("\n" + "="*55)
print("CHUNK SAMPLES")
print("="*55)

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Length  : {len(chunk.page_content)} chars")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content :\n{chunk.page_content}")
    print()

# -----------------------------------------------------------
# 4. Metadata is preserved through splitting!
# -----------------------------------------------------------
print("="*55)
print("METADATA PRESERVATION CHECK")
print("="*55)
print(f"\nAll {len(chunks)} chunks have country metadata:")
all_have_country = all("country" in c.metadata for c in chunks)
print(f"  All chunks have 'country': {all_have_country}")
print(f"  Sample: chunk[5].metadata = {chunks[5].metadata}")

# -----------------------------------------------------------
# 5. Chunk size experiment — see the difference
# -----------------------------------------------------------
print("\n" + "="*55)
print("CHUNK SIZE COMPARISON")
print("="*55)

for size in [400, 800, 1500]:
    exp_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=100
    )
    exp_chunks = exp_splitter.split_documents(docs)
    print(f"  chunk_size={size:4d} → {len(exp_chunks):2d} chunks "
          f"(avg {sum(len(c.page_content) for c in exp_chunks)//len(exp_chunks)} chars each)")

# -----------------------------------------------------------
# 6. Split all 3 country docs
# -----------------------------------------------------------
print("\n" + "="*55)
print("SPLITTING ALL COUNTRY DOCUMENTS")
print("="*55 + "\n")

def load_and_split(file_path: str, country: str) -> list:
    path   = Path(file_path)
    loader = TextLoader(file_path, encoding="utf-8")
    docs   = loader.load()
    for doc in docs:
        doc.metadata["country"]  = country
        doc.metadata["filename"] = path.name
        doc.metadata["doc_type"] = "hr_policy"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"{country:6} | {path.name:30} | {len(chunks):2d} chunks")
    return chunks

all_chunks = []
all_chunks += load_and_split("data/uploads/india/hr_policy_india.txt", "india")
all_chunks += load_and_split("data/uploads/usa/hr_policy_usa.txt",     "usa")
all_chunks += load_and_split("data/uploads/uk/hr_policy_uk.txt",       "uk")

print(f"\nTotal chunks across all countries: {len(all_chunks)}")
print("These chunks are ready to be embedded in Step 3.")

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# chunk_size=800, chunk_overlap=150 is a solid starting point.
# Too small → chunks lose context, poor answers.
# Too large → chunks are "blurry", retrieval is less precise.
# Metadata (country, filename) is carried on EVERY chunk —
# this is what lets us filter by country in the vector store.
# -----------------------------------------------------------
