# Phase 2 — Document Ingestion & RAG

## 🎯 Goal
Load HR documents, embed them into a vector store, and answer
questions using Retrieval-Augmented Generation (RAG).

---

## ⚙️ Install new dependencies first

```bash
pip install -r requirements.txt
```

> First run of any embedding step will download the
> `all-MiniLM-L6-v2` model (~90MB). This is a one-time download.

---

## 📂 Sample documents included

Three sample HR policy files are provided in `data/uploads/`:

```
data/uploads/
├── india/hr_policy_india.txt   ← Covers India employment law
├── usa/hr_policy_usa.txt       ← Covers US FLSA, FMLA, PTO
└── uk/hr_policy_uk.txt         ← Covers UK Working Time Regs
```

You can add your own documents to any country folder and
they'll be automatically picked up by the ingestion pipeline.

---

## 📚 Run the steps in order

```bash
# Step 1: Understand Document objects + metadata
python phase2/step1_document_loading.py

# Step 2: Chunking — split docs into searchable pieces
python phase2/step2_text_splitting.py

# Step 3: Embeddings — text → vectors (downloads model)
python phase2/step3_embeddings.py

# Step 4: ChromaDB — store vectors, run first searches
python phase2/step4_vector_store.py

# Step 5: Retrieval — country-aware search + confidence
python phase2/step5_retrieval.py

# Step 6: Full RAG chain — retriever + LLM + source citation
python phase2/step6_rag_chain.py

# Exercise: Interactive HR chatbot (run this last!)
python phase2/exercise_hr_rag_system.py
```

> **Important:** Run step4 before step5 and step6.
> Step 4 creates the vector store that steps 5 and 6 load.

---

## 🧠 What Each Step Teaches

| File | Concept | Key Class |
|---|---|---|
| `step1` | Load docs + add metadata | `TextLoader`, `PyPDFLoader`, `Docx2txtLoader` |
| `step2` | Chunk into searchable pieces | `RecursiveCharacterTextSplitter` |
| `step3` | Text → vectors (embeddings) | `HuggingFaceEmbeddings` |
| `step4` | Store + search vectors | `Chroma` |
| `step5` | Retriever interface + scoring | `.as_retriever()`, `filter={}` |
| `step6` | RAG chain end-to-end | LCEL + `RunnablePassthrough` |
| `exercise` | Full interactive HR chatbot | All combined |

---

## 💬 Sample questions to try in the exercise

**India:**
- "How many days of annual leave do I get?"
- "What is the maternity leave policy?"
- "How is gratuity calculated?"
- "What is the probation period?"

**USA:**
- "Do I get overtime pay?"
- "How many weeks of parental leave do I get?"
- "What is the 401k matching policy?"

**UK:**
- "What is the statutory notice period?"
- "How does shared parental leave work?"
- "What is the redundancy pay calculation?"

**Should NOT be found (fallback trigger):**
- "What is the company stock option policy?"
- "How do I apply for visa sponsorship?"

---

## ✅ You're ready for Phase 3 when you can:
- [ ] Load TXT, PDF, and DOCX files with metadata
- [ ] Split documents into chunks and verify metadata is preserved
- [ ] Embed a sentence and understand what a vector is
- [ ] Run a similarity search against ChromaDB
- [ ] Filter results by country
- [ ] Get an RAG answer with the source file cited
- [ ] See "NOT_FOUND_IN_DOCS" trigger for unknown questions
