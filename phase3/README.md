# Phase 3 — Country-Aware Multi-Tenant RAG

## 🎯 Goal
Add proper country isolation, confidence-based routing, document
management (add/delete per country), and a Streamlit UI for HR admins
and employees.

---

## ⚙️ Install new dependency

```bash
pip install -r requirements.txt   # adds streamlit + pandas
```

---

## 📚 Run in order

```bash
# Step 1: Multi-tenant isolation + safe incremental ingestion
python phase3/step1_multitenant_store.py

# Step 2: Confidence scoring — HIGH / MEDIUM / LOW routing
python phase3/step2_confidence_scoring.py

# Step 3: DocumentManager — add, list, delete docs
python phase3/step3_document_manager.py

# Step 4: Admin UI only (Streamlit)
streamlit run phase3/step4_admin_ui.py

# Step 5: Smart query engine with routing
python phase3/step5_query_engine.py

# Exercise: Full app (Admin + User chat, Streamlit)
streamlit run phase3/exercise_hr_app.py
```

> Steps 1–3 and 5 are terminal scripts (read output + learn).
> Steps 4 and exercise open a browser UI.

---

## 🧠 What Each Step Teaches

| File | Concept | Key idea |
|---|---|---|
| `step1` | Multi-tenant isolation | Metadata filter = country scoping |
| `step2` | Confidence scoring | distance → 0-1 score → HIGH/MEDIUM/LOW |
| `step3` | DocumentManager class | Safe add, list, delete per country |
| `step4` | Streamlit Admin UI | Upload + manage docs per country |
| `step5` | Smart query engine | Routing: rag_answer / llm_supplement / web_search |
| `exercise` | Full app | Admin panel + employee chatbot in one app |

---

## 💬 What to try in the exercise app

**Employee Chat tab:**
1. Select country → India
2. Ask: "How many days annual leave do I get?"
   → 🟢 HIGH — answered from docs
3. Ask: "What is the visa sponsorship policy?"
   → 🔴 LOW — not found (Phase 4 will web-search this)

**Admin tab:**
1. Switch to 🔧 HR Admin mode
2. Select a country
3. Upload a new HR document
4. Watch it appear in the Documents tab with chunk count
5. Delete it — verify it disappears

---

## 🔄 The Routing Logic (Preview of Phase 4)

```
User question
      ↓
smart_query(question, country)
      ↓
confidence score
      ↓
  ┌──────────────────────────────────┐
  │ HIGH (≥0.70)  → rag_answer       │ ← Phase 3 handles
  │ MEDIUM (≥0.45)→ llm_supplement   │ ← Phase 3 handles
  │ LOW (<0.45)   → web_search       │ ← Phase 4 adds this
  └──────────────────────────────────┘
```

Phase 4 (LangGraph) will wire these three routes as graph nodes,
adding a real web search for the LOW path.

---

## ✅ You're ready for Phase 4 when:
- [ ] step1: You understand why metadata filter = multi-tenancy
- [ ] step2: HIGH/MEDIUM/LOW routing makes sense to you
- [ ] step3: You can add and delete docs via DocumentManager
- [ ] step4: Admin UI lets you upload a new document
- [ ] step5: smart_query() returns the right tier for different questions
- [ ] exercise: Full app works — both admin and chat modes


## streamlit run exercise_hr_app.py --server.port 8520 --client.showErrorDetails=false --server.headless=true