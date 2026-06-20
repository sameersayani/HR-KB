# Phase 4 — LangGraph Orchestration

## 🎯 Goal
Wire everything from Phases 1–3 into a LangGraph state machine
with intelligent routing: HR docs → LLM supplement → web search.

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt    # adds langgraph, langchain-tavily
```

### 2. Get a free Tavily API key (for web search)
1. Go to → https://app.tavily.com
2. Sign up free (no credit card)
3. Copy your API key
4. Add to `.env`:
```
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

> Without Tavily, LOW confidence queries will show a fallback
> message instead of live web results. Everything else works fine.

---

## 📚 Run in order

```bash
# Step 1: LangGraph basics — StateGraph, nodes, edges
python phase4/step1_langgraph_basics.py

# Step 2: State design — HRGraphState TypedDict walkthrough
python phase4/step2_graph_state.py

# Step 3: All nodes — each node function tested independently
python phase4/step3_nodes.py

# Step 4: Graph assembly — compile + run full graph
python phase4/step4_graph_assembly.py

# Step 5: Web search — Tavily setup + LOW confidence path test
python phase4/step5_web_search.py

# Exercise: Full LangGraph-powered Streamlit app
streamlit run phase4/exercise_hr_langgraph_app.py
```

---

## 🧠 The LangGraph Flow

```
START
  ↓
extract_intent_node      → validates country from question
  ↓
rag_search_node          → searches ChromaDB, computes confidence
  ↓
[conditional_edge]       → routes based on confidence tier
  ↓              ↓                  ↓
rag_answer    llm_supplement    web_search
(HIGH ≥0.40)  (MEDIUM ≥0.25)   (LOW <0.25)
  ↓              ↓                  ↓
format_response_node     → adds badge + source attribution
  ↓
END
```

---

## 🔑 Key LangGraph Concepts Learned

| Concept | Where |
|---|---|
| `StateGraph` + `TypedDict` state | step1, step2 |
| Nodes as plain Python functions | step1, step3 |
| Fixed edges (`add_edge`) | step1, step4 |
| Conditional edges (`add_conditional_edges`) | step1, step4 |
| `Annotated[list, operator.add]` for appending | step2 |
| `.compile()` + `.invoke()` | step1, step4 |
| `.stream()` for live node updates | step4, exercise |
| Web search node with Tavily | step5, step3 |

---

## 💬 Try These in the Exercise App

**🟢 HIGH — from HR documents:**
- "How many days annual leave in India?"
- "What is the 401k matching policy in USA?"
- "Explain UK redundancy pay calculation"

**🟡 MEDIUM — docs + LLM:**
- "Can I carry forward unused leave?"
- "What happens if I don't serve my notice period?"

**🔴 LOW — web search (needs Tavily key):**
- "What is the H1B visa cap this year?"
- "Latest IR35 rules for UK contractors"
- "Germany work permit process for Indian citizens"

---

## ✅ You're ready for Phase 5 when:
- [ ] step1: You can build a basic StateGraph with conditional edges
- [ ] step2: You understand every field in HRGraphState and why it exists
- [ ] step3: You can write a LangGraph node function from scratch
- [ ] step4: Full graph runs all 3 routing paths correctly
- [ ] step5: LOW confidence questions return web search results with URLs
- [ ] exercise: Full app works — chat shows which node path was taken
