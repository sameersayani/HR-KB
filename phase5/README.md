# Phase 5 — Multi-Agent System

## 🎯 Goal
Replace single-path routing (Phase 4) with a true multi-agent
system: a Supervisor that can call 1, 2, or 3 specialist agents
in **parallel**, and a Summarizer that merges their outputs.

---

## 🧩 The Architecture

```
START
  → supervisor_node          (LLM decides which agents are needed)
  → [Send fan-out: 1-3 of]
       rag_agent      (company HR documents — ChromaDB)
       policy_agent   (general employment law — pure LLM reasoning)
       web_agent      (live web search — Tavily)
  → summarizer_node          (merges all agent outputs into ONE answer)
END
```

**Key difference from Phase 4:** Phase 4 picked exactly ONE path based
on a confidence number. Phase 5's Supervisor reads the question and can
trigger MULTIPLE agents in parallel — e.g. a comparison question
("how does our policy compare to EU law?") runs both `rag_agent` AND
`policy_agent`, and the Summarizer weaves their answers together.

---

## ⚙️ Setup

No new packages needed beyond Phase 4 (`langgraph`, `langchain-tavily`
already cover this). Just make sure `vectorstore/` exists in `phase5/`
(copy it from an earlier phase, same as you did for Phase 4):

```bash
# Windows
xcopy phase4\vectorstore phase5\vectorstore /E /I /H /Y

# Mac/Linux
cp -r phase4/vectorstore phase5/vectorstore
```

---

## 📚 Run in order

```bash
# Step 1: Multi-agent concepts — Send() fan-out/fan-in demo
python phase5/step1_multiagent_concepts.py

# Step 2: The 3 specialist agents, tested independently
python phase5/step2_subagents.py

# Step 3: Supervisor — LLM decides which agents are needed
python phase5/step3_supervisor.py

# Step 4: Summarizer — merges 1-3 agent outputs into one answer
python phase5/step4_summarizer.py

# Step 5: Full graph assembled — Supervisor → agents → Summarizer
python phase5/step5_graph_assembly.py

# Exercise: Full multi-agent Streamlit app
streamlit run phase5/exercise_multiagent_app.py
```

> Run all commands from the `hr-knowledgebase/` root folder.

---

## 🧠 What Each Step Teaches

| File | Concept |
|---|---|
| `step1` | `Send()` for dynamic parallel fan-out, fan-in pattern |
| `step2` | Three independent specialist agents (RAG / Web / Policy) |
| `step3` | LLM-based Supervisor — structured multi-label routing |
| `step4` | Summarizer — merges + attributes multiple agent outputs |
| `step5` | Full graph: Supervisor → parallel agents → Summarizer |
| `exercise` | Streamlit app showing live agent selection + individual answers |

---

## 💬 Try These in the Exercise App

**Triggers `rag_agent` only:**
- "How many days of annual leave do I get?"
- "What is the notice period?"

**Triggers `rag_agent` + `policy_agent` (comparison):**
- "How does our maternity leave compare to EU minimum standards?"
- "Is our probation period legally compliant?"

**Triggers `rag_agent` + `web_agent` (needs current info):**
- "What is our notice period, and are there any 2026 changes to redundancy law?"

**Triggers `policy_agent` only (no company-specific element):**
- "What is the general legal minimum notice period under German labour law?"

Open the "🔍 Individual agent answers" expander after each response to see
exactly what each agent said independently, before the Summarizer merged them.

---

## ✅ You're ready for Phase 6 when:
- [ ] step1: You understand `Send()` and why fan-in needs `Annotated[list, operator.add]`
- [ ] step2: Each agent runs independently and returns the same result shape
- [ ] step3: Supervisor correctly picks multiple agents for compound questions
- [ ] step4: Summarizer produces ONE coherent answer, not 3 stitched paragraphs
- [ ] step5: Full graph runs end-to-end with parallel agent execution
- [ ] exercise: App shows different agent combinations for different questions
