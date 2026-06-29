Phase 5 — Multi-Agent
Right now one graph handles everything. Phase 5 breaks it into specialized agents:
User Question
      ↓
Supervisor Agent          ← decides which expert to call
   ↓         ↓        ↓
RAG Agent  Web Agent  Policy Agent
(HR docs)  (Tavily)   (legal rules)
      ↓         ↓        ↓
   Summarizer Agent       ← combines all answers
          ↓
   Final Answer + All Sources