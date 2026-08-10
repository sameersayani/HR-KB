# HR-KB — HR Knowledge Base Solution

Brief: HR-KB is a multi-phase prototype that builds an HR knowledge-base and retrieval-powered assistant. It demonstrates document ingestion, embedding/vector storage, retrieval-augmented generation (RAG), multitenant document management, confidence scoring, and an experimental LangGraph integration for richer graph-driven workflows.

Goal: provide HR teams and employees with an easy-to-query knowledge base for country-specific policies, hiring templates, and procedural guidance. The project explores practical production patterns for scaling, multitenancy, and explainable retrieval for HR scenarios.

How this repo is organized
- **phase1/**: Foundations and prompt engineering; templates and the first structured-output experiment. See [phase1/exercise_job_description_generator.py](phase1/exercise_job_description_generator.py) and the step files for the incremental work.
- **phase2/**: Document ingestion, text-splitting, embeddings, and a vector store. Key artifacts: [phase2/step1_document_loading.py](phase2/step1_document_loading.py) → loaders, [phase2/step3_embeddings.py](phase2/step3_embeddings.py) → embeddings, and the Chroma DB under [phase2/vectorstore/hr_chroma_db/](phase2/vectorstore/hr_chroma_db/).
- **phase3/**: Multitenant store and query engine; adds a document manager, confidence scoring, and an admin UI prototype. See [phase3/exercise_hr_app.py](phase3/exercise_hr_app.py) and modular steps [phase3/step1_multitenant_store.py](phase3/step1_multitenant_store.py) → [phase3/step5_query_engine.py](phase3/step5_query_engine.py).
- **phase4/**: Experimental LangGraph integration and graph-based state for richer workflows and chaining. Main entry: [phase4/exercise_hr_langgraph_app.py](phase4/exercise_hr_langgraph_app.py) and supporting steps in [phase4/](phase4/).
- **phase5/**: Multi-agent HR assistant prototype with specialized subagents and a supervisor orchestrating task delegation, summarization, and workflow coordination. See [phase5/exercise_multiagent_app.py](phase5/exercise_multiagent_app.py) and the phase step files.

Data
- User-provided policy documents are placed under `phase2/data/uploads/` and `phase3/data/uploads/` by country (e.g., `usa`, `uk`, `india`, `mauritius`). Example files: [phase2/data/uploads/usa/hr_policy_usa.txt](phase2/data/uploads/usa/hr_policy_usa.txt).
- Vector stores are created under each phase's `vectorstore/hr_chroma_db/` directory; these contain the Chroma DB files used for retrieval experiments.

What each Phase does (summary)
- **Phase 1 — Prompting & Structured Output**: Rapid experiments that produce structured job descriptions and templated prompts. This phase is useful for understanding expected output formats and building deterministic prompt templates.
- **Phase 2 — Ingestion & RAG Pipeline**: Implements document loaders, text splitting, embedding generation, and a vector store for retrieval. It demonstrates sample RAG chains and how to persist vectors for reuse.
- **Phase 3 — Multitenancy & Query Engine**: Adds document management for multiple tenants/countries, confidence scoring for answers, and a simple admin UI to inspect data and query behavior. Focuses on UX for non-technical users and explainability.
- **Phase 4 — Graph Workflows (LangGraph)**: Experimental integration with LangGraph to model graph state, complex node orchestration, and web search integration. Explores more advanced orchestration patterns for HR workflows.
- **Phase 5 — Multi-Agent System**: Builds a prototype of a coordinated multi-agent HR assistant with subagents for domain-specific tasks, a supervisor for delegation and convergence, and a summarizer for cohesive user-facing responses.

Primary problems this project addresses
- Centralized HR knowledge: consolidating policies and templates across countries into a searchable knowledge base.
- Faster employee self-service: allow employees to ask natural-language questions and receive concise, sourced answers.
- Explainable retrieval: combine RAG with confidence scoring and provenance so HR staff can trust and verify answers.
- Multitenancy & scale: demonstrate approaches to keep tenant data separate while using shared retrieval infrastructure.

Quick start (developer)
1. Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run a phase script (example — Phase 2 ingestion):

```powershell
python phase2/step1_document_loading.py
```

3. Launch the Phase 4 Streamlit demo (if dependencies installed):

```powershell
cd phase4
streamlit run exercise_hr_langgraph_app.py --server.port 8520
```

Notes and next steps
- The repo is meant as a research/prototype codebase: it prioritizes clarity and experiments over production hardening.
- To progress toward production, next steps include: secure tenant isolation, CI for vector builds, data governance for source attribution, and UI polish.

Contributing
- Contributions welcome: open an issue or a PR explaining the change, which phase it targets, and any data/fixtures needed to reproduce behavior.

License
- This prototype does not include a license file.

----

