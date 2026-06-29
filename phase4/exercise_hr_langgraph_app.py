# ============================================================
# PHASE 4 - EXERCISE: Complete HR App with LangGraph
# ============================================================
# Run with: streamlit run phase4/exercise_hr_langgraph_app.py
#
# This is the complete Phase 4 application combining:
#   ✅ Phase 1: LLM via LangChain (Groq)
#   ✅ Phase 2: RAG with ChromaDB + HuggingFace embeddings
#   ✅ Phase 3: Multi-tenant store, confidence scoring, admin UI
#   ✅ Phase 4: LangGraph orchestration with full routing graph
#
# WHAT'S NEW vs Phase 3 exercise:
#   - Every query now runs through the LangGraph graph
#   - Real web search via Tavily for LOW confidence queries
#   - Node execution log visible in chat (which path was taken)
#   - Streaming support (shows nodes completing in real time)
# ============================================================

import streamlit as st
import sys, os, tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from phase3.step3_document_manager import DocumentManager
from phase4.step4_graph_assembly   import build_hr_graph, make_initial_state

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH   = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")
SUPPORTED_COUNTRIES = [
    "india", "usa", "uk", "germany", "australia",
    "canada", "singapore", "uae", "france", "japan",
]

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "HR Knowledge Base — LangGraph",
    page_icon  = "🤖",
    layout     = "wide",
)

# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner="Loading HR Knowledge Base...")
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name   = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True},
    )
    vs = Chroma(
        persist_directory  = VECTOR_STORE_PATH,
        embedding_function = embeddings,
        collection_name    = "hr_policies",
    )
    dm    = DocumentManager(vs)
    graph = build_hr_graph()
    return vs, dm, graph

vs, dm, graph = load_resources()

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "mode"         : "user",
    "chat_history" : [],
    "user_country" : "india",
    "admin_country": "india",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🤖 HR Knowledge Base")
    st.caption("Powered by LangGraph")
    st.divider()

    mode = st.radio(
        "Mode",
        ["💬 Employee Chat", "🔧 HR Admin"],
        index=0 if st.session_state.mode == "user" else 1,
    )
    st.session_state.mode = "user" if "Employee" in mode else "admin"
    st.divider()

    # Tavily status
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    st.markdown("**Web Search (Tavily)**")
    if tavily_ok:
        st.success("✅ Connected — web search active")
    else:
        st.warning("⚠️ Not configured — add TAVILY_API_KEY to .env")
        st.caption("Get free key: app.tavily.com")

    st.divider()

    # Stats
    stats = dm.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Documents",
                sum(v["num_files"] for v in stats["by_country"].values()))
    col2.metric("Countries", stats["total_countries"])
    st.metric("Total chunks", stats["total_chunks"])

    st.divider()
    st.caption("Phase 4 — LangGraph Orchestration")

# ============================================================
# USER MODE — Employee Chat
# ============================================================
if st.session_state.mode == "user":
    st.title("💬 HR Assistant")
    st.caption("Powered by LangGraph · Searches HR docs → LLM → Web automatically")

    col_country, col_clear = st.columns([3, 1])
    with col_country:
        selected = st.selectbox(
            "Your country",
            [c.upper() for c in SUPPORTED_COUNTRIES],
            index=SUPPORTED_COUNTRIES.index(st.session_state.user_country),
        )
        st.session_state.user_country = selected.lower()

    with col_clear:
        st.write("")
        if st.button("🗑️ Clear"):
            st.session_state.chat_history = []
            st.rerun()

    # ── LangGraph routing explanation ─────────────────────
    with st.expander("ℹ️ How this works (LangGraph routing)"):
        st.markdown("""
The LangGraph graph runs on every question:

```
Question → extract_intent → rag_search → [route decision]
                                              ↓
                              ┌───────────────┼───────────────┐
                         🟢 HIGH          🟡 MEDIUM        🔴 LOW
                       rag_answer     llm_supplement    web_search
                              ↓               ↓               ↓
                         format_response → END
```

- 🟢 **HIGH** confidence (≥0.40): Answered directly from your company's HR documents
- 🟡 **MEDIUM** confidence (≥0.25): HR docs + LLM general employment knowledge
- 🔴 **LOW** confidence (<0.25): Live web search via Tavily with source links
        """)

    st.divider()

    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                meta = msg["meta"]
                badge_map = {
                    "HIGH"  : ("🟢", "success"),
                    "MEDIUM": ("🟡", "warning"),
                    "LOW"   : ("🔴", "error"),
                }
                icon, color = badge_map.get(meta.get("tier", ""), ("⚪", "info"))
                route_label = {
                    "rag_answer"    : "HR Documents",
                    "llm_supplement": "HR Docs + LLM",
                    "web_search"    : "Web Search",
                }.get(meta.get("source_type", ""), meta.get("source_type", ""))

                col1, col2, col3 = st.columns(3)
                col1.metric("Source",     route_label)
                col2.metric("Confidence", f"{meta.get('confidence', 0):.0%}")
                col3.metric("Tier",       f"{icon} {meta.get('tier','')}")

                with st.expander("🔍 Node execution path"):
                    for step in meta.get("node_log", []):
                        st.markdown(f"  ✓ `{step}`")

    # ── Chat input ────────────────────────────────────────
    question = st.chat_input("Ask your HR question...")

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # Show live node progress using streaming
            status_area = st.empty()
            answer_area = st.empty()

            node_steps = []
            final_result = None

            with st.spinner("Running LangGraph..."):
                init_state = make_initial_state(
                    question, st.session_state.user_country
                )

                for step in graph.stream(init_state):
                    for node_name, node_output in step.items():
                        node_steps.append(node_name)
                        status_area.markdown(
                            f"⚙️ Running: `{node_name}`..."
                        )
                    final_result = node_output

            status_area.empty()

            # Get the final full state by invoking (stream gives partials)
            final_state = graph.invoke(
                make_initial_state(question, st.session_state.user_country)
            )

            answer      = final_state.get("answer", "No answer generated.")
            tier        = final_state.get("confidence_tier", "LOW")
            confidence  = final_state.get("confidence", 0.0)
            source_type = final_state.get("source_type", "")
            sources     = final_state.get("sources", [])
            node_log    = final_state.get("node_log", [])

            # Display answer
            answer_area.markdown(answer)

            # Source badges
            tier_map = {
                "HIGH"  : "🟢 Answered from company HR documents",
                "MEDIUM": "🟡 HR documents + general knowledge",
                "LOW"   : "🔴 Web search results",
            }
            st.caption(tier_map.get(tier, ""))

            if sources:
                clean_sources = [
                    s for s in sources
                    if s not in ("unknown", "web_search_unavailable",
                                 "LLM general knowledge")
                ]
                if clean_sources:
                    st.caption(f"📄 {', '.join(clean_sources)}")

        st.session_state.chat_history.append({
            "role"   : "assistant",
            "content": answer,
            "meta"   : {
                "tier"       : tier,
                "confidence" : confidence,
                "source_type": source_type,
                "sources"    : sources,
                "node_log"   : node_log,
            },
        })

# ============================================================
# ADMIN MODE
# ============================================================
else:
    st.title("🔧 HR Admin Panel")
    st.caption("Manage HR policy documents per country")

    admin_country = st.selectbox(
        "Managing country",
        [c.upper() for c in SUPPORTED_COUNTRIES],
        index=SUPPORTED_COUNTRIES.index(st.session_state.admin_country),
    )
    st.session_state.admin_country = admin_country.lower()
    country_key = st.session_state.admin_country

    tab_upload, tab_docs, tab_stats = st.tabs([
        "📤 Upload", "📁 Documents", "📊 Stats"
    ])

    # ── Upload ────────────────────────────────────────────
    with tab_upload:
        st.subheader(f"Upload for {admin_country}")
        uploaded = st.file_uploader(
            "HR policy documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("🚀 Upload to Knowledge Base", type="primary"):
            prog = st.progress(0)
            for i, f in enumerate(uploaded):
                prog.progress((i+1)/len(uploaded), text=f"Processing {f.name}...")
                suffix = Path(f.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    r = dm.add_document(tmp_path, country_key)
                    r["display_name"] = f.name
                    if r["status"] == "added":
                        st.success(f"✅ **{f.name}** — {r['chunks_added']} chunks added")
                    elif r["status"] == "skipped":
                        st.warning(f"⚠️ **{f.name}** — Already exists")
                    else:
                        st.error(f"❌ **{f.name}** — {r['message']}")
                finally:
                    os.unlink(tmp_path)
            # Clear cache so graph reloads updated vectorstore
            st.cache_resource.clear()
            st.rerun()

    # ── Documents ─────────────────────────────────────────
    with tab_docs:
        st.subheader(f"Documents — {admin_country}")
        docs = dm.list_documents(country=country_key)
        if not docs:
            st.info("No documents yet. Upload some above.")
        else:
            for doc in docs:
                c1, c2, c3 = st.columns([5, 2, 2])
                c1.markdown(f"📄 `{doc['filename']}`")
                c2.markdown(f"**{doc['chunk_count']}** chunks")
                if c3.button("🗑️", key=f"del_{doc['filename']}"):
                    r = dm.delete_document(doc["filename"])
                    if r["status"] == "deleted":
                        st.success(f"Deleted {r['chunks_deleted']} chunks")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(r["message"])

    # ── Stats ─────────────────────────────────────────────
    with tab_stats:
        st.subheader("Knowledge Base Overview")
        stats = dm.get_stats()

        if stats["by_country"]:
            cols = st.columns(min(len(stats["by_country"]), 4))
            for i, (ctry, info) in enumerate(stats["by_country"].items()):
                with cols[i % 4]:
                    st.metric(
                        ctry.upper(),
                        f"{info['chunks']} chunks",
                        f"{info['num_files']} file(s)",
                    )

        st.divider()
        all_docs = dm.list_documents()
        if all_docs:
            import pandas as pd
            df = pd.DataFrame(all_docs)[["country", "filename", "chunk_count"]]
            df.columns = ["Country", "Filename", "Chunks"]
            st.dataframe(
                df.sort_values(["Country", "Filename"]),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.subheader("LangGraph — Node Flow")
        st.code("""
START
  → extract_intent_node    (validate country)
  → rag_search_node        (ChromaDB search + confidence score)
  → [conditional edge]
       HIGH   → rag_answer_node      (answer from HR docs)
       MEDIUM → llm_supplement_node  (docs + LLM knowledge)
       LOW    → web_search_node      (Tavily live search)
  → format_response_node   (badge + source attribution)
END
        """, language="text")
