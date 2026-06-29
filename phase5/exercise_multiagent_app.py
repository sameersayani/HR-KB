# ============================================================
# PHASE 5 - EXERCISE: Multi-Agent HR System (Streamlit)
# ============================================================
# Run with: streamlit run phase5/exercise_multiagent_app.py
#
# Combines:
#   ✅ Phase 1-3: LLM, RAG, multi-tenant store, document mgmt
#   ✅ Phase 4: LangGraph fundamentals
#   ✅ Phase 5: Supervisor + 3 specialist agents + Summarizer,
#               with parallel fan-out via Send()
#
# WHAT'S NEW vs Phase 4:
#   - A question can trigger MULTIPLE agents in parallel
#   - Each agent's individual answer + confidence is visible
#   - The Summarizer's merge logic is shown transparently
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
from phase5.step5_graph_assembly import build_multiagent_graph, make_initial_state

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH   = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")
SUPPORTED_COUNTRIES = [
    "india", "usa", "uk", "germany", "australia",
    "canada", "singapore", "uae", "france", "japan",
]

AGENT_LABELS = {
    "rag_agent"   : ("📄", "RAG Agent",    "Company HR Documents"),
    "policy_agent": ("⚖️", "Policy Agent", "General Employment Law"),
    "web_agent"   : ("🌐", "Web Agent",    "Live Web Search"),
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "HR Knowledge Base — Multi-Agent",
    page_icon  = "🧩",
    layout     = "wide",
)

# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner="Loading multi-agent system...")
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
    graph = build_multiagent_graph()
    return dm, graph

dm, graph = load_resources()

# ============================================================
# SESSION STATE
# ============================================================
defaults = {"mode": "user", "chat_history": [], "user_country": "india", "admin_country": "india"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧩 HR Knowledge Base")
    st.caption("Multi-Agent System · Phase 5")
    st.divider()

    mode = st.radio("Mode", ["💬 Employee Chat", "🔧 HR Admin"],
                     index=0 if st.session_state.mode == "user" else 1)
    st.session_state.mode = "user" if "Employee" in mode else "admin"
    st.divider()

    st.markdown("**Specialist Agents**")
    for key, (icon, name, desc) in AGENT_LABELS.items():
        st.caption(f"{icon} **{name}** — {desc}")

    st.divider()
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    st.caption(f"{'✅' if tavily_ok else '⚠️'} Web Agent {'active' if tavily_ok else 'needs TAVILY_API_KEY'}")

    st.divider()
    stats = dm.get_stats()
    st.metric("Total chunks", stats["total_chunks"])
    st.metric("Countries",    stats["total_countries"])

# ============================================================
# USER MODE
# ============================================================
if st.session_state.mode == "user":
    st.title("🧩 HR Assistant — Multi-Agent")
    st.caption("A Supervisor decides which specialist agents to consult — sometimes more than one.")

    col_c, col_clear = st.columns([3, 1])
    with col_c:
        selected = st.selectbox("Your country", [c.upper() for c in SUPPORTED_COUNTRIES],
                                 index=SUPPORTED_COUNTRIES.index(st.session_state.user_country))
        st.session_state.user_country = selected.lower()
    with col_clear:
        st.write("")
        if st.button("🗑️ Clear"):
            st.session_state.chat_history = []
            st.rerun()

    with st.expander("ℹ️ How multi-agent routing works"):
        st.markdown("""
```
Question → Supervisor Agent (reads the question, decides agents needed)
                    ↓
       ┌────────────┼────────────┐
   RAG Agent   Policy Agent   Web Agent       ← 1, 2, or all 3 run IN PARALLEL
  (HR docs)   (legal reasoning) (Tavily)
       └────────────┼────────────┘
                    ↓
           Summarizer Agent (merges all outputs into one answer)
```
Try a comparison question like *"How does our maternity leave compare to EU minimum standards?"*
— it should trigger **both** the RAG Agent and the Policy Agent.
        """)

    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                meta = msg["meta"]
                st.markdown("**Agents consulted:**")
                cols = st.columns(len(meta["agent_results"]) or 1)
                for i, r in enumerate(meta["agent_results"]):
                    icon, name, desc = AGENT_LABELS.get(r["agent"], ("⚪", r["agent"], ""))
                    with cols[i]:
                        st.metric(f"{icon} {name}", f"{r['confidence']:.0%} conf")
                with st.expander("🔍 Individual agent answers"):
                    for r in meta["agent_results"]:
                        icon, name, desc = AGENT_LABELS.get(r["agent"], ("⚪", r["agent"], ""))
                        st.markdown(f"**{icon} {name}** ({desc})")
                        st.markdown(r["answer"])
                        if r["sources"]:
                            st.caption(f"Sources: {', '.join(r['sources'])}")
                        st.divider()

    question = st.chat_input("Ask your HR question...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Supervisor is routing your question to specialist agents..."):
                state  = make_initial_state(question, st.session_state.user_country)
                result = graph.invoke(state)

            st.markdown(result["final_answer"])

            agents_used = [r["agent"] for r in result["agent_results"]]
            labels = [AGENT_LABELS.get(a, (a, a, ""))[1] for a in agents_used]
            st.caption(f"🧩 Agents consulted: {', '.join(labels)}")

            st.markdown("**Agents consulted:**")
            cols = st.columns(len(result["agent_results"]) or 1)
            for i, r in enumerate(result["agent_results"]):
                icon, name, desc = AGENT_LABELS.get(r["agent"], ("⚪", r["agent"], ""))
                with cols[i]:
                    st.metric(f"{icon} {name}", f"{r['confidence']:.0%} conf")

            with st.expander("🔍 Individual agent answers"):
                for r in result["agent_results"]:
                    icon, name, desc = AGENT_LABELS.get(r["agent"], ("⚪", r["agent"], ""))
                    st.markdown(f"**{icon} {name}** ({desc})")
                    st.markdown(r["answer"])
                    if r["sources"]:
                        st.caption(f"Sources: {', '.join(r['sources'])}")
                    st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["final_answer"],
            "meta": {"agent_results": result["agent_results"]},
        })

# ============================================================
# ADMIN MODE (same as Phase 4 — document management unchanged)
# ============================================================
else:
    st.title("🔧 HR Admin Panel")
    st.caption("Manage HR policy documents per country")

    admin_country = st.selectbox("Managing country", [c.upper() for c in SUPPORTED_COUNTRIES],
                                  index=SUPPORTED_COUNTRIES.index(st.session_state.admin_country))
    st.session_state.admin_country = admin_country.lower()
    country_key = st.session_state.admin_country

    tab_upload, tab_docs, tab_stats = st.tabs(["📤 Upload", "📁 Documents", "📊 Stats"])

    with tab_upload:
        st.subheader(f"Upload for {admin_country}")
        uploaded = st.file_uploader("HR policy documents", type=["pdf", "docx", "txt"],
                                     accept_multiple_files=True)
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
                    if r["status"] == "added":
                        st.success(f"✅ **{f.name}** — {r['chunks_added']} chunks added")
                    elif r["status"] == "skipped":
                        st.warning(f"⚠️ **{f.name}** — Already exists")
                    else:
                        st.error(f"❌ **{f.name}** — {r['message']}")
                finally:
                    os.unlink(tmp_path)
            st.cache_resource.clear()
            st.rerun()

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

    with tab_stats:
        st.subheader("Knowledge Base Overview")
        stats = dm.get_stats()
        if stats["by_country"]:
            cols = st.columns(min(len(stats["by_country"]), 4))
            for i, (ctry, info) in enumerate(stats["by_country"].items()):
                with cols[i % 4]:
                    st.metric(ctry.upper(), f"{info['chunks']} chunks", f"{info['num_files']} file(s)")

        st.divider()
        all_docs = dm.list_documents()
        if all_docs:
            import pandas as pd
            df = pd.DataFrame(all_docs)[["country", "filename", "chunk_count"]]
            df.columns = ["Country", "Filename", "Chunks"]
            st.dataframe(df.sort_values(["Country", "Filename"]), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Multi-Agent Graph")
        st.code("""
START
  → supervisor_node          (LLM decides which agents are needed)
  → [Send fan-out: 1-3 of]
       rag_agent      (company HR documents)
       policy_agent   (general employment law)
       web_agent      (live Tavily search)
  → summarizer_node          (merges all agent outputs)
END
        """, language="text")
