# ============================================================
# PHASE 3 - EXERCISE (Capstone): Full HR Knowledge Base App
# ============================================================
# Run with: streamlit run phase3/exercise_hr_app.py
#
# This is the complete Phase 3 application combining:
#   ✅ Step 1: Multi-tenant store (country isolation)
#   ✅ Step 2: Confidence scoring (HIGH/MEDIUM/LOW routing)
#   ✅ Step 3: Document Manager (add/list/delete)
#   ✅ Step 4: Admin UI (upload, manage documents)
#   ✅ Step 5: Smart Query Engine (RAG + routing)
#
# TWO MODES IN ONE APP:
#   🔧 Admin Mode  → HR admins upload & manage documents
#   💬 User Mode   → Employees ask HR questions
# ============================================================

import streamlit as st
import tempfile, os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from phase3.step3_document_manager import DocumentManager
from phase3.step5_query_engine import smart_query

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
    page_title = "HR Knowledge Base",
    page_icon  = "🏢",
    layout     = "wide",
)

# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(
        persist_directory  = VECTOR_STORE_PATH,
        embedding_function = embeddings,
        collection_name    = "hr_policies",
    )
    dm  = DocumentManager(vs)
    return vs, dm

vs, dm = load_resources()

# ============================================================
# SESSION STATE
# ============================================================
if "mode"          not in st.session_state: st.session_state.mode = "user"
if "chat_history"  not in st.session_state: st.session_state.chat_history = []
if "user_country"  not in st.session_state: st.session_state.user_country = "india"
if "admin_country" not in st.session_state: st.session_state.admin_country = "india"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/company.png", width=64)
    st.title("HR Knowledge Base")
    st.divider()

    # Mode toggle
    mode = st.radio(
        "Mode",
        options=["💬 Employee Chat", "🔧 HR Admin"],
        index=0 if st.session_state.mode == "user" else 1,
    )
    st.session_state.mode = "user" if "Employee" in mode else "admin"
    st.divider()

    # Stats
    stats = dm.get_stats()
    st.metric("Total documents",
              sum(v["num_files"] for v in stats["by_country"].values()))
    st.metric("Total chunks", stats["total_chunks"])
    st.metric("Countries",    stats["total_countries"])

    st.divider()
    st.caption("Phase 3 — Multi-tenant HR Knowledge Base")

# ============================================================
# USER MODE
# ============================================================
if st.session_state.mode == "user":
    st.title("💬 HR Assistant")
    st.caption("Ask any question about your employment, leave policies, benefits, and more.")

    # Country selector
    col_c, col_clear = st.columns([3, 1])
    with col_c:
        selected = st.selectbox(
            "Your country",
            options=[c.upper() for c in SUPPORTED_COUNTRIES],
            index=SUPPORTED_COUNTRIES.index(st.session_state.user_country),
        )
        st.session_state.user_country = selected.lower()

    with col_clear:
        st.write("")
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("Details"):
                    st.json(msg["meta"])

    # Chat input
    question = st.chat_input("Ask your HR question...")
    if question:
        # Show user message
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching HR knowledge base..."):
                result = smart_query(question, st.session_state.user_country)

            tier_badge = {
                "HIGH"  : "🟢 Answered from your company's HR documents",
                "MEDIUM": "🟡 Answered from HR docs + general knowledge",
                "LOW"   : "🔴 Not found in HR documents",
            }

            if result["answer"]:
                st.markdown(result["answer"])
                badge = tier_badge.get(result["tier"], "")
                st.caption(badge)
                if result["sources"]:
                    st.caption(f"📄 Sources: {', '.join(result['sources'])}")
            else:
                st.warning(
                    "I couldn't find this in your company's HR documents. "
                    "In the full system (Phase 4), I would automatically "
                    "search the web for you. Please contact your HR team directly."
                )

        # Save to history
        answer_text = result["answer"] or "Not found in HR documents."
        st.session_state.chat_history.append({
            "role"   : "assistant",
            "content": answer_text,
            "meta"   : {
                "confidence": result["confidence"],
                "tier"      : result["tier"],
                "route"     : result["route"],
                "sources"   : result["sources"],
            },
        })

# ============================================================
# ADMIN MODE
# ============================================================
else:
    st.title("🔧 HR Admin Panel")
    st.caption("Manage HR policy documents for each country.")

    admin_country = st.selectbox(
        "Managing documents for",
        options=[c.upper() for c in SUPPORTED_COUNTRIES],
        index=SUPPORTED_COUNTRIES.index(st.session_state.admin_country),
    )
    st.session_state.admin_country = admin_country.lower()
    country_key = st.session_state.admin_country

    tab_upload, tab_docs, tab_stats = st.tabs([
        "📤 Upload", "📁 Documents", "📊 Stats"
    ])

    # ── UPLOAD TAB ──
    with tab_upload:
        st.subheader(f"Upload for {admin_country}")

        uploaded = st.file_uploader(
            "Select HR policy files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded and st.button("🚀 Upload to Knowledge Base", type="primary"):
            bar     = st.progress(0)
            results = []
            for i, f in enumerate(uploaded):
                bar.progress((i + 1) / len(uploaded), text=f"Processing {f.name}...")
                suffix = Path(f.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    r = dm.add_document(tmp_path, country_key)
                    r["display_name"] = f.name
                    results.append(r)
                finally:
                    os.unlink(tmp_path)

            for r in results:
                name = r.get("display_name", r["filename"])
                if r["status"] == "added":
                    st.success(f"✅ **{name}** — {r['chunks_added']} chunks added")
                elif r["status"] == "skipped":
                    st.warning(f"⚠️ **{name}** — Already exists, skipped")
                else:
                    st.error(f"❌ **{name}** — {r['message']}")
            st.rerun()

    # ── DOCUMENTS TAB ──
    with tab_docs:
        st.subheader(f"Documents for {admin_country}")
        docs = dm.list_documents(country=country_key)

        if not docs:
            st.info(f"No documents for {admin_country}. Upload some above.")
        else:
            for doc in docs:
                c1, c2, c3 = st.columns([5, 2, 2])
                c1.markdown(f"📄 `{doc['filename']}`")
                c2.markdown(f"**{doc['chunk_count']}** chunks")
                if c3.button("🗑️", key=f"del_{doc['filename']}", help="Delete"):
                    r = dm.delete_document(doc["filename"])
                    if r["status"] == "deleted":
                        st.success(f"Deleted {r['chunks_deleted']} chunks")
                        st.rerun()
                    else:
                        st.error(r["message"])

    # ── STATS TAB ──
    with tab_stats:
        st.subheader("Knowledge Base Overview")
        stats = dm.get_stats()

        cols = st.columns(min(len(stats["by_country"]), 4))
        for i, (ctry, info) in enumerate(stats["by_country"].items()):
            with cols[i % 4]:
                st.metric(ctry.upper(), f"{info['chunks']} chunks",
                          f"{info['num_files']} file(s)")

        st.divider()
        st.subheader("All Documents")
        all_docs = dm.list_documents()
        if all_docs:
            import pandas as pd
            df = pd.DataFrame(all_docs)[["country","filename","chunk_count"]]
            df.columns = ["Country","Filename","Chunks"]
            st.dataframe(df.sort_values(["Country","Filename"]),
                         use_container_width=True, hide_index=True)
