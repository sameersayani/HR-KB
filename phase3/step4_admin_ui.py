# ============================================================
# PHASE 3 - STEP 4: HR Admin UI (Streamlit)
# ============================================================
# Run with:  streamlit run phase3/step4_admin_ui.py
#
# What this provides:
#   - HR admins select their country from a dropdown
#   - Upload PDF, DOCX, or TXT documents
#   - View all uploaded documents per country
#   - Delete documents from the knowledge base
#   - Live stats: total documents & chunks per country
# ============================================================

import streamlit as st
import sys
from pathlib import Path

# Allow importing from parent directory
sys.path.append(str(Path(__file__).parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from phase3.step3_document_manager import DocumentManager
import tempfile, os

VECTOR_STORE_PATH = "vectorstore/hr_chroma_db"
SUPPORTED_COUNTRIES = [
    "India", "USA", "UK", "Germany", "Australia",
    "Canada", "Singapore", "UAE", "France", "Japan",
]

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "HR Knowledge Base — Admin",
    page_icon  = "📋",
    layout     = "wide",
)

# ============================================================
# INIT: load embeddings + vector store (cached)
# ============================================================
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_components():
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
    return DocumentManager(vs)

dm = load_components()

# ============================================================
# SIDEBAR — Country selection + Stats
# ============================================================
with st.sidebar:
    st.title("📋 HR Admin Panel")
    st.divider()

    selected_country = st.selectbox(
        "Select your country",
        options=SUPPORTED_COUNTRIES,
        index=0,
    )
    country_key = selected_country.lower()

    st.divider()

    # Live stats
    stats = dm.get_stats()
    st.subheader("Knowledge Base Stats")
    st.metric("Total chunks",    stats["total_chunks"])
    st.metric("Countries loaded", stats["total_countries"])

    if country_key in stats["by_country"]:
        info = stats["by_country"][country_key]
        st.metric(f"{selected_country} docs",   info["num_files"])
        st.metric(f"{selected_country} chunks", info["chunks"])
    else:
        st.info(f"No documents for {selected_country} yet.")

# ============================================================
# MAIN AREA
# ============================================================
st.title(f"📋 HR Knowledge Base — {selected_country}")
st.caption("Upload and manage HR policy documents for your country.")

tab_upload, tab_documents = st.tabs(["📤 Upload Documents", "📁 Manage Documents"])

# ============================================================
# TAB 1: UPLOAD
# ============================================================
with tab_upload:
    st.subheader(f"Upload HR Documents for {selected_country}")
    st.markdown(
        "Upload your company HR policy documents. Supported formats: "
        "**PDF**, **DOCX**, **TXT**. Each file will be processed, "
        "chunked, and stored in the knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type       = ["pdf", "docx", "txt"],
        accept_multiple_files = True,
        help="You can upload multiple files at once.",
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            st.write(f"  • `{f.name}` ({size_kb} KB)")

        col1, col2 = st.columns([1, 4])
        with col1:
            upload_btn = st.button(
                "🚀 Process & Upload",
                type="primary",
                use_container_width=True,
            )

        if upload_btn:
            progress = st.progress(0, text="Starting upload...")
            results  = []

            for i, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Processing {uploaded_file.name}..."
                )

                # Save to a temp file so loaders can read it
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    result = dm.add_document(tmp_path, selected_country)
                    # Rename for display (use original filename)
                    result["filename"] = uploaded_file.name
                    results.append(result)
                finally:
                    os.unlink(tmp_path)

            progress.progress(1.0, text="Done!")

            # Show results
            st.divider()
            st.subheader("Upload Results")
            for result in results:
                if result["status"] == "added":
                    st.success(
                        f"✅ **{result['filename']}** — "
                        f"Added {result['chunks_added']} chunks to "
                        f"{selected_country} knowledge base."
                    )
                elif result["status"] == "skipped":
                    st.warning(
                        f"⚠️ **{result['filename']}** — "
                        f"Already exists. Skipped."
                    )
                else:
                    st.error(
                        f"❌ **{result['filename']}** — "
                        f"Error: {result['message']}"
                    )

            st.rerun()

# ============================================================
# TAB 2: MANAGE DOCUMENTS
# ============================================================
with tab_documents:
    st.subheader(f"Documents in {selected_country} Knowledge Base")

    docs = dm.list_documents(country=country_key)

    if not docs:
        st.info(
            f"No documents found for {selected_country}. "
            "Upload some documents in the Upload tab."
        )
    else:
        st.write(f"**{len(docs)} document(s)** in the knowledge base:")
        st.divider()

        for doc in docs:
            col_name, col_chunks, col_action = st.columns([5, 2, 2])

            with col_name:
                st.markdown(f"📄 `{doc['filename']}`")

            with col_chunks:
                st.markdown(f"**{doc['chunk_count']}** chunks")

            with col_action:
                if st.button(
                    "🗑️ Delete",
                    key=f"del_{doc['filename']}",
                    type="secondary",
                ):
                    with st.spinner(f"Deleting {doc['filename']}..."):
                        result = dm.delete_document(doc["filename"])
                    if result["status"] == "deleted":
                        st.success(
                            f"Deleted {result['chunks_deleted']} chunks "
                            f"from `{doc['filename']}`."
                        )
                        st.rerun()
                    else:
                        st.error(result["message"])

        # Divider + all-countries overview
        st.divider()
        with st.expander("📊 View all countries"):
            all_docs = dm.list_documents()
            if all_docs:
                import pandas as pd
                df = pd.DataFrame(all_docs)[["country", "filename", "chunk_count"]]
                df.columns = ["Country", "Filename", "Chunks"]
                df = df.sort_values(["Country", "Filename"])
                st.dataframe(df, use_container_width=True, hide_index=True)
