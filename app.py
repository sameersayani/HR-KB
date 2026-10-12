"""Launch a lesson or capstone app from any project phase."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PHASES = {
    "Phase 1 - LangChain Foundations": [
        ("Step 1 - First LLM call", "phase1/step1_first_call.py", "script"),
        ("Step 2 - Prompt templates", "phase1/step2_prompt_templates.py", "script"),
        ("Step 3 - LCEL chains", "phase1/step3_chains.py", "script"),
        ("Step 4 - Structured output", "phase1/step4_structured_output.py", "script"),
        ("Step 5 - Conversation memory", "phase1/step5_memory.py", "script"),
        ("Capstone - Job description generator", "phase1/exercise_job_description_generator.py", "script"),
    ],
    "Phase 2 - Document Ingestion and RAG": [
        ("Step 1 - Document loading", "phase2/step1_document_loading.py", "script"),
        ("Step 2 - Text splitting", "phase2/step2_text_splitting.py", "script"),
        ("Step 3 - Embeddings", "phase2/step3_embeddings.py", "script"),
        ("Step 4 - Vector store", "phase2/step4_vector_store.py", "script"),
        ("Step 5 - Retrieval", "phase2/step5_retrieval.py", "script"),
        ("Step 6 - RAG chain", "phase2/step6_rag_chain.py", "script"),
        ("Capstone - HR RAG chatbot", "phase2/exercise_hr_rag_system.py", "script"),
    ],
    "Phase 3 - Multi-Tenant RAG": [
        ("Step 1 - Multi-tenant store", "phase3/step1_multitenant_store.py", "script"),
        ("Step 2 - Confidence scoring", "phase3/step2_confidence_scoring.py", "script"),
        ("Step 3 - Document manager", "phase3/step3_document_manager.py", "script"),
        ("Step 4 - Admin UI", "phase3/step4_admin_ui.py", "streamlit"),
        ("Step 5 - Query engine", "phase3/step5_query_engine.py", "script"),
        ("Capstone - HR app", "phase3/exercise_hr_app.py", "streamlit"),
    ],
    "Phase 4 - LangGraph": [
        ("Step 1 - LangGraph basics", "phase4/step1_langgraph_basics.py", "script"),
        ("Step 2 - Graph state", "phase4/step2_graph_state.py", "script"),
        ("Step 3 - Graph nodes", "phase4/step3_nodes.py", "script"),
        ("Step 4 - Graph assembly", "phase4/step4_graph_assembly.py", "script"),
        ("Step 5 - Web search", "phase4/step5_web_search.py", "script"),
        ("Capstone - LangGraph HR app", "phase4/exercise_hr_langgraph_app.py", "streamlit"),
    ],
    "Phase 5 - Multi-Agent System": [
        ("Step 1 - Multi-agent concepts", "phase5/step1_multiagent_concepts.py", "script"),
        ("Step 2 - Specialist agents", "phase5/step2_subagents.py", "script"),
        ("Step 3 - Supervisor", "phase5/step3_supervisor.py", "script"),
        ("Step 4 - Summarizer", "phase5/step4_summarizer.py", "script"),
        ("Step 5 - Graph assembly", "phase5/step5_graph_assembly.py", "script"),
        ("Capstone - Multi-agent HR app", "phase5/exercise_multiagent_app.py", "streamlit"),
    ],
}


def choose(options, prompt):
    for index, option in enumerate(options, start=1):
        label = option[0] if isinstance(option, tuple) else option
        print(f"  {index}. {label}")
    print("  0. Back / exit")

    while True:
        selection = input(prompt).strip()
        if selection == "0":
            return None
        if selection.isdigit() and 1 <= int(selection) <= len(options):
            return int(selection) - 1
        print(f"Enter a number from 0 to {len(options)}.")


def launch(label, script, kind):
    script_path = PROJECT_ROOT / script
    if not script_path.is_file():
        print(f"Cannot launch {label}: file not found: {script_path}")
        return

    if kind == "streamlit":
        command = [sys.executable, "-m", "streamlit", "run", str(script_path)]
    else:
        command = [sys.executable, str(script_path)]

    print(f"\nLaunching {label}. Press Ctrl+C to stop it and return here.\n")
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except KeyboardInterrupt:
        print("\nLaunch interrupted.")
    else:
        if result.returncode:
            print(f"\nThe program exited with code {result.returncode}.")


def main():
    print("HR Knowledge Base - Phase Launcher\n")
    phase_names = list(PHASES)

    while True:
        phase_index = choose(phase_names, "Select a phase: ")
        if phase_index is None:
            return

        phase_name = phase_names[phase_index]
        actions = PHASES[phase_name]
        while True:
            print(f"\n{phase_name}")
            action_index = choose(actions, "Select a step or app: ")
            if action_index is None:
                break
            launch(*actions[action_index])
            print()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting phase launcher.")