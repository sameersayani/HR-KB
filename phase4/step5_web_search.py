# ============================================================
# PHASE 4 - STEP 5: Web Search with Tavily
# ============================================================
# What you'll learn:
#   - Getting a free Tavily API key
#   - Using TavilySearch inside a LangGraph node
#   - Formatting web results with source URLs
#   - Testing the full LOW confidence path end-to-end
#
# HOW TO GET A FREE TAVILY KEY:
#   1. Go to https://app.tavily.com
#   2. Sign up (free, no credit card)
#   3. Copy your API key from the dashboard
#   4. Add to .env:  TAVILY_API_KEY=tvly-xxxxxxxx
# ============================================================

import os, sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# CHECK: is Tavily key configured?
# ============================================================
tavily_key = os.getenv("TAVILY_API_KEY")

if not tavily_key:
    print("""
╔══════════════════════════════════════════════════════╗
║           TAVILY API KEY NOT FOUND                   ║
╠══════════════════════════════════════════════════════╣
║  1. Go to https://app.tavily.com                     ║
║  2. Sign up free (no credit card needed)             ║
║  3. Copy your API key from the dashboard             ║
║  4. Add to your .env file:                           ║
║     TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx             ║
║  5. Re-run this file                                 ║
╚══════════════════════════════════════════════════════╝
    """)
else:
    print(f"✅ Tavily key found: {tavily_key[:10]}...")

# ============================================================
# TEST 1: Raw Tavily search
# ============================================================
print("\n" + "="*55)
print("TEST 1: Raw Tavily Search")
print("="*55)

if tavily_key:
    try:
        from langchain_tavily import TavilySearch

        tool    = TavilySearch(max_results=3)
        results = tool.invoke({
            "query": "maternity leave policy India employment law 2024"
        })

        print(f"\nResults returned: {len(results)}")
        for i, r in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"    Title  : {r.get('title', 'N/A')}")
            print(f"    URL    : {r.get('url',   'N/A')}")
            print(f"    Preview: {r.get('content', '')[:150]}...")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Skipped — no Tavily key.")

# ============================================================
# TEST 2: Full LOW confidence path through the graph
# ============================================================
print("\n" + "="*55)
print("TEST 2: Full graph — LOW confidence question")
print("="*55)

from phase4.step4_graph_assembly import build_hr_graph, make_initial_state, display_result

graph = build_hr_graph()

low_confidence_queries = [
    ("What are the visa sponsorship rules for H1B?",     "usa"),
    ("How do I apply for a work permit in Germany?",     "india"),
    ("What are the tax implications of remote work UK?", "uk"),
]

for question, country in low_confidence_queries:
    print(f"\nQuery: '{question}'")
    state  = make_initial_state(question, country)
    result = graph.invoke(state)
    display_result(result)

# ============================================================
# TEST 3: Verify all three paths work
# ============================================================
print("\n" + "="*55)
print("TEST 3: All three routing paths")
print("="*55)

all_path_queries = [
    ("How many days maternity leave in India?",    "india", "HIGH"),
    ("Can I take casual leave during probation?",  "india", "MEDIUM"),
    ("What is the H1B visa cap this year?",        "usa",   "LOW"),
]

print(f"\n{'Question':<45} {'Expected':>8}  {'Actual':>8}  {'Match'}")
print("-"*70)

for question, country, expected_tier in all_path_queries:
    state  = make_initial_state(question, country)
    result = graph.invoke(state)
    actual = result.get("confidence_tier", "?")
    match  = "✓" if actual == expected_tier else "✗"
    print(f"{question:<45} {expected_tier:>8}  {actual:>8}  {match}")
