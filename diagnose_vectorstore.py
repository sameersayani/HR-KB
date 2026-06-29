# ============================================================
# DIAGNOSTIC: Inspect the root vectorstore
# ============================================================
# Run this from the HR-KB/ root folder:
#   python diagnose_vectorstore.py
# ============================================================

from pathlib import Path
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vectorstore" / "hr_chroma_db")

print(f"Inspecting vector store at: {VECTOR_STORE_PATH}\n")

# ------------------------------------------------------------
# 1. List ALL collections in this store (not just "hr_policies")
# ------------------------------------------------------------
client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
collections = client.list_collections()

print("=" * 60)
print(f"COLLECTIONS FOUND: {len(collections)}")
print("=" * 60)
for col in collections:
    print(f"  - name: '{col.name}'  |  count: {col.count()}")

if not collections:
    print("\n❌ NO COLLECTIONS FOUND AT ALL.")
    print("This vector store is empty — it was never populated at this path.")
    exit()

# ------------------------------------------------------------
# 2. Look specifically for "hr_policies" (what our code expects)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("LOOKING FOR 'hr_policies' COLLECTION")
print("=" * 60)

target = None
for col in collections:
    if col.name == "hr_policies":
        target = col
        break

if target is None:
    print("❌ No collection named 'hr_policies' exists.")
    print(f"   Available collection name(s): {[c.name for c in collections]}")
    print("\n   Our code does: collection_name='hr_policies'")
    print("   If your data is under a different name, that's the mismatch.")
    exit()

print(f"✅ Found 'hr_policies' with {target.count()} items.")

# ------------------------------------------------------------
# 3. Inspect actual metadata — check country casing/values
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("METADATA INSPECTION — country field values")
print("=" * 60)

data = target.get(limit=2000, include=["metadatas"])
metadatas = data["metadatas"]

from collections import Counter
country_values = Counter(m.get("country", "<<MISSING>>") for m in metadatas)
filename_values = Counter(m.get("filename", "<<MISSING>>") for m in metadatas)

print("\nDistinct 'country' values found in metadata:")
for val, count in country_values.items():
    print(f"  '{val}'  →  {count} chunks")

print("\nDistinct 'filename' values found in metadata:")
for val, count in filename_values.items():
    print(f"  '{val}'  →  {count} chunks")

# ------------------------------------------------------------
# 4. Test the EXACT filter our code uses
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("TESTING FILTER: {'country': 'india'}  (lowercase, as our code uses)")
print("=" * 60)

filtered = target.get(where={"country": "india"}, limit=10, include=["metadatas"])
print(f"Chunks matching country='india': {len(filtered['ids'])}")

if len(filtered['ids']) == 0:
    print("\n❌ THIS IS THE BUG — no chunks have country='india' exactly.")
    print("   Check the 'Distinct country values' list above —")
    print("   if you see 'India' (capitalized) instead of 'india',")
    print("   that casing mismatch is exactly why RAG agent finds nothing.")
else:
    print("✅ Filter works — chunks ARE found for India.")
    print("   If the app still fails, the issue may be elsewhere")
    print("   (e.g. embedding query mismatch, or app not using this path).")