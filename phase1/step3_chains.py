# ============================================================
# PHASE 1 - STEP 3: LCEL Chains (LangChain Expression Language)
# ============================================================
# What you'll learn:
#   - The | (pipe) operator — the core of LCEL
#   - Building chains: prompt | llm | parser
#   - StrOutputParser: turning AIMessage → plain string
#   - Why chains are better than calling each step manually
# ============================================================

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# -----------------------------------------------------------
# WITHOUT a chain (what you did in steps 1 & 2):
# -----------------------------------------------------------
# prompt_filled = prompt.format_messages(...)
# response      = llm.invoke(prompt_filled)
# text          = response.content
#
# Three separate lines, and they can't be composed or reused.

# -----------------------------------------------------------
# WITH a chain (LCEL style):
# -----------------------------------------------------------
# The | operator passes the output of the left side as the
# input to the right side — just like a Unix pipe.
#
#   prompt  →  fills placeholders  →  returns list of messages
#   llm     →  calls Gemini API    →  returns AIMessage
#   parser  →  extracts .content   →  returns plain string

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert HR assistant. Be concise and professional."),
    ("human", "Write a short job description for the role of {job_title} in a {industry} company."),
])

parser = StrOutputParser()

# Build the chain — this does NOT call the API yet
chain = prompt | llm | parser

print("Chain created:", chain)
print("\n--- Running chain ---\n")

# .invoke() triggers the whole pipeline end-to-end
result = chain.invoke({
    "job_title": "Senior Software Engineer",
    "industry": "FinTech",
})

print(result)
print("\nType of result:", type(result))  # str, not AIMessage!

# -----------------------------------------------------------
# 2. Reuse the chain for different inputs — no rewriting
# -----------------------------------------------------------
print("\n" + "="*50 + "\n")

result_2 = chain.invoke({
    "job_title": "HR Business Partner",
    "industry": "Healthcare",
})
print(result_2)

# -----------------------------------------------------------
# 3. Batch: run the same chain over multiple inputs at once
# -----------------------------------------------------------
print("\n" + "="*50)
print("Batch mode — 3 roles at once:")
print("="*50 + "\n")

batch_inputs = [
    {"job_title": "Data Analyst",        "industry": "Retail"},
    {"job_title": "Marketing Manager",   "industry": "E-commerce"},
    {"job_title": "Legal Counsel",       "industry": "Banking"},
]

results = chain.batch(batch_inputs)
for inp, res in zip(batch_inputs, results):
    print(f"Role: {inp['job_title']} ({inp['industry']})")
    print(res[:500], "...\n")  # print first 500 chars

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# chain = prompt | llm | parser
#
# This is the fundamental pattern in LangChain.
# Every more complex system (RAG, agents) is built on this.
# The pipe makes it readable, composable, and easy to swap parts.
# e.g. swap `llm` for a different model → rest stays the same.
# -----------------------------------------------------------
