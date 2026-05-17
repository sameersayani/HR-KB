# ============================================================
# PHASE 1 - STEP 4: Structured Output (JSON Parser)
# ============================================================
# What you'll learn:
#   - Getting structured JSON back from the LLM
#   - JsonOutputParser
#   - Why structured output matters for our HR system
#     (e.g. confidence score, source type, answer)
# ============================================================

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)

# -----------------------------------------------------------
# Why structured output?
# -----------------------------------------------------------
# In our HR system, when the LLM answers a question we need:
#   - The actual answer text
#   - A confidence score (so we know when to search the web)
#   - Whether it found info or not
#
# Getting this as a Python dict is much easier than parsing text.

# -----------------------------------------------------------
# 1. Define the expected JSON shape in the prompt
# -----------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR assistant. Answer the user's HR question.
    
    You MUST respond ONLY with a valid JSON object in this exact format:
    {{
        "answer": "your answer here",
        "confidence": "high | medium | low",
        "topics_covered": ["list", "of", "topics"],
        "needs_legal_advice": true or false
    }}
    
    Do not include any text outside the JSON object."""),
    ("human", "Question: {question}\nCountry context: {country}"),
])

parser = JsonOutputParser()

chain = prompt | llm | parser

# -----------------------------------------------------------
# 2. Run a query and get structured output
# -----------------------------------------------------------
print("=== Query 1: Clear HR question ===\n")

result = chain.invoke({
    "question": "How many days of annual leave is an employee entitled to?",
    "country": "India",
})

print("Type:", type(result))   # dict!
print("Full result:", result)
print("\nAnswer:", result["answer"])
print("Confidence:", result["confidence"])
print("Topics:", result["topics_covered"])
print("Needs legal advice:", result["needs_legal_advice"])

# -----------------------------------------------------------
# 3. A vaguer question — notice the confidence changes
# -----------------------------------------------------------
print("\n" + "="*50)
print("=== Query 2: Vague / edge-case question ===\n")

result_2 = chain.invoke({
    "question": "Can my employer make me work during a festival?",
    "country": "Japan",
})

print("Answer:", result_2["answer"])
print("Confidence:", result_2["confidence"])
print("Needs legal advice:", result_2["needs_legal_advice"])

# -----------------------------------------------------------
# HOW THIS CONNECTS TO OUR HR SYSTEM
# -----------------------------------------------------------
# Later in LangGraph, we will use the "confidence" field to
# decide what to do next:
#
#   high   → return the answer directly to the user
#   medium → search the RAG knowledge base first
#   low    → fall back to web search
#
# This JSON structure is the foundation of our routing logic.
# -----------------------------------------------------------
