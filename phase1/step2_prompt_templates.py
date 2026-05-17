# ============================================================
# PHASE 1 - STEP 2: Prompt Templates
# ============================================================
# What you'll learn:
#   - Why hardcoding prompts is bad
#   - ChatPromptTemplate: the standard way to build prompts
#   - How to use variables (placeholders) in prompts
#   - PromptTemplate vs ChatPromptTemplate
# ============================================================

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# -----------------------------------------------------------
# PROBLEM with hardcoding prompts:
# -----------------------------------------------------------
# If you hardcode "What is the leave policy in India?" you can
# only ever answer that one question. We need templates.

# -----------------------------------------------------------
# 1. Basic ChatPromptTemplate
# -----------------------------------------------------------
# {country} and {topic} are placeholders — filled at runtime.

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert HR consultant specialising in employment law for {country}."),
    ("human", "Explain the rules around {topic} in simple terms."),
])

# .format_messages() fills in the placeholders and returns a list
# of messages ready to send to the LLM.
filled_messages = prompt.format_messages(country="India", topic="maternity leave")

print("=== Filled prompt messages ===")
for msg in filled_messages:
    print(f"[{msg.type.upper()}]: {msg.content}\n")

# Now invoke the model with the filled messages
response = llm.invoke(filled_messages)
print("=== ANSWER ===")
print(response.content)

# -----------------------------------------------------------
# 2. Reuse the same template with different inputs
# -----------------------------------------------------------
print("\n" + "="*50)
print("Same template, different country + topic:")
print("="*50 + "\n")

filled_messages_2 = prompt.format_messages(country="Germany", topic="notice period")
response_2 = llm.invoke(filled_messages_2)
print(response_2.content)

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# Templates separate STRUCTURE from DATA.
# You define the prompt shape once and fill in variables later.
# This is exactly how our HR Knowledge Base will work —
# the same template will serve HR queries from any country.
# -----------------------------------------------------------
