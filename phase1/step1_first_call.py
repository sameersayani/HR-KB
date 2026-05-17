# ============================================================
# PHASE 1 - STEP 1: Your First Gemini Call via LangChain
# ============================================================
# What you'll learn:
#   - How to load environment variables
#   - How to initialise a Gemini LLM in LangChain
#   - The difference between .invoke() and a raw API call
#   - What HumanMessage / SystemMessage / AIMessage are
# ============================================================

from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq


# Always load your .env file first so the API key is available
load_dotenv()

# -----------------------------------------------------------
# 1. Initialise the model
# -----------------------------------------------------------
# model        → which Gemini variant to use
# temperature  → 0 = deterministic, 1 = creative
# ChatGoogleGenerativeAI wraps the raw Gemini API so LangChain
# can treat it the same as Claude, OpenAI, etc.
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",   # free, fast, great for learning
#     temperature=0.3,
# )

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
# -----------------------------------------------------------
# 2. Messages — the building blocks of every LLM conversation
# -----------------------------------------------------------
# SystemMessage  → sets the behaviour / persona of the model
# HumanMessage   → what the user says
# AIMessage      → what the model replies (you'll see this in history)

messages = [
    SystemMessage(content="You are a helpful HR assistant. Answer clearly and concisely."),
    HumanMessage(content="What is the purpose of a probation period in employment?"),
]

# -----------------------------------------------------------
# 3. Call the model
# -----------------------------------------------------------
print("Calling Gemini...\n")
response = llm.invoke(messages)

# response is an AIMessage object
print("Type of response:", type(response))
print("\n--- ANSWER ---")
print(response.content)

# -----------------------------------------------------------
# 4. Explore the response object
# -----------------------------------------------------------
print("\n--- METADATA ---")
print("Model used  :", response.response_metadata.get("model_name", "N/A"))
print("Finish reason:", response.response_metadata.get("finish_reason", "N/A"))

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# llm.invoke(messages) → sends the full message list to Gemini
#                         and returns ONE AIMessage.
# The SystemMessage is NOT shown to users; it guides the model.
# You can change the SystemMessage to completely change behaviour.
# -----------------------------------------------------------
