# ============================================================
# PHASE 1 - STEP 5: Conversation Memory (Chat History)
# ============================================================
# What you'll learn:
#   - LLMs have NO memory between calls — each .invoke() is fresh
#   - How to manually maintain chat history
#   - MessagesPlaceholder — inject history into a prompt
#   - Building a simple multi-turn HR chat loop
# ============================================================

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# -----------------------------------------------------------
# 1. Prompt with a placeholder for history
# -----------------------------------------------------------
# MessagesPlaceholder("history") → inserts the full message list
# at this position in the prompt.

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable HR assistant for {country}.
    Answer questions about employment, policies, and workplace rules.
    If you don't know something specific to the country, say so."""),
    MessagesPlaceholder("history"),   # ← chat history goes here
    ("human", "{question}"),
])

parser = StrOutputParser()
chain = prompt | llm | parser

# -----------------------------------------------------------
# 2. Maintain history in a plain Python list
# -----------------------------------------------------------
# Each turn:
#   1. Add HumanMessage to history
#   2. Invoke chain with current history
#   3. Add AIMessage (the response) to history
#   4. Repeat
#
# The list grows with each turn, giving the model context.

def chat(country: str):
    history = []
    print(f"\n HR Assistant for {country}")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Add user message to history BEFORE invoking
        history.append(HumanMessage(content=user_input))

        # Invoke the chain — passes full history each time
        response = chain.invoke({
            "country": country,
            "history": history[:-1],   # history up to (not including) current msg
            "question": user_input,
        })

        print(f"\nAssistant: {response}\n")

        # Add assistant response to history for next turn
        history.append(AIMessage(content=response))

# -----------------------------------------------------------
# 3. Demo — run a simulated conversation (no user input needed)
# -----------------------------------------------------------
print("=== Simulated 3-turn conversation ===\n")

history = []
country = "India"

questions = [
    "What is the standard probation period for new employees?",
    "Can the employer extend it beyond that?",     # refers to "it" from previous answer
    "What rights does the employee have during probation?",
]

for q in questions:
    print(f"You: {q}")
    history.append(HumanMessage(content=q))

    response = chain.invoke({
        "country": country,
        "history": history[:-1],
        "question": q,
    })

    print(f"Assistant: {response}\n")
    history.append(AIMessage(content=response))

print("="*50)
print(f"Total messages in history: {len(history)}")
print("(Each turn = 1 HumanMessage + 1 AIMessage)")

# -----------------------------------------------------------
# KEY TAKEAWAY
# -----------------------------------------------------------
# The LLM itself has no memory. YOU maintain the history list
# and pass it on every call. This is the exact same pattern
# LangGraph will manage for us automatically in Phase 4.
# -----------------------------------------------------------

# Uncomment to run the live interactive version:
chat("India")
