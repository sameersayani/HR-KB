# # ============================================================
# # PHASE 1 - MINI EXERCISE (Capstone)
# # HR Job Description Generator
# # ============================================================
# # This exercise combines EVERYTHING from Phase 1:
# #   ✅ Step 1: Initialise Gemini via LangChain
# #   ✅ Step 2: Prompt templates with variables
# #   ✅ Step 3: LCEL chain with | operator
# #   ✅ Step 4: Structured JSON output
# #   ✅ Step 5: Multi-turn conversation memory
# #
# # WHAT IT DOES:
# #   Takes a job title + country → generates a structured
# #   job description as a JSON object with sections:
# #   title, summary, responsibilities, requirements, benefits
# # ============================================================

# from dotenv import load_dotenv
# # from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# import json

# load_dotenv()

# #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
# llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)

# # -----------------------------------------------------------
# # 1. The prompt — asks for structured JSON output
# # -----------------------------------------------------------
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are an expert HR specialist who writes compelling job descriptions.
    
#     Always respond with ONLY a valid JSON object in this exact structure:
#     {{
#         "job_title": "exact job title",
#         "country": "country name",
#         "summary": "2-3 sentence role overview",
#         "responsibilities": ["responsibility 1", "responsibility 2", "...up to 6 items"],
#         "requirements": {{
#             "must_have": ["skill/qualification 1", "...up to 5 items"],
#             "nice_to_have": ["skill 1", "...up to 3 items"]
#         }},
#         "benefits": ["benefit 1", "...up to 4 items"],
#         "salary_range": "typical range in local currency (if well-known)",
#         "employment_notes": "any country-specific employment law notes (probation, notice, etc.)"
#     }}
    
#     Make it realistic and country-appropriate. No text outside the JSON."""),
#     ("human", "Generate a job description for: {job_title} in {country}. Industry: {industry}."),
# ])

# parser = JsonOutputParser()
# chain = prompt | llm | parser

# # -----------------------------------------------------------
# # 2. Helper to display output nicely
# # -----------------------------------------------------------
# def display_job_description(jd: dict):
#     print(f"\n{'='*60}")
#     print(f"  {jd['job_title'].upper()}")
#     print(f"  {jd['country']} | {jd.get('salary_range', 'Salary TBD')}")
#     print(f"{'='*60}")

#     print(f"\n📋 SUMMARY\n{jd['summary']}")

#     print("\n🎯 RESPONSIBILITIES")
#     for r in jd["responsibilities"]:
#         print(f"  • {r}")

#     print("\n✅ MUST-HAVE REQUIREMENTS")
#     for r in jd["requirements"]["must_have"]:
#         print(f"  • {r}")

#     print("\n⭐ NICE TO HAVE")
#     for r in jd["requirements"]["nice_to_have"]:
#         print(f"  • {r}")

#     print("\n🎁 BENEFITS")
#     for b in jd["benefits"]:
#         print(f"  • {b}")

#     if jd.get("employment_notes"):
#         print(f"\n⚖️  EMPLOYMENT NOTES\n{jd['employment_notes']}")
#     print()

# # -----------------------------------------------------------
# # 3. Generate job descriptions for different countries
# # -----------------------------------------------------------
# roles = [
#     {"job_title": "HR Manager",          "country": "India",   "industry": "IT Services"},
#     {"job_title": "Software Engineer",   "country": "Germany", "industry": "Automotive"},
#     {"job_title": "Marketing Analyst",   "country": "USA",     "industry": "E-commerce"},
# ]

# for role in roles:
#     print(f"\nGenerating JD for {role['job_title']} in {role['country']}...")
#     try:
#         jd = chain.invoke(role)
#         display_job_description(jd)
#     except Exception as e:
#         print(f"Error: {e}")

# # -----------------------------------------------------------
# # 4. BONUS: Save the output to a JSON file
# # -----------------------------------------------------------
# print("\nGenerating final JD and saving to file...\n")

# final_jd = chain.invoke({
#     "job_title": "Senior HR Business Partner",
#     "country": "United Kingdom",
#     "industry": "Financial Services",
# })

# with open("phase1_output_job_description.json", "w") as f:
#     json.dump(final_jd, f, indent=2)

# print("✅ Saved to phase1_output_job_description.json")
# display_job_description(final_jd)

# # -----------------------------------------------------------
# # PHASE 1 COMPLETE — YOU NOW KNOW:
# # -----------------------------------------------------------
# # ✅ How to call Gemini via LangChain
# # ✅ How to build reusable prompt templates
# # ✅ How to chain prompt | llm | parser with LCEL
# # ✅ How to get structured JSON output from an LLM
# # ✅ How to maintain chat history across turns
# #
# # NEXT → Phase 2: Document loading, chunking, embeddings, RAG
# # -----------------------------------------------------------

# ============================================================
# PHASE 1 - MINI EXERCISE (Capstone)
# HR Job Description Generator
# ============================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.5)

# -----------------------------------------------------------
# 1. The prompt
# -----------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert HR specialist who writes compelling job descriptions.

    Always respond with ONLY a valid JSON object in this exact structure:
    {{
        "job_title": "exact job title",
        "country": "country name",
        "summary": "2-3 sentence role overview",
        "responsibilities": ["responsibility 1", "responsibility 2", "up to 6 items"],
        "requirements": {{
            "must_have": ["skill 1", "up to 5 items"],
            "nice_to_have": ["skill 1", "up to 3 items"]
        }},
        "benefits": ["benefit 1", "up to 4 items"],
        "salary_range": "typical range in local currency",
        "employment_notes": "country-specific employment law notes"
    }}

    Make it realistic and country-appropriate. No text outside the JSON."""),
    ("human", "Generate a job description for: {job_title} in {country}. Industry: {industry}."),
])

parser = JsonOutputParser()
chain = prompt | llm | parser

# -----------------------------------------------------------
# 2. Display helper
# -----------------------------------------------------------
def display_job_description(jd: dict):
    print(f"\n{'='*60}")
    print(f"  {jd['job_title'].upper()}")
    print(f"  {jd['country']} | {jd.get('salary_range', 'Salary TBD')}")
    print(f"{'='*60}")
    print(f"\nSUMMARY\n{jd['summary']}")
    print("\nRESPONSIBILITIES")
    for r in jd["responsibilities"]:
        print(f"  - {r}")
    print("\nMUST-HAVE REQUIREMENTS")
    for r in jd["requirements"]["must_have"]:
        print(f"  - {r}")
    print("\nNICE TO HAVE")
    for r in jd["requirements"]["nice_to_have"]:
        print(f"  - {r}")
    print("\nBENEFITS")
    for b in jd["benefits"]:
        print(f"  - {b}")
    if jd.get("employment_notes"):
        print(f"\nEMPLOYMENT NOTES\n{jd['employment_notes']}")
    print()

# -----------------------------------------------------------
# 3. Generate JDs
# -----------------------------------------------------------
roles = [
    {"job_title": "HR Manager",        "country": "India",   "industry": "IT Services"},
    {"job_title": "Software Engineer", "country": "Germany", "industry": "Automotive"},
    {"job_title": "Marketing Analyst", "country": "USA",     "industry": "E-commerce"},
]

for role in roles:
    print(f"\nGenerating JD for {role['job_title']} in {role['country']}...")
    try:
        jd = chain.invoke(role)
        display_job_description(jd)
    except Exception as e:
        print(f"Error: {e}")

# -----------------------------------------------------------
# 4. Save final JD to file
# -----------------------------------------------------------
print("\nGenerating final JD and saving to file...\n")
final_jd = chain.invoke({
    "job_title": "Senior HR Business Partner",
    "country": "India",
    "industry": "Software Products & Services",
})

with open("phase1_output_job_description.json", "w") as f:
    json.dump(final_jd, f, indent=2)

print("Saved to phase1_output_job_description.json")
display_job_description(final_jd)