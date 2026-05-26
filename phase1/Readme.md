# Phase 1 — LangChain Foundations with Google Gemini

## 🎯 Goal
Learn the fundamental building blocks of LangChain before touching
any RAG or agent logic.

---

## ⚙️ Setup (Do this ONCE before running any file)

### 1. Get your free Gemini API key
1. Go to → https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key

### 2. Create your .env file
```bash
# In the hr-knowledgebase/ folder
cp .env.example .env
```
Open `.env` and replace `your_google_api_key_here` with your actual key.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📚 Run the steps in order

```bash
# Step 1: Your first Gemini/Groq call
python phase1/step1_first_call.py

# Step 2: Prompt templates
python phase1/step2_prompt_templates.py

# Step 3: LCEL chains (prompt | llm | parser)
python phase1/step3_chains.py

# Step 4: Structured JSON output
python phase1/step4_structured_output.py

# Step 5: Conversation memory / chat history
python phase1/step5_memory.py

# Mini Exercise: HR Job Description Generator
python phase1/exercise_job_description_generator.py
```

---

## 🧠 What Each Step Teaches

| File | Concept | Key Class/Method |
|---|---|---|
| `step1_first_call.py` | Call an LLM | `ChatGoogleGenerativeAI`, `.invoke()` |
| `step2_prompt_templates.py` | Reusable prompts | `ChatPromptTemplate`, variables |
| `step3_chains.py` | LCEL pipe operator | `prompt \| llm \| parser` |
| `step4_structured_output.py` | JSON from LLM | `JsonOutputParser` |
| `step5_memory.py` | Chat history | `MessagesPlaceholder`, manual history |
| `exercise_*.py` | All combined | Full pipeline |

---

## ❓ Common Issues

**`google.auth.exceptions.DefaultCredentialsError`**
→ Your `.env` file is missing or `GOOGLE_API_KEY` is not set correctly.

**`ModuleNotFoundError: langchain_google_genai`**
→ Run `pip install langchain-google-genai`

**`429 Resource has been exhausted`**
→ Free tier rate limit hit. Wait 60 seconds and try again.

---

## ✅ You're ready for Phase 2 when you can:
- [ ] Call Gemini and get a response
- [ ] Use a prompt template with variables
- [ ] Build a chain with `|` and get plain string output
- [ ] Get structured JSON back from the LLM
- [ ] Hold a 3-turn conversation using history
