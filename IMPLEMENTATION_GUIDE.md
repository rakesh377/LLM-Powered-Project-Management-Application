# Project Tracker Pro — AI Implementation Guide
## Files delivered

```
ProjectTracker_AI/
├── backend/
│   ├── app.py                  ← Flask backend (replace existing)
│   ├── ai_engine.py            ← NEW: LLM Router + SQL Agent + use-case handlers
│   ├── rag_indexer.py          ← NEW: FAISS index builder + 3-tier search
│   ├── prompts.py              ← NEW: all LLM prompt templates
│   ├── requirements.txt        ← NEW: replaces requirements1.txt
│   └── vector_store/           ← NEW: auto-created at runtime (FAISS files go here)
├── config/
│   └── .env                    ← Edit DATABASE_URL and HUGGINGFACE_API_KEY
├── documents/                  ← NEW: drop BRS / spec .txt files here per project
│   └── website_redesign/
│       └── BRS_website_redesign.txt   ← sample BRS file
└── streamlit_app1.py           ← Frontend (replace existing)
```

---

## Step-by-step setup

### Step 1 — Place files
Copy all delivered files into your existing ProjectTracker_FullApp folder:

```
ProjectTracker_FullApp/
├── backend/          ← drop app.py, ai_engine.py, rag_indexer.py, prompts.py, requirements.txt here
├── config/.env       ← edit this file
├── documents/        ← create this folder at the root level (next to backend/)
└── streamlit_app1.py ← replace the existing one
```

### Step 2 — Edit config/.env
Open `config/.env` and set:

```
DATABASE_URL=sqlite:////Users/YOUR_NAME/Downloads/ProjectTracker/database/project_tracker.db
HUGGINGFACE_API_KEY=hf_YOUR_TOKEN_HERE
```

### Step 3 — Install dependencies
```bash
cd ProjectTracker_FullApp/backend
pip install -r requirements.txt
```

### Step 4 — Start the Flask backend
```bash
cd ProjectTracker_FullApp/backend
python app.py
```

On first start you will see:
```
✅ .env loaded
✅ HF key loaded
✅ DB tables ready
✅ DB verified — N project(s)
[RAG] Building index...        ← first run: builds FAISS from your DB rows
✅ AI engine ready
🚀 Flask running on :5000
```

The FAISS index is saved to `backend/vector_store/` and reused on subsequent starts.

### Step 5 — Start the Streamlit frontend
Open a second terminal:
```bash
cd ProjectTracker_FullApp
streamlit run streamlit_app1.py
```

Open http://localhost:8501 in your browser.
The sidebar AI widget will show "Online ✅" when the backend is up.

### Step 6 — Verify AI is working
Go to AI Bot page and type:
> "Show me the budget of all projects"

Expected: the LLM router picks `Structured_Database_Lookup`, the SQL agent generates
a SELECT query, executes it, and returns a natural-language summary.

---

## How the AI works (from the notebook)

### 1. LLM Router (ai_engine.py → _llm_route)
Every query goes through the LLM orchestrator first.
It reads the question and returns ONE of these tool names:
- `Structured_Database_Lookup` → SQL Agent
- `Unstructured_Document_RAG`  → 3-Tier RAG


### 2. SQL Agent (ai_engine.py → _sql_agent)
Three steps (from notebook Cell 16):
- Step A: SQL-writer LLM (temperature=0) generates a raw SQLite query from the schema
- Step B: Execute query directly against the .db file via sqlite3
- Step C: Orchestrator LLM synthesizes the rows into a human answer

### 3. 3-Tier RAG (rag_indexer.py → tiered_search)
From notebook Cell 18:
- Tier 1: Explicit namespace passed (e.g. project_namespace="pcidss")
- Tier 2: Keyword inference from the query text (scans document folder names)
- Tier 3: Global semantic scan across all indexed documents

### 4. Use-case handlers
Direct DB fetch + structured LLM prompt for:
Risk Radar, Project Summary, Budget Analysis, Schedule Health, Portfolio Summary

---

## Adding project documents for RAG

Create a folder inside `documents/` named after your project (lowercase, underscores):
```
documents/
  pcidss/
    PCI_DSS_BRS.txt
    requirements.txt
  svt/
    SVT_spec.txt
```

Then click Re-index DB in the sidebar. The AI will find and search these files
when users ask document-related questions.

---

## API endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | /api/bot/query | Main AI query (question + project_id) |
| POST | /api/bot/reindex | Rebuild FAISS index from DB + documents |
| GET | /api/bot/history | Chat history (optional ?project_id=N) |
| GET | /api/bot/status | AI engine status + index metadata |
| GET | /api/projects | List all projects |
| POST | /api/projects | Create project |
| GET | /api/projects/:id | Get project with tasks + risks |
| PUT | /api/projects/:id | Update project |
| DELETE | /api/projects/:id | Delete project |
| GET | /api/tasks | List tasks (?project_id=N) |
| POST | /api/projects/:id/tasks | Create task |
| PUT | /api/tasks/:id | Update task |
| GET | /api/risks | List risks (?project_id=N) |
| POST | /api/projects/:id/risks | Create risk |
| GET | /api/health | Backend health check |

---

## Troubleshooting

**AI shows "Offline"**
→ Flask backend is not running. Start it with `python app.py`.

**"HUGGINGFACE_API_KEY missing"**
→ Check config/.env — the key must start with `hf_`.

**"No documents to index"**
→ Add at least one project to the database first, then restart Flask.

**LLM times out (90s)**
→ DeepSeek-V4-Pro is a large model. For faster responses switch to
  `mistralai/Mistral-7B-Instruct-v0.3` in ai_engine.py (init_ai function).

**SQL agent generates wrong SQL**
→ The model ignored the schema. Add a real example row to DATABASE_SCHEMA_CONTEXT
  in prompts.py to help it.
