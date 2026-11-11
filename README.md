# 📊 Project Tracker Pro

AI-powered project management system with RAG, SQL Agent, and LLM routing.  
Built with Flask · Streamlit · LangChain · HuggingFace · FAISS · SQLite.

---

## 🗂️ Project Structure


```
ProjectTracker_FullApp/
├── backend/
│   ├── app.py                  ← Flask API server (run this first)
│   ├── ai_engine.py            ← LLM router + SQL agent + use-case handlers
│   ├── rag_indexer.py          ← FAISS index builder + 3-tier search
│   ├── prompts.py              ← All LLM prompt templates
│   ├── requirements.txt        ← Python dependencies
│   └── vector_store/           ← Auto-generated FAISS index (do not delete)
│       └── project_index/
│           ├── index.faiss
│           └── index.pkl
├── config/
│   ├── .env                    ← Your secrets (edit this before running)
│   └── env.template            ← Safe template to share
├── documents/                  ← BRS / spec files for RAG (one folder per project)
│   ├── pcidss/
│   │   └── pcidss_BRS.txt
│   ├── svt/
│   │   └── svt_BRS.txt
│   └── <project_name>/         ← Add your own folders here
│       └── <name>_BRS.txt
├── streamlit_app1.py           ← Main UI (run this second)
└── IMPLEMENTATION_GUIDE.md
```


---

##  Prerequisites

| Requirement | Version |
| Python 3.13


## 📦 Dependencies

Install all dependencies using the Python 3.13 framework binary:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pip install \
    flask==3.0.3 \
    flask-sqlalchemy==3.1.1 \
    flask-cors==4.0.1 \
    langchain==0.3.7 \
    langchain-core==0.3.28 \
    langchain-huggingface==0.1.2 \
    langchain-community==0.3.7 \
    langchain-experimental==0.3.0 \
    langchain-text-splitters==0.3.2 \
    sentence-transformers==3.3.1 \
    faiss-cpu==1.9.0 \
    huggingface-hub==0.26.5 \
    "tokenizers>=0.19.0" \
    "transformers>=4.39.0" \
    python-dotenv==1.0.1 \
    requests==2.32.3 \
    sqlalchemy==2.0.35 \
    "numpy>=1.26.0" \
    pydantic \
    streamlit \
    pandas \
    plotly
```

Or install from the requirements file:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pip install -r backend/requirements.txt
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pip install streamlit pandas plotly requests
```

---

## 🔑 Configuration

**1. Copy the env template:**
```bash
cp config/env.template config/.env
```

**2. Edit `config/.env`:**
```env
# Required — get your free key at https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=hf_your_token_here

# Path to your SQLite database
DATABASE_URL=sqlite:////Users/YOUR_USERNAME/Downloads/ProjectTracker/database/project_tracker.db

# Flask settings
FLASK_ENV=production


---

## 🚀 Running the Application

You need **two terminals** running simultaneously.

### Terminal 1 — Flask Backend

```bash
cd ~/Downloads/ProjectTracker_FullApp/backend
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 app.py
```

Expected output:
```
✅ .env loaded
✅ HF key loaded
✅ DB tables ready
✅ DB verified — N project(s)
[RAG] Building index... (first run only)
✅ AI engine ready
🚀 Flask running on :5000
```

Flask runs on: `http://localhost:5000`

### Terminal 2 — Streamlit Frontend

```bash
cd ~/Downloads/ProjectTracker_FullApp
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m streamlit run streamlit_app1.py
```

Streamlit runs on: `http://localhost:8501`

Open `http://localhost:8501` in your browser.

---

## 🤖 AI Architecture

```
User query
    │
    ▼
LLM Router (DeepSeek-V4-Pro via HuggingFace)
    │
    ├── Structured_Database_Lookup
    │       └── SQL Agent: LLM writes SQL → executes → LLM synthesizes answer
    │
    └── Unstructured_Document_RAG
            └── 3-Tier FAISS Search:
                  Tier 1 — explicit project namespace
                  Tier 2 — inferred from query text
                  Tier 3 — global semantic scan
```

### Use-case buttons (bypass router, direct handlers)
| Button | What it does |
|---|---|
| ✨ Summarize | Executive summary from DB data |
| ⚠️ Risk Radar | Ranked risks with mitigation suggestions |
| 💰 Budget | Utilisation analysis and overrun flags |
| 📅 Schedule | Deadline variance and health check |
| 📊 Portfolio | Cross-project weekly briefing |

---

## 📄 Adding Documents for RAG

Drop `.txt` or `*_BRS*` files into a project subfolder inside `documents/`:

```
documents/
  your_project_name/
    your_project_name_BRS.txt
```

The folder name must match the project name (lowercase, no spaces — e.g. `pcidss`, `svt`, `websiteredesign`).

Then click Re-index DB in the sidebar to rebuild the FAISS index.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health + AI status |
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create project |
| GET | `/api/projects/:id` | Get project with tasks + risks |
| PUT | `/api/projects/:id` | Update project |
| DELETE | `/api/projects/:id` | Delete project |
| GET | `/api/tasks` | List tasks (`?project_id=N`) |
| POST | `/api/projects/:id/tasks` | Create task |
| PUT | `/api/tasks/:id` | Update task |
| GET | `/api/risks` | List risks (`?project_id=N`) |
| POST | `/api/projects/:id/risks` | Create risk |
| **POST** | **`/api/bot/query`** | **Main AI endpoint** |
| POST | `/api/bot/reindex` | Rebuild FAISS index |
| GET | `/api/bot/history` | Chat history |
| GET | `/api/bot/status` | AI engine status + index metadata |

### AI query payload
```json
{
  "question": "Which project has the highest budget?",
  "project_id": 3,
  "project_namespace": "pcidss",
  "user_id": "user_001"
}
```

---

## 🗃️ Database Schema

The app uses a shared SQLite database (same file as the companion Jupyter notebook).

| Table | Key columns |
|---|---|
| `projects` | id, name, description, owner, status, priority, progress_percentage, budget, start_date, end_date |
| `tasks` | id, project_id, title, status, priority, assigned_to, estimated_hours, actual_hours, progress_percentage |
| `risks` | id, project_id, title, severity, probability, mitigation_plan, owner, status |
| `chat_history` | id, user_id, user_message, bot_response, project_context_id, query_type, confidence_score |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Use the full Python 3.13 path — not `python3` or `python` |
| `AI engine not ready` | Check `HUGGINGFACE_API_KEY` in `config/.env` |
| `DNS / ConnectionError` | HF API unreachable — check internet / VPN |
| `No documents to index` | Add at least one project to the DB first |
| AI shows "Offline" | Flask backend is not running — start Terminal 1 first |
| Port 5000 already in use | `lsof -ti:5000 \| xargs kill` then restart Flask |
| Port 8501 already in use | `lsof -ti:8501 \| xargs kill` then restart Streamlit |


## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Flask 3.0 + Flask-SQLAlchemy |
| Database | SQLite (via SQLAlchemy ORM) |
| AI / LLM | DeepSeek-V4-Pro via HuggingFace Inference API |
| LLM Framework | LangChain 0.3.7 (structured output via Pydantic) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (CPU) |
| Python | 3.13 (Framework install) |

