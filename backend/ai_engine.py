"""
ai_engine.py  —  Core AI layer for Project Tracker Pro  (v3)

v3 changes from v2:
1. Structured output via Pydantic + with_structured_output() — replaces all
   regex/sqlglot SQL extraction. The LLM is asked for a typed object directly.
2. Router simplified to exactly 2 tools for free-text chat:
     Structured_Database_Lookup, Unstructured_Document_RAG
   The 5 use-case tools (Risk_Radar, Project_Summary, Budget_Analysis,
   Schedule_Health, Portfolio_Summary) are now UI-button-only — they are
   exposed as separate functions called directly by app.py's dedicated
   endpoints, and are no longer reachable through process_query's router.
3. Keyword router removed entirely. process_query always asks the LLM router.
4. Conversation memory — process_query accepts a chat_history list and passes
   it into the router, SQL writer, and RAG prompts so follow-up questions
   ("what about its budget?") resolve correctly until Clear chat is pressed.

File location: ProjectTracker_FullApp/backend/ai_engine.py
"""

import os
import sqlite3
import logging
from datetime import date
from typing import Literal, Optional
from typing import TypedDict
from typing import TypedDict, Annotated, Optional, Literal

from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from sqlalchemy import create_engine, text

from prompts import (
    ROUTER_PROMPT, SQL_WRITER_PROMPT, SQL_SYNTHESIS_PROMPT, RAG_PROMPT,
    GENERAL_CHAT_PROMPT,
    RISK_RADAR_PROMPT, SUMMARIZE_PROMPT, BUDGET_PROMPT,
    SCHEDULE_PROMPT, PORTFOLIO_PROMPT,
)
from rag_indexer import build_index, load_index, get_index_meta, tiered_search, clean_namespace

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_chat_orchestrator       = None   # general LLM — synthesis, RAG, use-case handlers
_chat_sql_writer         = None   # strict LLM for SQL generation (temp=0)
_router_structured       = None   # _chat_orchestrator bound to RouterDecision schema
_sql_writer_structured   = None   # _chat_sql_writer bound to SQLQuery schema
_vector_store            = None
_db_url                  = None
_db_file_path            = None
_known_projects          = []     # cache of {id, name, namespace} — refreshed on init/reindex


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS  (structured output — no regex parsing needed)
# ═════════════════════════════════════════════════════════════════════════════

class RouterDecision(TypedDict):
    """Which of the 2 chat-facing tools should answer this question."""
    tool: Annotated[Literal["Structured_Database_Lookup", "Unstructured_Document_RAG"], "Return Structured_Database_Lookup for data/numbers/counts answerable by SQL. Unstructured_Document_RAG for specs, BRS, conceptual or documentation questions."]
    reasoning: Annotated[str, "One short sentence explaining the choice."]


#from typing import TypedDict, Annotated

class SQLQuery(TypedDict):
    """A single SQLite statement that answers the user's question."""
    
    sql: Annotated[str, "One complete SQLite statement, ending in a semicolon. No markdown, no commentary, just the raw SQL."]
    explanation: Annotated[str,"One short sentence describing what the query does."]

# ═════════════════════════════════════════════════════════════════════════════
# INIT
# ═════════════════════════════════════════════════════════════════════════════

def init_ai(db_url: str):
    global _chat_orchestrator, _chat_sql_writer
    global _router_structured, _sql_writer_structured
    global _vector_store, _db_url, _db_file_path, _known_projects

    _db_url = db_url
    import re as _re
    _db_file_path = _re.sub(r"^sqlite:///+", "/", db_url) if "sqlite" in db_url else db_url
    if not _db_file_path.startswith("/"):
        _db_file_path = "/" + _db_file_path

    api_key = os.getenv("HUGGINGFACE_API_KEY", "")
    if not api_key.startswith("hf_"):
        raise EnvironmentError("HUGGINGFACE_API_KEY missing or invalid in .env")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    os.environ["HUGGINGFACE_API_KEY"]       = api_key

    llm_orchestrator = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        task="text-generation",
        max_new_tokens=1024,
        huggingfacehub_api_token=api_key,
        temperature=0.1,
    )
    _chat_orchestrator = ChatHuggingFace(llm=llm_orchestrator)

    llm_sql_writer = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        task="text-generation",
        max_new_tokens=300,
        huggingfacehub_api_token=api_key,
        temperature=0.0,
    )
    _chat_sql_writer = ChatHuggingFace(llm=llm_sql_writer)

    # Bind structured output schemas — LLM replies are parsed straight into
    # typed Pydantic objects, no manual extraction needed.
    _router_structured     = _chat_orchestrator.with_structured_output(RouterDecision)
    _sql_writer_structured = _chat_sql_writer.with_structured_output(SQLQuery)

    logger.info("[AI] LLMs initialised: deepseek-ai/DeepSeek-V4-Pro (structured output bound)")

    _vector_store = load_index()
    if _vector_store is None:
        logger.info("[AI] No saved index — building now...")
        _vector_store = build_index(db_url)

    _refresh_known_projects()
    logger.info("[AI] Engine ready ✅")


def refresh_index():
    global _vector_store
    if not _db_url:
        raise RuntimeError("init_ai() has not been called yet")
    _vector_store = build_index(_db_url)
    _refresh_known_projects()
    logger.info("[AI] Index refreshed ✅")


def _refresh_known_projects():
    global _known_projects
    try:
        conn   = sqlite3.connect(_db_file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM projects")
        rows   = cursor.fetchall()
        conn.close()
        _known_projects = [
            {"id": r[0], "name": r[1], "namespace": clean_namespace(r[1])}
            for r in rows
        ]
        logger.info(f"[AI] Known projects cached: {[p['name'] for p in _known_projects]}")
    except Exception as e:
        logger.warning(f"[AI] Could not cache project names: {e}")
        _known_projects = []


# ═════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY FORMATTING
# ═════════════════════════════════════════════════════════════════════════════

def _format_chat_history(chat_history: Optional[list[dict]], max_turns: int = 8) -> str:
    """
    Convert a list of {"role": "user"|"assistant", "content": "..."} dicts
    into a plain-text transcript for prompt injection.
    Truncates to the most recent max_turns messages to control token usage.
    Returns "(no prior conversation)" if empty/None.
    """
    if not chat_history:
        return "(no prior conversation)"

    recent = chat_history[-max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if content:
            # Trim very long assistant answers so history doesn't dominate the prompt
            if role == "Assistant" and len(content) > 400:
                content = content[:400] + "..."
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


# ═════════════════════════════════════════════════════════════════════════════
# NAMESPACE / PROJECT RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_namespace(project_id: Optional[int], project_namespace: Optional[str],
                       question: str) -> Optional[str]:
    if project_namespace and project_namespace.strip():
        return clean_namespace(project_namespace)

    if project_id:
        proj = _fetch_project(project_id)
        if proj:
            return clean_namespace(proj["name"])

    q_clean = clean_namespace(question)
    for p in _known_projects:
        if p["namespace"] and p["namespace"] in q_clean:
            logger.info(f"[NS] Dynamic match: '{p['name']}' → '{p['namespace']}'")
            return p["namespace"]

    q_lower = question.lower()
    for p in _known_projects:
        if p["name"].lower() in q_lower:
            logger.info(f"[NS] Name match: '{p['name']}' → '{p['namespace']}'")
            return p["namespace"]

    return None


def _project_id_from_query(question: str) -> Optional[int]:
    q_lower = question.lower()
    for p in _known_projects:
        if p["name"].lower() in q_lower:
            logger.info(f"[NS] Inferred project_id={p['id']} from '{p['name']}'")
            return p["id"]
    return None


# ═════════════════════════════════════════════════════════════════════════════
# GENERAL CHAT  (greetings / capability questions — bypasses tool routing)
# ═════════════════════════════════════════════════════════════════════════════

_GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "what can you do", "what all can you do", "help", "what are you",
    "who are you", "what is this", "how does this work", "thanks", "thank you",
    "bye", "goodbye",
]


def _is_general_chat(question: str) -> bool:
    q = question.lower().strip().rstrip("?!.")
    if q in _GREETING_PATTERNS:
        return True
    if len(q.split()) <= 6 and any(
        p in q for p in ["what can you do", "help me", "who are you", "what are you"]
    ):
        return True
    return False


def _handle_general_chat(question: str, chat_history_text: str) -> dict:
    try:
        prompt = GENERAL_CHAT_PROMPT.format(query=question, chat_history=chat_history_text)
        resp   = _chat_orchestrator.invoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        logger.warning(f"[General chat] LLM failed, using canned reply: {e}")
        answer = (
            "Hi! I can help you with database questions about your projects, tasks "
            "and risks, search project documents, and you can use the dedicated "
            "Summarize, Budget, Risk Radar, Schedule and Portfolio buttons for "
            "structured reports."
        )
    return {"response": answer, "query_type": "qa", "confidence_score": 1.0, "metadata": {}}


# ═════════════════════════════════════════════════════════════════════════════
# SQL DATABASE AGENT  (structured output — no regex extraction)
# ═════════════════════════════════════════════════════════════════════════════

def _sql_agent(question: str, project_name: Optional[str] = None,
              chat_history_text: str = "(no prior conversation)") -> str:
    """
    1. Formulates a SQL prompt based on user question and chat history.
    2. Invokes the structured LLM chain.
    3. Safely extracts the SQL string, handling key mismatches and markdown fences.
    4. Executes query against SQLite DB and synthesizes a human response.
    """
    generated_sql = None
    try:
        # Enrich prompt if a specific project context is active
        enriched_question = question
        if project_name:
            enriched_question = f"{question} (Note: focus on project named '{project_name}'.)"
            
        sql_prompt = SQL_WRITER_PROMPT.format(query=enriched_question, chat_history=chat_history_text)
        parsed = _sql_writer_structured.invoke(sql_prompt)
        
        # --- SAFE PARSING START ---
        # Handles dict vs Pydantic object, key variations, and fallback extractions
        if isinstance(parsed, dict):
            generated_sql = parsed.get('sql') or parsed.get('query') or parsed.get('sql_query')
            if not generated_sql:
                # Fallback: grab the first string value containing standard SQL keywords
                for val in parsed.values():
                    if isinstance(val, str) and ("select" in val.lower() or "from" in val.lower()):
                        generated_sql = val
                        break
        else:
            generated_sql = getattr(parsed, 'sql', None) or str(parsed)

        if not generated_sql or not isinstance(generated_sql, str):
            raise ValueError(f"Could not extract a valid SQL string from response: {parsed}")

        # Clean out any markdown backticks if the LLM output fenced code blocks
        import re
        generated_sql = re.sub(r"```(?:sql)?", "", generated_sql, flags=re.IGNORECASE)
        generated_sql = generated_sql.replace("```", "").strip()

        if not generated_sql.endswith(";"):
            generated_sql += ";"

        explanation = parsed.get('explanation', '') if isinstance(parsed, dict) else getattr(parsed, 'explanation', '')
        logger.info(f"[SQL Agent] Extracted SQL: {generated_sql} | Explanation: {explanation}")
        # --- SAFE PARSING END ---

        # Execute query against SQLite database
        conn = sqlite3.connect(_db_file_path)
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()
        logger.info(f"[SQL Agent] Rows returned: {len(rows)}")

        # Synthesize human-readable answer via LLM
        synthesis_prompt = SQL_SYNTHESIS_PROMPT.format(
            query=question, sql=generated_sql, columns=columns, rows=rows,
        )
        final = _chat_orchestrator.invoke(synthesis_prompt)
        return final.content if hasattr(final, "content") else str(final)

    except sqlite3.OperationalError as e:
        logger.error(f"[SQL Agent] SQLite error: {e} | SQL was: {generated_sql}")
        return f"The generated database query was invalid ({e}). Try rephrasing your question."
    except Exception as e:
        logger.exception(f"[SQL Agent] Unexpected error: {e}")
        return f"I couldn't process that database question right now: {e}"
        
# def _sql_agent(question: str, project_name: Optional[str] = None,
#               chat_history_text: str = "(no prior conversation)") -> str:
#     """
#     A. LLM writes SQL directly as a validated SQLQuery object (with_structured_output)
#     B. Execute against SQLite
#     C. LLM synthesizes human answer
#     """
#     generated_sql = None
#     try:
#         enriched_question = question
#         if project_name:
#             enriched_question = (f"{question} (Note: focus on project named '{project_name}'.)")
#         sql_prompt = SQL_WRITER_PROMPT.format(query=enriched_question, chat_history=chat_history_text)
#         parsed: SQLQuery = _sql_writer_structured.invoke(sql_prompt)
#         generated_sql = parsed['sql'].strip()
#         if not generated_sql.endswith(";"):
#             generated_sql += ";"

#         logger.info(f"[SQL Agent] Structured SQL: {generated_sql} | {parsed['explanation']}")

#         conn   = sqlite3.connect(_db_file_path)
#         cursor = conn.cursor()
#         cursor.execute(generated_sql)
#         rows    = cursor.fetchall()
#         columns = [d[0] for d in cursor.description] if cursor.description else []
#         conn.close()
#         logger.info(f"[SQL Agent] Rows returned: {len(rows)}")

#         synthesis_prompt = SQL_SYNTHESIS_PROMPT.format(
#             query=question, sql=generated_sql, columns=columns, rows=rows,
#         )
#         final = _chat_orchestrator.invoke(synthesis_prompt)
#         return final.content if hasattr(final, "content") else str(final)

#     except sqlite3.OperationalError as e:
#         logger.error(f"[SQL Agent] SQLite error: {e} | SQL was: {generated_sql}")
#         return (f"The generated database query was invalid ({e}). "
#                 f"Try rephrasing your question, e.g. naming the project explicitly.")
#     except Exception as e:
#         logger.exception(f"[SQL Agent] Unexpected error: {e}")
#         return f"I couldn't process that database question right now: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# RAG TOOL
# ═════════════════════════════════════════════════════════════════════════════

def _rag_tool(question: str, namespace: Optional[str] = None,
              chat_history_text: str = "(no prior conversation)") -> str:
    if _vector_store is None:
        return "RAG index not ready — click Re-index DB in the sidebar."

    docs = tiered_search(_vector_store, question, project_namespace=namespace, k=4)
    context = "\n---\n".join(d.page_content for d in docs)
    if not context.strip():
        context = "No matching documentation found in the project database."

    prompt = RAG_PROMPT.format(context=context, query=question, chat_history=chat_history_text)
    response = _chat_orchestrator.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ═════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _engine():
    return create_engine(_db_url)


def _fetch_project(project_id: int) -> Optional[dict]:
    try:
        with _engine().connect() as conn:
            row = conn.execute(text(
                "SELECT id, name, description, owner, status, priority, "
                "progress_percentage, start_date, end_date, budget "
                "FROM projects WHERE id = :pid"
            ), {"pid": project_id}).fetchone()
        if not row:
            return None
        return dict(zip(["id","name","description","owner","status","priority",
                         "progress_percentage","start_date","end_date","budget"], row))
    except Exception as e:
        logger.error(f"[DB] _fetch_project error: {e}")
        return None


def _fetch_tasks(project_id: int) -> list[dict]:
    try:
        with _engine().connect() as conn:
            rows = conn.execute(text(
                "SELECT title, status, priority, assigned_to, "
                "progress_percentage, estimated_hours, actual_hours "
                "FROM tasks WHERE project_id = :pid"
            ), {"pid": project_id}).fetchall()
        return [dict(zip(["title","status","priority","assigned_to",
                          "progress_percentage","estimated_hours","actual_hours"], r))
                for r in rows]
    except Exception as e:
        logger.error(f"[DB] _fetch_tasks error: {e}")
        return []


def _fetch_risks(project_id: int) -> list[dict]:
    try:
        with _engine().connect() as conn:
            rows = conn.execute(text(
                "SELECT title, description, severity, probability, "
                "mitigation_plan, owner, status "
                "FROM risks WHERE project_id = :pid"
            ), {"pid": project_id}).fetchall()
        return [dict(zip(["title","description","severity","probability",
                          "mitigation_plan","owner","status"], r))
                for r in rows]
    except Exception as e:
        logger.error(f"[DB] _fetch_risks error: {e}")
        return []


def _fetch_all_projects() -> list[dict]:
    try:
        with _engine().connect() as conn:
            rows = conn.execute(text(
                "SELECT id, name, status, priority, progress_percentage, budget "
                "FROM projects"
            )).fetchall()
        return [dict(zip(["id","name","status","priority","progress_percentage","budget"], r))
                for r in rows]
    except Exception as e:
        logger.error(f"[DB] _fetch_all_projects error: {e}")
        return []


def _sev_score(s: str) -> int:
    return {"Critical":4,"High":3,"Medium":2,"Low":1}.get(s, 1)


def _strip_reasoning(text_in: str) -> str:
    """Remove any leaked chain-of-thought, keep only the final structured answer."""
    import re as _re
    text_in = _re.sub(r"<think>.*?</think>", "", text_in, flags=_re.DOTALL).strip()
    markers = ["Budget Status:", "Schedule Status:", "RISK RADAR", "Overall verdict:"]
    for marker in markers:
        idx = text_in.find(marker)
        if idx > 20:
            return text_in[idx:].strip()
    return text_in


def _run_chain(prompt_template, variables: dict) -> str:
    try:
        chain   = prompt_template | _chat_orchestrator
        result  = chain.invoke(variables)
        content = result.content.strip() if hasattr(result, "content") else str(result).strip()
        return _strip_reasoning(content)
    except Exception as e:
        logger.error(f"[Chain] Error: {e}")
        return f"AI processing error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# USE-CASE HANDLERS  (UI-button-only — called directly by app.py endpoints)
# ═════════════════════════════════════════════════════════════════════════════

def handle_risk_radar(project_id: int) -> dict:
    project = _fetch_project(project_id)
    if not project:
        return _error(f"Project ID {project_id} not found in database")
    risks = _fetch_risks(project_id)
    if not risks:
        return {"response": f"No risks logged for **{project['name']}** yet. Add risks in the Risks tab.",
                "query_type": "risk", "confidence_score": 1.0,
                "metadata": {"project_name": project["name"], "risk_count": 0}}

    sorted_risks = sorted(
        risks,
        key=lambda r: _sev_score(r["severity"]) + _sev_score(r["probability"]),
        reverse=True
    )
    risk_table = "\n\n".join(
        f"{i}. [{r['severity']} / {r['probability']}] {r['title']}\n"
        f"   Desc: {r['description'] or 'N/A'}\n"
        f"   Mitigation: {r['mitigation_plan'] or 'None'}\n"
        f"   Owner: {r['owner'] or 'Unassigned'} | Status: {r['status']}"
        for i, r in enumerate(sorted_risks, 1)
    )
    response = _run_chain(RISK_RADAR_PROMPT, {
        "project_name":   project["name"],
        "project_status": project["status"],
        "progress":       project["progress_percentage"],
        "budget":         project["budget"] or "Not set",
        "risk_table":     risk_table,
    })
    return {
        "response": response, "query_type": "risk", "confidence_score": 0.92,
        "metadata": {"project_name": project["name"], "risk_count": len(risks),
                     "top_risk": sorted_risks[0]["title"]}
    }


def handle_summarize(project_id: int) -> dict:
    project = _fetch_project(project_id)
    if not project:
        return _error(f"Project ID {project_id} not found in database")
    tasks = _fetch_tasks(project_id)
    risks = _fetch_risks(project_id)
    response = _run_chain(SUMMARIZE_PROMPT, {
        "project_name":   project["name"],
        "owner":          project["owner"] or "N/A",
        "project_status": project["status"],
        "progress":       project["progress_percentage"],
        "budget":         project["budget"] or "Not set",
        "total_tasks":    len(tasks),
        "done_tasks":     sum(1 for t in tasks if t["status"] == "Completed"),
        "blocked_tasks":  sum(1 for t in tasks if t["status"] == "Blocked"),
        "top_risks":      ", ".join(r["title"] for r in risks[:3]) or "None",
        "description":    project["description"] or "N/A",
    })
    return {
        "response": response, "query_type": "summarize", "confidence_score": 0.95,
        "metadata": {"project_name": project["name"],
                     "progress": project["progress_percentage"]}
    }


def handle_budget(project_id: int) -> dict:
    project = _fetch_project(project_id)
    if not project:
        return _error(f"Project ID {project_id} not found in database")
    tasks = _fetch_tasks(project_id)
    response = _run_chain(BUDGET_PROMPT, {
        "project_name":    project["name"],
        "budget":          project["budget"] or "Not set",
        "estimated_hours": round(sum((t["estimated_hours"] or 0) for t in tasks), 1),
        "actual_hours":    round(sum((t["actual_hours"]    or 0) for t in tasks), 1),
        "progress":        project["progress_percentage"],
        "project_status":  project["status"],
    })
    return {
        "response": response, "query_type": "budget", "confidence_score": 0.90,
        "metadata": {"project_name": project["name"], "budget": project["budget"]}
    }


def handle_schedule(project_id: int) -> dict:
    project = _fetch_project(project_id)
    if not project:
        return _error(f"Project ID {project_id} not found in database")
    tasks = _fetch_tasks(project_id)
    response = _run_chain(SCHEDULE_PROMPT, {
        "project_name":  project["name"],
        "start_date":    project["start_date"] or "N/A",
        "end_date":      project["end_date"]   or "N/A",
        "today":         date.today().isoformat(),
        "progress":      project["progress_percentage"],
        "total_tasks":   len(tasks),
        "done_tasks":    sum(1 for t in tasks if t["status"] == "Completed"),
        "blocked_tasks": sum(1 for t in tasks if t["status"] == "Blocked"),
    })
    return {
        "response": response, "query_type": "schedule", "confidence_score": 0.88,
        "metadata": {"project_name": project["name"],
                     "end_date": project["end_date"],
                     "progress": project["progress_percentage"]}
    }


def handle_portfolio() -> dict:
    projects = _fetch_all_projects()
    if not projects:
        return _error("No projects found in the database")
    portfolio_table = "\n".join(
        f"• {p['name']} | {p['status']} | {p['priority']} priority | "
        f"{p['progress_percentage']}% done | Budget: {p['budget'] or 'N/A'}"
        for p in projects
    )
    response = _run_chain(PORTFOLIO_PROMPT, {"portfolio_table": portfolio_table})
    return {
        "response": response, "query_type": "portfolio", "confidence_score": 0.90,
        "metadata": {"project_count": len(projects)}
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER  (LLM-only, structured output, exactly 2 tools)
# ═════════════════════════════════════════════════════════════════════════════

def _llm_route(question: str, chat_history_text: str) -> str:
    """
    Always asks the LLM — no keyword shortcuts. Returns one of:
      Structured_Database_Lookup | Unstructured_Document_RAG
    Falls back to Structured_Database_Lookup only if the LLM call itself fails.
    """
    try:
        prompt   = ROUTER_PROMPT.format(input=question, chat_history=chat_history_text)
        decision: RouterDecision = _router_structured.invoke(prompt)
        logger.info(f"[Router LLM] '{decision['tool']}' — {decision['reasoning']}")
        return decision['tool']
    except Exception as e:
        logger.warning(f"[Router LLM] Structured call failed, defaulting to SQL: {e}")
        return "Structured_Database_Lookup"


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT  (free-text chat — only ever reaches 2 tools + general chat)
# ═════════════════════════════════════════════════════════════════════════════

def process_query(question: str,
                  project_id: Optional[int] = None,
                  project_namespace: Optional[str] = None,
                  chat_history: Optional[list[dict]] = None) -> dict:
    """
    chat_history: list of {"role": "user"|"assistant", "content": "..."} dicts,
    most recent message last. Frontend sends the full session history (minus
    the current question) on every call, until the user clicks Clear chat.
    """
    logger.info(f"[AI] process_query: '{question[:80]}' project_id={project_id} "
               f"history_len={len(chat_history) if chat_history else 0}")

    chat_history_text = _format_chat_history(chat_history)

    # General chat bypass — greetings / capability questions, no LLM tool call
    if _is_general_chat(question):
        logger.info("[AI] → General_Chat (bypassing router)")
        return _handle_general_chat(question, chat_history_text)

    # Resolve project context (explicit param, else name mentioned in query/history)
    effective_project_id = project_id
    if not effective_project_id:
        effective_project_id = _project_id_from_query(question)
        if effective_project_id:
            logger.info(f"[AI] Inferred project_id={effective_project_id} from query")

    resolved_ns = _resolve_namespace(effective_project_id, project_namespace, question)
    logger.info(f"[AI] Resolved namespace: {resolved_ns or 'global'}")

    project_name_hint = None
    if effective_project_id:
        p = _fetch_project(effective_project_id)
        if p:
            project_name_hint = p["name"]

    # Route — always via LLM, only 2 possible outcomes
    tool_choice = _llm_route(question, chat_history_text)
    logger.info(f"[AI] → {tool_choice}")

    try:
        if tool_choice == "Structured_Database_Lookup":
            return {
                "response": _sql_agent(question, project_name_hint, chat_history_text),
                "query_type": "sql", "confidence_score": 0.85,
                "metadata": {"tool": "sql_agent",
                             "project_hint": project_name_hint or "none"}
            }

        # Unstructured_Document_RAG
        return {
            "response": _rag_tool(question, namespace=resolved_ns,
                                  chat_history_text=chat_history_text),
            "query_type": "rag", "confidence_score": 0.82,
            "metadata": {"namespace_used": resolved_ns or "global",
                         "index_meta": get_index_meta()}
        }

    except Exception as e:
        logger.exception(f"[AI] Unhandled error in process_query: {e}")
        return _error(str(e))


def _error(msg: str) -> dict:
    return {
        "response": f"⚠️ {msg}",
        "query_type": "error",
        "confidence_score": 0.0,
        "metadata": {"error": msg}
    }
