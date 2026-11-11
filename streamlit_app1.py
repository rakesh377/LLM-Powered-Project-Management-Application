"""
Project Tracker Pro — Streamlit Frontend v3 (ICICI Design System)
AI: LLM Router → SQL Agent + 3-Tier RAG + Use-Case Handlers
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Tracker Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://localhost:5000/api"

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("user_id", "user_001"),
    ("chat_history", []),
    ("ai_cache", {}),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── ICICI Design System CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&display=swap');

:root {
  --org:    #C8510A;
  --org-dk: #A84008;
  --org-lt: #FEF3EC;
  --org-mid:#E8690A;
  --bg:     #F3F5F8;
  --sf:     #ffffff;
  --bd:     #E4E7EC;
  --bdlt:   #F2F4F7;
  --tx:     #101828;
  --txm:    #344054;
  --txl:    #667085;
  --txd:    #98A2B3;
  --gn:     #027A48; --gnl: #ECFDF3;
  --am:     #B54708; --aml: #FFFAEB;
  --rd:     #B42318; --rdl: #FEF3F2;
  --bl:     #175CD3; --bll: #EFF8FF;
  --pu:     #6941C6; --pul: #F4F3FF;
  --sh: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.1);
  --shm: 0 4px 8px -2px rgba(16,24,40,.1), 0 2px 4px -2px rgba(16,24,40,.06);
  --r: 8px; --rl: 12px;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--tx) !important;
}

/* ── HIDE Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── APP HEADER BAR ── */
.app-header {
  background: var(--org);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 8px rgba(0,0,0,.15);
  margin-bottom: 0;
}
.app-header-brand {
  display: flex; align-items: center; gap: 10px;
}
.app-header-icon {
  width: 32px; height: 32px;
  background: white; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; font-weight: 900; color: var(--org);
}
.app-header-title {
  font-size: 16px; font-weight: 800; color: white; letter-spacing: -0.2px;
}
.app-header-tag {
  font-size: 11px; color: rgba(255,255,255,.7); font-weight: 400;
}
.app-header-right {
  display: flex; align-items: center; gap: 8px;
}
.header-pill {
  background: rgba(255,255,255,.15);
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 20px; padding: 4px 12px;
  color: white; font-size: 11px; font-weight: 600;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: var(--sf) !important;
  border-right: 1px solid var(--bd) !important;
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
  padding-top: 12px !important;
}
.sidebar-label {
  font-size: 9px; font-weight: 700; color: var(--txd);
  letter-spacing: 1.5px; text-transform: uppercase;
  padding: 6px 13px 3px; margin-bottom: 2px;
}
/* Radio nav styling */
div[data-testid="stRadio"] > div {
  gap: 2px !important;
}
div[data-testid="stRadio"] label {
  border-radius: 4px !important;
  padding: 8px 12px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--txl) !important;
  cursor: pointer !important;
  transition: all .12s !important;
  border-left: 3px solid transparent !important;
}
div[data-testid="stRadio"] label:hover {
  background: var(--org-lt) !important;
  color: var(--org) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
  background: var(--org-lt) !important;
  border-left-color: var(--org) !important;
  color: var(--org) !important;
  font-weight: 700 !important;
}

/* ── AI STATUS BADGE ── */
.ai-badge-online {
  background: var(--gnl); color: var(--gn);
  border: 1px solid #A7F3D0; border-radius: 20px;
  padding: 4px 12px; font-size: 12px; font-weight: 700;
  display: inline-block;
}
.ai-badge-offline {
  background: var(--rdl); color: var(--rd);
  border: 1px solid #FECACA; border-radius: 20px;
  padding: 4px 12px; font-size: 12px; font-weight: 700;
  display: inline-block;
}
.ai-meta {
  font-size: 11px; color: var(--txl); margin-top: 4px; line-height: 1.6;
}

/* ── PAGE HEADER ── */
.page-header {
  background: var(--sf);
  border-bottom: 1px solid var(--bd);
  padding: 14px 24px;
  margin-bottom: 20px;
}
.page-title {
  font-size: 18px; font-weight: 800; color: var(--tx);
}
.page-subtitle {
  font-size: 12px; color: var(--txl); margin-top: 2px;
}

/* ── KPI CARDS ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 20px; }
.kpi-card {
  flex: 1; background: var(--sf);
  border: 1px solid var(--bd); border-radius: var(--rl);
  padding: 16px 18px; box-shadow: var(--sh);
  border-top: 3px solid var(--org);
}
.kpi-card.green  { border-top-color: var(--gn); }
.kpi-card.amber  { border-top-color: var(--am); }
.kpi-card.red    { border-top-color: var(--rd); }
.kpi-card.blue   { border-top-color: var(--bl); }
.kpi-value { font-size: 28px; font-weight: 800; color: var(--tx); line-height: 1; }
.kpi-label {
  font-size: 10px; color: var(--txl); margin-top: 4px;
  font-weight: 600; text-transform: uppercase; letter-spacing: .4px;
}

/* ── PROJECT CARD ── */
.proj-card {
  background: var(--sf); border: 1px solid var(--bd);
  border-radius: var(--rl); padding: 16px 20px;
  box-shadow: var(--sh); margin-bottom: 12px;
  border-left: 4px solid var(--org);
  transition: box-shadow .15s;
}
.proj-card:hover { box-shadow: var(--shm); }
.proj-name { font-size: 15px; font-weight: 700; color: var(--tx); }
.proj-desc { font-size: 12px; color: var(--txl); margin-top: 2px; }
.proj-meta { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
.proj-meta-item { font-size: 11px; color: var(--txl); }
.proj-meta-item strong { color: var(--txm); font-weight: 600; }

/* ── BADGES ── */
.badge {
  display: inline-flex; align-items: center; gap: 3px;
  border-radius: 12px; padding: 2px 8px;
  font-size: 10px; font-weight: 700; white-space: nowrap;
}
.badge-green  { background: var(--gnl); color: var(--gn); }
.badge-amber  { background: var(--aml); color: var(--am); }
.badge-red    { background: var(--rdl); color: var(--rd); }
.badge-blue   { background: var(--bll); color: var(--bl); }
.badge-purple { background: var(--pul); color: var(--pu); }
.badge-gray   { background: var(--bdlt); color: var(--txl); }
.badge-orange { background: var(--org-lt); color: var(--org); }

/* ── PROGRESS BAR ── */
.stProgress > div > div { background: var(--org) !important; border-radius: 4px !important; }
.stProgress > div { background: var(--bdlt) !important; border-radius: 4px !important; }

/* ── BUTTONS ── */
.stButton > button {
  background: var(--sf) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 20px !important;
  color: var(--txm) !important;
  font-size: 12px !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  padding: 6px 16px !important;
  transition: all .12s !important;
}
.stButton > button:hover {
  border-color: var(--org) !important;
  color: var(--org) !important;
  background: var(--org-lt) !important;
}
.stButton > button[kind="primary"] {
  background: var(--org) !important;
  color: white !important;
  border: none !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--org-dk) !important;
  color: white !important;
}

/* ── FORM SUBMIT ── */
.stFormSubmitButton > button {
  background: var(--org) !important;
  color: white !important; border: none !important;
  border-radius: 20px !important;
  font-weight: 700 !important; font-size: 13px !important;
  padding: 8px 24px !important;
}
.stFormSubmitButton > button:hover {
  background: var(--org-dk) !important;
}

/* ── INPUTS ── */
.stTextInput input, .stTextArea textarea, .stSelectbox select,
.stNumberInput input {
  border: 1px solid var(--bd) !important;
  border-radius: var(--r) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  color: var(--tx) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--org) !important;
  box-shadow: 0 0 0 2px rgba(200,81,10,.1) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 2px solid var(--bd) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-size: 13px !important; font-weight: 600 !important;
  color: var(--txl) !important; padding: 10px 20px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--org) !important;
  border-bottom-color: var(--org) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: var(--sf); border: 1px solid var(--bd);
  border-radius: var(--rl); padding: 16px 20px;
  box-shadow: var(--sh); border-top: 3px solid var(--org);
}
[data-testid="stMetricLabel"] {
  font-size: 10px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .4px !important;
  color: var(--txl) !important;
}
[data-testid="stMetricValue"] {
  font-size: 28px !important; font-weight: 800 !important;
  color: var(--tx) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--bd) !important;
  border-radius: var(--rl) !important;
  overflow: hidden !important;
  box-shadow: var(--sh) !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
  background: var(--sf) !important;
  border: 1px solid var(--bd) !important;
  border-radius: var(--rl) !important;
  padding: 12px 16px !important;
  margin-bottom: 8px !important;
  box-shadow: var(--sh) !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
  border-left: 3px solid var(--org) !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
  border-left: 3px solid var(--bl) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
  border: 2px solid var(--bd) !important;
  border-radius: 24px !important;
  background: var(--sf) !important;
  padding: 4px 8px !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--org) !important;
  box-shadow: 0 0 0 3px rgba(200,81,10,.08) !important;
}

/* ── AI RESULT BOX ── */
.ai-result-box {
  background: linear-gradient(135deg, #FEF3EC 0%, #FFF8F5 100%);
  border: 1px solid #FECBA1; border-left: 4px solid var(--org);
  border-radius: var(--rl); padding: 14px 18px;
  margin: 10px 0; box-shadow: var(--sh);
}
.ai-result-header {
  font-size: 11px; font-weight: 700; color: var(--org);
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px;
}
.ai-result-body { font-size: 13px; color: var(--txm); line-height: 1.7; }

/* ── CHIP BUTTONS ── */
.chip-btn {
  background: var(--sf); border: 1px solid var(--bd);
  border-radius: 20px; padding: 6px 14px;
  font-size: 11px; font-weight: 600; color: var(--txm);
  cursor: pointer; transition: all .12s;
  font-family: 'DM Sans', sans-serif;
}
.chip-btn:hover {
  border-color: var(--org); color: var(--org); background: var(--org-lt);
}

/* ── SECTION CARD ── */
.section-card {
  background: var(--sf); border: 1px solid var(--bd);
  border-radius: var(--rl); padding: 20px 24px;
  margin-bottom: 16px; box-shadow: var(--sh);
}
.section-title {
  font-size: 13px; font-weight: 700; color: var(--tx);
  margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--bd);
}

/* ── DIVIDER ── */
hr { border-color: var(--bd) !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
  font-size: 12px !important; font-weight: 600 !important;
  color: var(--txm) !important;
  background: var(--bdlt) !important;
  border-radius: var(--r) !important;
}

/* ── SUCCESS / ERROR / INFO ── */
.stSuccess { background: var(--gnl) !important; border-left-color: var(--gn) !important; }
.stError   { background: var(--rdl) !important; border-left-color: var(--rd) !important; }
.stWarning { background: var(--aml) !important; border-left-color: var(--am) !important; }
.stInfo    { background: var(--bll) !important; border-left-color: var(--bl) !important; }

/* ── RISK SEVERITY COLORS ── */
.sev-critical { color: var(--rd); font-weight: 700; }
.sev-high     { color: var(--am); font-weight: 700; }
.sev-medium   { color: #B45309; font-weight: 600; }
.sev-low      { color: var(--gn); font-weight: 600; }

/* ── FOOTER ── */
.app-footer {
  text-align: center; padding: 16px;
  font-size: 10px; color: var(--txd);
  border-top: 1px solid var(--bd); margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── APP HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-header-brand">
    <div class="app-header-icon">PT</div>
    <div>
      <div class="app-header-title">Project Tracker Pro</div>
      <div class="app-header-tag">AI-Powered — Project Management</div>
    </div>
  </div>
  <div class="app-header-right">
    <span class="header-pill">👤 user_001</span>
    <span class="header-pill">v2.0</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# API HELPERS  (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_projects():
    try:
        r = requests.get(f"{API_BASE_URL}/projects", timeout=5)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception as e:
        st.error(f"Cannot reach backend: {e}")
        return []

def get_project_detail(pid):
    try:
        r = requests.get(f"{API_BASE_URL}/projects/{pid}", timeout=5)
        return r.json().get("data") if r.status_code == 200 else None
    except Exception as e:
        st.error(str(e))
        return None

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload, timeout=5)
        data = r.json()
        if r.status_code in (200, 201):
            st.cache_data.clear()
            return True, data
        return False, data.get("error", "Unknown error")
    except Exception as e:
        return False, str(e)

def api_put(endpoint, payload):
    try:
        r = requests.put(f"{API_BASE_URL}/{endpoint}", json=payload, timeout=5)
        data = r.json()
        if r.status_code == 200:
            st.cache_data.clear()
            return True, data
        return False, data.get("error", "Unknown error")
    except Exception as e:
        return False, str(e)

def api_delete(endpoint):
    try:
        r = requests.delete(f"{API_BASE_URL}/{endpoint}", timeout=5)
        data = r.json()
        if r.status_code == 200:
            st.cache_data.clear()
            return True, data.get("message", "Deleted")
        return False, data.get("error", "Unknown error")
    except Exception as e:
        return False, str(e)

def get_ai_status():
    try:
        r = requests.get(f"{API_BASE_URL}/bot/status", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def query_bot(question: str, project_id=None, project_namespace: str = None,
              include_history: bool = True) -> dict | None:
    try:
        chat_history_payload = []
        if include_history:
            # st.session_state.chat_history already has the current user
            # message appended by the caller before this is invoked — exclude
            # it here so we send only what came BEFORE this question.
            full_history = st.session_state.chat_history
            if full_history and full_history[-1].get("role") == "user" \
               and full_history[-1].get("content") == question:
                history_to_send = full_history[:-1]
            else:
                history_to_send = full_history
            chat_history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in history_to_send
            ]
        payload = {
            "question":          question,
            "message":           question,
            "user_id":           st.session_state.user_id,
            "project_id":        project_id,
            "project_namespace": project_namespace,
            "chat_history":      chat_history_payload,
        }
        r = requests.post(f"{API_BASE_URL}/bot/query", json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()["data"]
        elif r.status_code == 503:
            st.warning("⚠️ AI engine not ready. Check backend logs.")
        else:
            st.error(f"AI error {r.status_code}: {r.json().get('error', 'Unknown')}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ LLM timed out (90s). Try a shorter question.")
        return None
    except Exception as e:
        st.error(f"Bot error: {e}")
        return None


def call_ui_tool(endpoint: str, project_id: int = None) -> dict | None:
    """
    Calls one of the 5 dedicated use-case endpoints (Risk Radar, Summarize,
    Budget, Schedule, Portfolio) directly — these bypass the chat router
    entirely and are only reachable via UI buttons.
    """
    try:
        url = f"{API_BASE_URL}/bot/{endpoint}"
        if project_id is not None:
            url = f"{url}/{project_id}"
        r = requests.post(url, timeout=90)
        if r.status_code == 200:
            return r.json()["data"]
        elif r.status_code == 503:
            st.warning("⚠️ AI engine not ready. Check backend logs.")
        else:
            st.error(f"AI error {r.status_code}: {r.json().get('error', 'Unknown')}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ LLM timed out (90s).")
        return None
    except Exception as e:
        st.error(f"Bot error: {e}")
        return None

def trigger_reindex():
    try:
        r = requests.post(f"{API_BASE_URL}/bot/reindex", timeout=120)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_chat_history_from_api(project_id=None, limit=20):
    try:
        params = {"limit": limit}
        if project_id:
            params["project_id"] = project_id
        r = requests.get(f"{API_BASE_URL}/bot/history", params=params, timeout=5)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception:
        return []


# ── UI helpers ─────────────────────────────────────────────────────────────────
_QUERY_ICONS = {
    "risk": "⚠️", "summarize": "✨", "budget": "💰",
    "schedule": "📅", "portfolio": "📊", "sql": "🗃️",
    "rag": "📄", "qa": "🤖", "error": "❌",
}

def badge(text, color="orange"):
    return f'<span class="badge badge-{color}">{text}</span>'

def render_ai_result(result: dict, collapsed: bool = False):
    if not result:
        return
    qt   = result.get("query_type", "ai")
    icon = _QUERY_ICONS.get(qt, "🤖")
    conf = result.get("confidence_score", result.get("confidence", 0))
    response = result.get("response", "No response")
    st.markdown(f"""
    <div class="ai-result-box">
      <div class="ai-result-header">{icon} AI · {qt.replace('_',' ').title()} · {conf:.0%} confidence</div>
      <div class="ai-result-body">{response}</div>
    </div>
    """, unsafe_allow_html=True)
    meta = result.get("metadata", {})
    if meta:
        with st.expander("Details", expanded=False):
            st.json(meta)

def render_ai_buttons(project: dict) -> tuple:
    """
    These 4 buttons call the dedicated use-case endpoints directly
    (handle_summarize, handle_budget, handle_risk_radar, handle_schedule) —
    they never go through the chat router. Returns (result_dict, project_id)
    so the caller can render it; question/result handling stays local here.
    """
    pid  = project["id"]
    c1, c2, c3, c4 = st.columns(4)
    result = None
    with c1:
        if st.button("✨ Summarize", key=f"sum_{pid}", use_container_width=True):
            with st.spinner("🤖 Summarizing..."):
                result = call_ui_tool("summarize", pid)
    with c2:
        if st.button("💰 Budget", key=f"bud_{pid}", use_container_width=True):
            with st.spinner("🤖 Analysing budget..."):
                result = call_ui_tool("budget", pid)
    with c3:
        if st.button("⚠️ Risk radar", key=f"rsk_{pid}", use_container_width=True):
            with st.spinner("🤖 Running risk radar..."):
                result = call_ui_tool("risk_radar", pid)
    with c4:
        if st.button("📅 Schedule", key=f"sch_{pid}", use_container_width=True):
            with st.spinner("🤖 Checking schedule..."):
                result = call_ui_tool("schedule", pid)
    return result, pid

def status_badge(status: str) -> str:
    color_map = {
        "Completed": "green", "In-Progress": "blue",
        "Planning": "gray",   "On-Hold": "amber",
        "Blocked": "red",
    }
    return badge(status, color_map.get(status, "gray"))

def priority_badge(priority: str) -> str:
    color_map = {
        "Critical": "red", "High": "amber",
        "Medium": "blue",  "Low": "green",
    }
    return badge(priority, color_map.get(priority, "gray"))

def severity_badge(sev: str) -> str:
    color_map = {
        "Critical": "red", "High": "amber",
        "Medium": "blue",  "Low": "green",
    }
    return badge(sev, color_map.get(sev, "gray"))


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
    selected = st.radio(
        "Go to",
        ["🏠 Dashboard", "📁 Projects", "✅ Tasks",
         "⚠️ Risks", "📄 Documents", "🤖 AI Bot", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown('<div class="sidebar-label">AI Engine</div>', unsafe_allow_html=True)
    ai_st = get_ai_status()
    if ai_st.get("ai_ready"):
        st.markdown('<span class="ai-badge-online">Online ✅</span>', unsafe_allow_html=True)
        meta = ai_st.get("index_meta", {})
        st.markdown(f"""<div class="ai-meta">
          📚 {meta.get('document_count','?')} docs
          ({meta.get('db_docs','?')} DB + {meta.get('file_docs','?')} files)<br>
          🕐 {str(meta.get('last_indexed','N/A'))[:16]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<span class="ai-badge-offline">Offline ❌</span>', unsafe_allow_html=True)
        err = ai_st.get("ai_error", "Backend not reachable")
        if err:
            st.caption(str(err)[:100])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Re-index DB", use_container_width=True):
        with st.spinner("Rebuilding FAISS index..."):
            res = trigger_reindex()
        if res.get("success"):
            st.success(f"Done — {res['meta']['document_count']} docs")
        else:
            st.error(res.get("error", "Failed"))

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
if "Dashboard" in selected:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">📊 Dashboard Overview</div>
      <div class="page-subtitle">Portfolio-level view of all projects and AI insights</div>
    </div>
    """, unsafe_allow_html=True)

    projects = get_projects()
    st.markdown('<div style="padding: 0 24px;">', unsafe_allow_html=True)

    if projects:
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Projects", len(projects))
        with m2: st.metric("Active", sum(1 for p in projects if p["status"] in ["Planning","In-Progress"]))
        with m3: st.metric("Completed", sum(1 for p in projects if p["status"] == "Completed"))
        with m4:
            avg = sum(p.get("progress_percentage", 0) for p in projects) / len(projects)
            st.metric("Avg Progress", f"{avg:.0f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 AI Portfolio Briefing", type="primary"):
            with st.spinner("🤖 Analysing portfolio..."):
                result = call_ui_tool("portfolio")
            if result:
                render_ai_result(result)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            df = pd.DataFrame(projects)
            color_map = {
                "In-Progress": "#C8510A", "Planning": "#175CD3",
                "Completed": "#027A48",   "On-Hold": "#B54708",
            }
            fig = px.bar(
                df, x="name", y="progress_percentage", color="status",
                title="Progress by Project",
                labels={"name": "Project", "progress_percentage": "Progress %"},
                color_discrete_map=color_map,
            )
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                font_family="DM Sans",
                title_font_size=13, title_font_color="#101828",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            pc = pd.DataFrame(projects)["priority"].value_counts()
            fig2 = px.pie(
                values=pc.values, names=pc.index,
                title="Projects by Priority",
                color_discrete_sequence=["#C8510A","#B54708","#175CD3","#027A48"],
            )
            fig2.update_layout(
                font_family="DM Sans",
                title_font_size=13, title_font_color="#101828",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-card"><div class="section-title">All Projects</div>', unsafe_allow_html=True)
        df2 = pd.DataFrame(projects)[["name","status","priority","progress_percentage","owner","budget"]]
        df2.columns = ["Name","Status","Priority","Progress %","Owner","Budget"]
        st.dataframe(df2, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No projects found. Create one from the Projects page.")

    st.markdown('</div>', unsafe_allow_html=True)


# ── PROJECTS ──────────────────────────────────────────────────────────────────
elif "Projects" in selected:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">📁 Projects</div>
      <div class="page-subtitle">Manage and analyse all your projects</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 24px;">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["View Projects", "Create Project"])

    with tab1:
        projects = get_projects()
        if not projects:
            st.info("No projects yet.")
        else:
            st.markdown('<div class="section-card"><div class="section-title">💡 Quick AI Questions</div>', unsafe_allow_html=True)
            qc = st.columns(4)
            chips = [
                "Which projects are at risk?",
                "Who is overloaded with tasks?",
                "Which project has the highest budget?",
                "What tasks are blocked?",
            ]
            for i, chip_q in enumerate(chips):
                with qc[i]:
                    if st.button(chip_q, key=f"chip_{i}", use_container_width=True):
                        with st.spinner("🤖 Thinking..."):
                            r = query_bot(chip_q)
                        if r:
                            render_ai_result(r)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for proj in projects:
                pid = proj["id"]
                status_color = {
                    "Completed":"#027A48","In-Progress":"#C8510A",
                    "Planning":"#175CD3","On-Hold":"#B54708","Blocked":"#B42318"
                }.get(proj["status"],"#667085")

                st.markdown(f"""
                <div class="proj-card" style="border-left-color:{status_color}">
                  <div class="proj-name">{proj["name"]}</div>
                  <div class="proj-desc">{proj.get("description","")}</div>
                  <div class="proj-meta">
                    <div class="proj-meta-item">Status: <strong>{proj["status"]}</strong></div>
                    <div class="proj-meta-item">Priority: <strong>{proj["priority"]}</strong></div>
                    <div class="proj-meta-item">Owner: <strong>{proj.get("owner","N/A")}</strong></div>
                    <div class="proj-meta-item">Budget: <strong>₹{proj.get("budget","N/A")}</strong></div>
                    <div class="proj-meta-item">Tasks: <strong>{proj.get("task_count",0)}</strong></div>
                    <div class="proj-meta-item">Risks: <strong>{proj.get("risk_count",0)}</strong></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                prog = proj.get("progress_percentage", 0)
                st.progress(prog / 100, text=f"{prog}% complete")

                # ── Action row: AI buttons + Edit + Delete ────────────────
                button_result, _ = render_ai_buttons(proj)
                if button_result:
                    import time as _time
                    ck = f"{pid}_{button_result.get('query_type','ai')}_{int(_time.time())}"
                    st.session_state.ai_cache[ck] = button_result

                proj_cache = {k: v for k, v in st.session_state.ai_cache.items()
                              if k.startswith(f"{pid}_")}
                for ck, cv in proj_cache.items():
                    render_ai_result(cv)
                    if st.button("✖ Clear", key=f"clr_{ck}"):
                        del st.session_state.ai_cache[ck]
                        st.rerun()
                    break

                # ── Edit expander ─────────────────────────────────────────
                with st.expander(f"✏️ Edit — {proj['name']}", expanded=False):
                    st.markdown("**Edit Project Details**")
                    with st.form(f"edit_proj_{pid}"):
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            e_name   = st.text_input("Name", value=proj["name"], key=f"en_{pid}")
                            e_owner  = st.text_input("Owner", value=proj.get("owner",""), key=f"eo_{pid}")
                            e_status = st.selectbox("Status",
                                ["Planning","In-Progress","On-Hold","Completed","Blocked"],
                                index=["Planning","In-Progress","On-Hold","Completed","Blocked"]
                                    .index(proj["status"]) if proj["status"] in
                                    ["Planning","In-Progress","On-Hold","Completed","Blocked"] else 0,
                                key=f"est_{pid}")
                        with ec2:
                            e_priority = st.selectbox("Priority",
                                ["Low","Medium","High","Critical"],
                                index=["Low","Medium","High","Critical"]
                                    .index(proj["priority"]) if proj["priority"] in
                                    ["Low","Medium","High","Critical"] else 1,
                                key=f"ep_{pid}")
                            e_progress = st.slider("Progress %", 0, 100,
                                proj.get("progress_percentage", 0), key=f"eprog_{pid}")
                            e_budget = st.number_input("Budget (₹)", min_value=0.0,
                                value=float(proj.get("budget") or 0), step=10000.0,
                                key=f"eb_{pid}")
                        e_desc = st.text_area("Description",
                            value=proj.get("description",""), key=f"ed_{pid}")
                        if st.form_submit_button("💾 Save Project Changes"):
                            ok, resp = api_put(f"projects/{pid}", {
                                "name": e_name, "owner": e_owner, "status": e_status,
                                "priority": e_priority, "progress_percentage": e_progress,
                                "budget": e_budget if e_budget > 0 else None,
                                "description": e_desc,
                            })
                            if ok:
                                st.success("✅ Project updated!")
                                st.rerun()
                            else:
                                st.error(str(resp))

                    # ── Add Task inside edit expander ─────────────────────
                    st.markdown("---")
                    st.markdown("**➕ Add Task to this Project**")
                    with st.form(f"add_task_{pid}"):
                        at_title    = st.text_input("Task Title *", key=f"att_{pid}")
                        atc1, atc2  = st.columns(2)
                        with atc1:
                            at_assigned = st.text_input("Assigned To", key=f"ata_{pid}")
                            at_status   = st.selectbox("Status",
                                ["To-Do","In-Progress","In-Review","Completed","Blocked"],
                                key=f"ats_{pid}")
                        with atc2:
                            at_priority = st.selectbox("Priority",
                                ["Low","Medium","High","Critical"], index=1,
                                key=f"atp_{pid}")
                            at_est_h    = st.number_input("Estimated Hours",
                                min_value=0.0, step=0.5, key=f"ateh_{pid}")
                        at_desc = st.text_area("Description", key=f"atd_{pid}")
                        if st.form_submit_button("✅ Add Task"):
                            if not at_title:
                                st.error("Task title required.")
                            else:
                                ok, data = api_post(f"projects/{pid}/tasks", {
                                    "title": at_title, "description": at_desc,
                                    "assigned_to": at_assigned, "status": at_status,
                                    "priority": at_priority, "estimated_hours": at_est_h,
                                })
                                if ok:
                                    st.success("✅ Task added!")
                                    st.rerun()
                                else:
                                    st.error(str(data))

                    # ── Add Risk inside edit expander ─────────────────────
                    st.markdown("---")
                    st.markdown("**➕ Add Risk to this Project**")
                    with st.form(f"add_risk_{pid}"):
                        ar_title   = st.text_input("Risk Title *", key=f"art_{pid}")
                        arc1, arc2 = st.columns(2)
                        with arc1:
                            ar_sev   = st.selectbox("Severity",
                                ["Low","Medium","High","Critical"], index=1,
                                key=f"ars_{pid}")
                            ar_prob  = st.selectbox("Probability",
                                ["Low","Medium","High"], key=f"arp_{pid}")
                        with arc2:
                            ar_owner  = st.text_input("Risk Owner", key=f"aro_{pid}")
                            ar_status = st.selectbox("Risk Status",
                                ["Open","Mitigated","Closed"], key=f"arst_{pid}")
                        ar_desc = st.text_area("Description", key=f"ard_{pid}")
                        ar_mit  = st.text_area("Mitigation Plan", key=f"arm_{pid}")
                        if st.form_submit_button("✅ Add Risk"):
                            if not ar_title:
                                st.error("Risk title required.")
                            else:
                                ok, data = api_post(f"projects/{pid}/risks", {
                                    "title": ar_title, "description": ar_desc,
                                    "severity": ar_sev, "probability": ar_prob,
                                    "owner": ar_owner, "status": ar_status,
                                    "mitigation_plan": ar_mit,
                                })
                                if ok:
                                    st.success("✅ Risk added!")
                                    st.rerun()
                                else:
                                    st.error(str(data))

                # ── Delete button ─────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"🗑️ Delete project '{proj['name']}' and all its data",
                             key=f"del_proj_{pid}", type="secondary"):
                    st.session_state[f"confirm_del_proj_{pid}"] = True

                if st.session_state.get(f"confirm_del_proj_{pid}"):
                    st.warning(
                        f"⚠️ This will permanently delete **{proj['name']}** "
                        f"including all its tasks, risks and chat history. Are you sure?"
                    )
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("✅ Yes, delete permanently",
                                     key=f"confirm_yes_{pid}", type="primary"):
                            ok, msg = api_delete(f"projects/{pid}")
                            st.session_state.pop(f"confirm_del_proj_{pid}", None)
                            if ok:
                                st.success(f"Project deleted. Re-index RAG if needed.")
                                st.rerun()
                            else:
                                st.error(str(msg))
                    with cc2:
                        if st.button("❌ Cancel", key=f"confirm_no_{pid}"):
                            st.session_state.pop(f"confirm_del_proj_{pid}", None)
                            st.rerun()

                st.markdown("---")

    with tab2:
        st.markdown('<div class="section-card"><div class="section-title">Create New Project</div>', unsafe_allow_html=True)
        with st.form("create_project_form"):
            name = st.text_input("Project Name *")
            desc = st.text_area("Description")
            c1, c2 = st.columns(2)
            with c1:
                owner    = st.text_input("Owner")
                priority = st.selectbox("Priority", ["Low","Medium","High","Critical"])
            with c2:
                start_date = st.date_input("Start Date")
                end_date   = st.date_input("End Date")
            budget = st.number_input("Budget (₹)", min_value=0.0, step=10000.0)
            if st.form_submit_button("✅ Create Project"):
                if not name:
                    st.error("Project name is required.")
                else:
                    ok, data = api_post("projects", {
                        "name": name, "description": desc, "owner": owner,
                        "priority": priority, "budget": budget,
                        "start_date": start_date.isoformat(),
                        "end_date":   end_date.isoformat(),
                    })
                    if ok:
                        st.success("✅ Project created!")
                        st.rerun()
                    else:
                        st.error(str(data))
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── TASKS ─────────────────────────────────────────────────────────────────────
elif "Tasks" in selected:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">✅ Tasks</div>
      <div class="page-subtitle">Track and manage tasks across projects</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 24px;">', unsafe_allow_html=True)
    projects = get_projects()
    if not projects:
        st.info("Create a project first.")
    else:
        sel_proj = st.selectbox("Project", options=projects,
                                format_func=lambda x: x["name"],
                                key="task_proj_sel")
        tab1, tab2 = st.tabs(["View Tasks", "Create Task"])

        with tab1:
            if st.button("🤖 What's blocking this project?", key="task_ai_btn", type="primary"):
                with st.spinner("🤖 Analysing..."):
                    r = query_bot(f"What tasks are blocked in {sel_proj['name']}?", sel_proj["id"])
                if r:
                    render_ai_result(r)

            proj = get_project_detail(sel_proj["id"])
            if proj and proj.get("tasks"):
                tasks = proj["tasks"]
                sf, pf = st.columns(2)
                with sf:
                    status_f = st.multiselect("Filter by status",
                        ["To-Do","In-Progress","In-Review","Completed","Blocked"],
                        default=["To-Do","In-Progress","Blocked"])
                with pf:
                    pri_f = st.multiselect("Filter by priority",
                        ["Low","Medium","High","Critical"],
                        default=["Medium","High","Critical"])
                if status_f: tasks = [t for t in tasks if t["status"] in status_f]
                if pri_f:    tasks = [t for t in tasks if t["priority"] in pri_f]

                for task in tasks:
                    tid = task["id"]
                    sev_icon = {"Blocked":"🔴","In-Progress":"🟡","Completed":"🟢","To-Do":"⚪"}.get(task["status"],"⚪")
                    st.markdown(f"""
                    <div class="proj-card" style="border-left-color:#175CD3; margin-bottom:8px;">
                      <div class="proj-name">{sev_icon} {task["title"]}</div>
                      <div class="proj-desc">{task.get("description","")}</div>
                      <div class="proj-meta">
                        <div class="proj-meta-item">Status: <strong>{task["status"]}</strong></div>
                        <div class="proj-meta-item">Priority: <strong>{task["priority"]}</strong></div>
                        <div class="proj-meta-item">Assigned: <strong>{task.get("assigned_to","—")}</strong></div>
                        <div class="proj-meta-item">Progress: <strong>{task["progress_percentage"]}%</strong></div>
                        <div class="proj-meta-item">Est: <strong>{task.get("estimated_hours","—")}h</strong></div>
                        <div class="proj-meta-item">Actual: <strong>{task.get("actual_hours","—")}h</strong></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(task["progress_percentage"] / 100)

                    # ── Edit expander per task ────────────────────────────
                    with st.expander(f"✏️ Edit task", expanded=False):
                        with st.form(f"edit_task_{tid}"):
                            et_title = st.text_input("Title", value=task["title"], key=f"ett_{tid}")
                            etc1, etc2 = st.columns(2)
                            with etc1:
                                et_status = st.selectbox("Status",
                                    ["To-Do","In-Progress","In-Review","Completed","Blocked"],
                                    index=["To-Do","In-Progress","In-Review","Completed","Blocked"]
                                          .index(task["status"])
                                          if task["status"] in ["To-Do","In-Progress","In-Review","Completed","Blocked"] else 0,
                                    key=f"ets_{tid}")
                                et_priority = st.selectbox("Priority",
                                    ["Low","Medium","High","Critical"],
                                    index=["Low","Medium","High","Critical"]
                                          .index(task["priority"])
                                          if task["priority"] in ["Low","Medium","High","Critical"] else 1,
                                    key=f"etp_{tid}")
                                et_assigned = st.text_input("Assigned To",
                                    value=task.get("assigned_to",""), key=f"eta_{tid}")
                            with etc2:
                                et_progress = st.slider("Progress %", 0, 100,
                                    task.get("progress_percentage", 0), key=f"etprog_{tid}")
                                et_est  = st.number_input("Estimated Hours", min_value=0.0,
                                    value=float(task.get("estimated_hours") or 0), step=0.5,
                                    key=f"eteh_{tid}")
                                et_act  = st.number_input("Actual Hours", min_value=0.0,
                                    value=float(task.get("actual_hours") or 0), step=0.5,
                                    key=f"etah_{tid}")
                            et_desc = st.text_area("Description",
                                value=task.get("description",""), key=f"etd_{tid}")
                            if st.form_submit_button("💾 Save Task Changes"):
                                ok, resp = api_put(f"tasks/{tid}", {
                                    "title": et_title, "status": et_status,
                                    "priority": et_priority, "assigned_to": et_assigned,
                                    "progress_percentage": et_progress,
                                    "estimated_hours": et_est if et_est > 0 else None,
                                    "actual_hours":    et_act if et_act > 0 else None,
                                    "description": et_desc,
                                })
                                if ok:
                                    st.success("✅ Task updated!")
                                    st.rerun()
                                else:
                                    st.error(str(resp))

                    # ── Delete task ───────────────────────────────────────
                    if st.button(f"🗑️ Delete task", key=f"del_task_{tid}", type="secondary"):
                        ok, msg = api_delete(f"tasks/{tid}")
                        if ok:
                            st.success("Task deleted.")
                            st.rerun()
                        else:
                            st.error(str(msg))

                    st.markdown("---")
            else:
                st.info("No tasks yet for this project.")

        with tab2:
            st.markdown('<div class="section-card"><div class="section-title">Create Task</div>', unsafe_allow_html=True)
            with st.form("create_task_form"):
                title = st.text_input("Task Title *")
                desc  = st.text_area("Description")
                c1, c2 = st.columns(2)
                with c1:
                    assigned = st.text_input("Assigned To")
                    status   = st.selectbox("Status", ["To-Do","In-Progress","In-Review","Completed"])
                with c2:
                    priority = st.selectbox("Priority", ["Low","Medium","High","Critical"])
                    est_h    = st.number_input("Estimated Hours", min_value=0.0, step=0.5)
                if st.form_submit_button("✅ Create Task"):
                    if not title:
                        st.error("Title required.")
                    else:
                        ok, data = api_post(f"projects/{sel_proj['id']}/tasks", {
                            "title": title, "description": desc, "assigned_to": assigned,
                            "status": status, "priority": priority, "estimated_hours": est_h,
                        })
                        if ok:
                            st.success("✅ Task created!")
                            st.rerun()
                        else:
                            st.error(str(data))
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── RISKS ─────────────────────────────────────────────────────────────────────
elif "Risks" in selected:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">⚠️ Risk Management</div>
      <div class="page-subtitle">Identify, assess and mitigate project risks</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 24px;">', unsafe_allow_html=True)
    projects = get_projects()
    if not projects:
        st.info("Create a project first.")
    else:
        sel_proj = st.selectbox("Project", options=projects,
                                format_func=lambda x: x["name"],
                                key="risk_proj_sel")
        tab1, tab2 = st.tabs(["View Risks", "Log Risk"])

        with tab1:
            if st.button("⚠️ Run AI Risk Radar", key="risk_ai_btn", type="primary"):
                with st.spinner("🤖 Analysing risks..."):
                    r = call_ui_tool("risk_radar", sel_proj["id"])
                if r:
                    render_ai_result(r)

            proj = get_project_detail(sel_proj["id"])
            if proj and proj.get("risks"):
                risks = proj["risks"]
                rm1, rm2, rm3 = st.columns(3)
                with rm1: st.metric("Total Risks", len(risks))
                with rm2: st.metric("Critical", sum(1 for r in risks if r["severity"]=="Critical"))
                with rm3: st.metric("High", sum(1 for r in risks if r["severity"]=="High"))

                sc = pd.Series([r["severity"] for r in risks]).value_counts()
                fig = px.bar(
                    sc, x=sc.index, y=sc.values,
                    title="Risks by Severity",
                    labels={"x":"Severity","y":"Count"},
                    color=sc.index,
                    color_discrete_map={"Critical":"#B42318","High":"#B54708","Medium":"#175CD3","Low":"#027A48"},
                )
                fig.update_layout(font_family="DM Sans", showlegend=False,
                                  plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

                for risk in risks:
                    rid = risk["id"]
                    sev_color = {"Critical":"#B42318","High":"#B54708","Medium":"#175CD3","Low":"#027A48"}.get(risk["severity"],"#667085")
                    sev_icon  = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}.get(risk["severity"],"⚪")
                    st.markdown(f"""
                    <div class="proj-card" style="border-left-color:{sev_color}; margin-bottom:8px;">
                      <div class="proj-name">{sev_icon} {risk["title"]}</div>
                      <div class="proj-desc">{risk.get("description","")}</div>
                      <div class="proj-meta">
                        <div class="proj-meta-item">Severity: <strong style="color:{sev_color}">{risk["severity"]}</strong></div>
                        <div class="proj-meta-item">Probability: <strong>{risk["probability"]}</strong></div>
                        <div class="proj-meta-item">Status: <strong>{risk["status"]}</strong></div>
                        <div class="proj-meta-item">Owner: <strong>{risk.get("owner","—")}</strong></div>
                      </div>
                      {"<div class='proj-desc' style='margin-top:8px;'>🛡️ <strong>Mitigation:</strong> " + risk["mitigation_plan"] + "</div>" if risk.get("mitigation_plan") else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Edit expander per risk ────────────────────────────
                    with st.expander("✏️ Edit risk", expanded=False):
                        with st.form(f"edit_risk_{rid}"):
                            er_title = st.text_input("Title", value=risk["title"], key=f"ert_{rid}")
                            erc1, erc2 = st.columns(2)
                            with erc1:
                                er_sev = st.selectbox("Severity",
                                    ["Low","Medium","High","Critical"],
                                    index=["Low","Medium","High","Critical"]
                                          .index(risk["severity"])
                                          if risk["severity"] in ["Low","Medium","High","Critical"] else 1,
                                    key=f"ers_{rid}")
                                er_prob = st.selectbox("Probability",
                                    ["Low","Medium","High"],
                                    index=["Low","Medium","High"]
                                          .index(risk["probability"])
                                          if risk["probability"] in ["Low","Medium","High"] else 0,
                                    key=f"erp_{rid}")
                            with erc2:
                                er_owner = st.text_input("Risk Owner",
                                    value=risk.get("owner",""), key=f"ero_{rid}")
                                er_status = st.selectbox("Risk Status",
                                    ["Open","Mitigated","Closed"],
                                    index=["Open","Mitigated","Closed"]
                                          .index(risk["status"])
                                          if risk["status"] in ["Open","Mitigated","Closed"] else 0,
                                    key=f"erst_{rid}")
                            er_desc = st.text_area("Description",
                                value=risk.get("description",""), key=f"erd_{rid}")
                            er_mit  = st.text_area("Mitigation Plan",
                                value=risk.get("mitigation_plan",""), key=f"erm_{rid}")
                            if st.form_submit_button("💾 Save Risk Changes"):
                                ok, resp = api_put(f"risks/{rid}", {
                                    "title": er_title, "severity": er_sev,
                                    "probability": er_prob, "owner": er_owner,
                                    "status": er_status, "description": er_desc,
                                    "mitigation_plan": er_mit,
                                })
                                if ok:
                                    st.success("✅ Risk updated!")
                                    st.rerun()
                                else:
                                    st.error(str(resp))

                    # ── Delete risk ───────────────────────────────────────
                    if st.button("🗑️ Delete risk", key=f"del_risk_{rid}", type="secondary"):
                        ok, msg = api_delete(f"risks/{rid}")
                        if ok:
                            st.success("Risk deleted.")
                            st.rerun()
                        else:
                            st.error(str(msg))

                    st.markdown("---")
            else:
                st.info("No risks logged yet.")

        with tab2:
            st.markdown('<div class="section-card"><div class="section-title">Log New Risk</div>', unsafe_allow_html=True)
            with st.form("create_risk_form"):
                title = st.text_input("Risk Title *")
                desc  = st.text_area("Description")
                c1, c2 = st.columns(2)
                with c1:
                    severity    = st.selectbox("Severity", ["Low","Medium","High","Critical"])
                    probability = st.selectbox("Probability", ["Low","Medium","High"])
                with c2:
                    owner  = st.text_input("Risk Owner")
                    status = st.selectbox("Risk Status", ["Open","Mitigated","Closed"])
                mitigation = st.text_area("Mitigation Plan")
                if st.form_submit_button("✅ Log Risk"):
                    if not title:
                        st.error("Title required.")
                    else:
                        ok, data = api_post(f"projects/{sel_proj['id']}/risks", {
                            "title": title, "description": desc,
                            "severity": severity, "probability": probability,
                            "owner": owner, "status": status,
                            "mitigation_plan": mitigation,
                        })
                        if ok:
                            st.success("✅ Risk logged!")
                            st.rerun()
                        else:
                            st.error(str(data))
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── DOCUMENTS ─────────────────────────────────────────────────────────────────
elif "Documents" in selected:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">📄 Project Documents & RAG Search</div>
      <div class="page-subtitle">Search BRS, specs and documentation using AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 24px;">', unsafe_allow_html=True)
    st.info("Drop `.txt` or `*_BRS*` files into `documents/<project_name_lowercase>/` on the server, "
            "then click **Re-index DB** in the sidebar. The AI will automatically search these files "
            "when you ask document-related questions.")

    st.markdown('<div class="section-card"><div class="section-title">🔍 Try a Document Search</div>', unsafe_allow_html=True)
    projects = get_projects()
    doc_proj = st.selectbox("Scope to project (optional)",
                            [{"id": None, "name": "All projects"}] + projects,
                            format_func=lambda x: x["name"],
                            key="doc_proj_sel")
    doc_q = st.text_input("Search your documents...",
                          placeholder="What are the CSS requirements in the BRS?")
    if st.button("🔍 Search documents", type="primary") and doc_q:
        ns = None
        if doc_proj["id"]:
            p = next((p for p in projects if p["id"] == doc_proj["id"]), None)
            if p:
                ns = p["name"].lower().replace(" ","_")
        with st.spinner("Searching documents via RAG..."):
            result = query_bot(doc_q, doc_proj["id"], project_namespace=ns)
        if result:
            render_ai_result(result)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── AI BOT ────────────────────────────────────────────────────────────────────
elif "AI Bot" in selected:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🤖 AI Project Assistant</div>
      <div class="page-subtitle">Powered by: LLM Router → SQL Agent + 3-Tier RAG · DeepSeek-V4-Pro · FAISS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 24px;">', unsafe_allow_html=True)
    projects = get_projects()
    sel_proj_id   = None
    sel_namespace = None

    if projects:
        sel_proj = st.selectbox(
            "Project context",
            [{"id": None, "name": "General — all projects"}] + projects,
            format_func=lambda x: x["name"],
            key="bot_proj_sel",
        )
        sel_proj_id   = sel_proj["id"]
        if sel_proj_id:
            sel_namespace = sel_proj["name"].lower().replace(" ", "_")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Suggested questions ────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">💡 Suggested Questions</div>', unsafe_allow_html=True)
    sugg_cols = st.columns(3)

    # (label, mode, payload) — mode "tool" calls a dedicated UI-only endpoint
    # directly (bypasses the chat router entirely); mode "chat" sends free
    # text through the normal 2-tool router (Structured_Database_Lookup or
    # Unstructured_Document_RAG), exactly like typing it into the box.
    suggestions = [
        ("📊 Portfolio status",  "tool", "portfolio"),
        ("⚠️ Riskiest projects", "chat", "Which projects have the highest risks?"),
        ("💰 Budget overview",   "chat", "Show me the budget of all projects"),
        ("📅 Behind schedule?",  "chat", "Which projects are behind schedule?"),
        ("👤 Team workload",     "chat", "Who is assigned the most tasks across projects?"),
        ("✨ Project summary",   "tool", "summarize"),
    ]
    for i, (label, mode, payload) in enumerate(suggestions):
        with sugg_cols[i % 3]:
            disabled = (mode == "tool" and payload == "summarize" and not sel_proj_id)
            if st.button(label, key=f"sug_{i}", use_container_width=True, disabled=disabled):
                if mode == "tool":
                    display_q = ("Portfolio summary of all projects" if payload == "portfolio"
                                 else f"Summarize {sel_proj['name']}")
                    st.session_state.chat_history.append({"role":"user","content":display_q})
                    with st.spinner("🤖 Thinking..."):
                        if payload == "portfolio":
                            result = call_ui_tool("portfolio")
                        else:
                            result = call_ui_tool("summarize", sel_proj_id)
                else:
                    enriched_q = payload
                    if sel_proj_id and sel_proj:
                        proj_name = sel_proj["name"]
                        for phrase in ["this project", "the project", "current project"]:
                            if phrase in payload.lower():
                                enriched_q = payload.lower().replace(phrase, proj_name)
                                break
                    st.session_state.chat_history.append({"role":"user","content":payload})
                    with st.spinner("🤖 Thinking..."):
                        result = query_bot(enriched_q, sel_proj_id, sel_namespace)

                if result:
                    st.session_state.chat_history.append({
                        "role":"assistant",
                        "content":    result.get("response",""),
                        "query_type": result.get("query_type",""),
                        "confidence": result.get("confidence_score", result.get("confidence",0)),
                        "metadata":   result.get("metadata",{}),
                    })
                st.rerun()
    if not sel_proj_id:
        st.caption("💡 Select a project above to enable the Project summary button.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat thread ────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                qt   = msg.get("query_type","")
                conf = msg.get("confidence", 0)
                icon = _QUERY_ICONS.get(qt, "🤖")
                st.write(msg["content"])
                st.markdown(
                    f'<span class="badge badge-orange">{icon} {qt.replace("_"," ").title()}</span> '
                    f'<span style="font-size:11px;color:var(--txl);">· {conf:.0%} confidence</span>',
                    unsafe_allow_html=True
                )
                meta = msg.get("metadata", {})
                if meta:
                    with st.expander("Details"):
                        st.json(meta)

    # ── Input ──────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your projects...")
    if user_input:
        enriched_input = user_input
        if sel_proj_id and sel_proj:
            proj_name = sel_proj["name"]
            for phrase in ["this project", "the project", "current project"]:
                if phrase in user_input.lower():
                    enriched_input = user_input.lower().replace(phrase, proj_name)
                    break

        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.spinner("🤖 Routing and thinking..."):
            result = query_bot(enriched_input, sel_proj_id, sel_namespace)
        if result:
            st.session_state.chat_history.append({
                "role":"assistant",
                "content":    result.get("response",""),
                "query_type": result.get("query_type",""),
                "confidence": result.get("confidence_score", result.get("confidence",0)),
                "metadata":   result.get("metadata",{}),
            })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ── SETTINGS ──────────────────────────────────────────────────────────────────
elif "Settings" in selected:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">⚙️ Settings</div>
      <div class="page-subtitle">System configuration and diagnostics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 24px;">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card"><div class="section-title">User Profile</div>', unsafe_allow_html=True)
        uid = st.text_input("User ID", value=st.session_state.user_id)
        st.session_state.user_id = uid
        st.text_input("Backend URL", value=API_BASE_URL, disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><div class="section-title">AI Engine Status</div>', unsafe_allow_html=True)
        ai_st = get_ai_status()
        st.json(ai_st)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Re-index Database</div>', unsafe_allow_html=True)
    st.write("Run this after adding new projects, tasks, risks, or document files.")
    if st.button("🔄 Re-index now", type="primary"):
        with st.spinner("Rebuilding FAISS index..."):
            res = trigger_reindex()
        if res.get("success"):
            st.success(f"✅ Done — {res['meta']['document_count']} documents indexed")
        else:
            st.error(res.get("error","Failed"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Cache</div>', unsafe_allow_html=True)
    if st.button("Clear data cache"):
        st.cache_data.clear()
        st.session_state.ai_cache = {}
        st.success("✅ Cache cleared!")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    **Project Tracker Pro v2.0**

    Stack: Flask · Streamlit · LangChain · HuggingFace · FAISS · DeepSeek-V4-Pro

    AI Architecture: LLM Router → SQL Agent (LLM writes SQL → executes → synthesizes)
    + 3-Tier RAG (Namespace → Keyword → Global) + Use-case handlers
    (Summarize · Budget · Schedule · Risk Radar · Portfolio)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  Project Tracker Pro v2.0 &nbsp;·&nbsp; AI-Powered by DeepSeek + FAISS + LangChain
  &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
