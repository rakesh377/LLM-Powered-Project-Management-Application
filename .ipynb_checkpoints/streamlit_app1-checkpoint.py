"""
Project Tracker Pro — Streamlit Frontend v2
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
    ("ai_cache", {}),         # {cache_key: result_dict}
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# API HELPERS
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


def get_ai_status():
    try:
        r = requests.get(f"{API_BASE_URL}/bot/status", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def query_bot(question: str, project_id=None, project_namespace: str = None) -> dict | None:
    """Call the AI backend. Returns result dict or None."""
    try:
        payload = {
            "question":          question,
            "message":           question,   # legacy key
            "user_id":           st.session_state.user_id,
            "project_id":        project_id,
            "project_namespace": project_namespace,
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
        st.error("⏱️ LLM timed out (90 s). Try a shorter question.")
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


# ── UI helpers ────────────────────────────────────────────────────────────────

_QUERY_ICONS = {
    "risk": "⚠️", "summarize": "✨", "budget": "💰",
    "schedule": "📅", "portfolio": "📊", "sql": "🗃️",
    "rag": "📄", "qa": "🤖", "error": "❌",
}

def render_ai_result(result: dict, collapsed: bool = False):
    """Render an AI result dict with icon, response text, and metadata expander."""
    if not result:
        return
    qt   = result.get("query_type", "ai")
    icon = _QUERY_ICONS.get(qt, "🤖")
    conf = result.get("confidence_score", result.get("confidence", 0))
    with st.container():
        st.markdown(f"**{icon} AI — {qt.replace('_',' ').title()}**  "
                    f"<span style='font-size:11px; color:gray;'>confidence {conf:.0%}</span>",
                    unsafe_allow_html=True)
        st.write(result.get("response", "No response"))
        meta = result.get("metadata", {})
        if meta:
            with st.expander("Details", expanded=False):
                st.json(meta)


def render_ai_buttons(project: dict) -> tuple:
    """4-button AI row for a project card. Returns (question, project_id) or (None, None)."""
    pid  = project["id"]
    name = project["name"]
    c1, c2, c3, c4 = st.columns(4)
    question = None
    with c1:
        if st.button("✨ Summarize", key=f"sum_{pid}", use_container_width=True):
            question = f"Summarize the {name} project"
    with c2:
        if st.button("💰 Budget", key=f"bud_{pid}", use_container_width=True):
            question = f"Budget analysis for {name}"
    with c3:
        if st.button("⚠️ Risk radar", key=f"rsk_{pid}", use_container_width=True):
            question = f"Risk radar for {name}"
    with c4:
        if st.button("📅 Schedule", key=f"sch_{pid}", use_container_width=True):
            question = f"Schedule health of {name}"
    return question, pid


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

hc1, hc2, hc3 = st.columns([1, 3, 1])
with hc2:
    st.title("📊 Project Tracker Pro")
    st.caption("AI-Powered — LLM Router · SQL Agent · RAG · DeepSeek")
with hc3:
    st.markdown(f"**User:** {st.session_state.user_id}")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("Navigation")
    selected = st.radio(
        "Go to",
        ["🏠 Dashboard", "📁 Projects", "✅ Tasks",
         "⚠️ Risks", "📄 Documents", "🤖 AI Bot", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.divider()

    # ── AI engine status widget ───────────────────────────────────────────
    st.markdown("**🤖 AI Engine**")
    ai_st = get_ai_status()
    if ai_st.get("ai_ready"):
        st.success("Online ✅")
        meta = ai_st.get("index_meta", {})
        st.caption(f"Index: {meta.get('document_count','?')} docs "
                   f"({meta.get('db_docs','?')} DB + {meta.get('file_docs','?')} files)")
        st.caption(f"Last indexed: {str(meta.get('last_indexed','N/A'))[:16]}")
        if st.button("🔄 Re-index DB", use_container_width=True):
            with st.spinner("Rebuilding FAISS index..."):
                res = trigger_reindex()
            if res.get("success"):
                st.success(f"Done — {res['meta']['document_count']} docs")
            else:
                st.error(res.get("error", "Failed"))
    else:
        st.error("Offline ❌")
        err = ai_st.get("ai_error", "Backend not reachable")
        st.caption(str(err)[:100] if err else "")


# ═════════════════════════════════════════════════════════════════════════════
# PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ── DASHBOARD ────────────────────────────────────────────────────────────────
if "Dashboard" in selected:
    st.header("Dashboard Overview")
    projects = get_projects()

    if projects:
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Projects", len(projects))
        with m2: st.metric("Active", sum(1 for p in projects if p["status"] in ["Planning","In-Progress"]))
        with m3: st.metric("Completed", sum(1 for p in projects if p["status"] == "Completed"))
        with m4:
            avg = sum(p.get("progress_percentage", 0) for p in projects) / len(projects)
            st.metric("Avg Progress", f"{avg:.0f}%")

        st.divider()

        # ── AI Portfolio Summary ─────────────────────────────────────────
        if st.button("📊 AI Portfolio Summary", use_container_width=False):
            with st.spinner("🤖 Analysing portfolio..."):
                result = query_bot("Portfolio summary of all projects")
            if result:
                render_ai_result(result)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            df = pd.DataFrame(projects)
            fig = px.bar(df, x="name", y="progress_percentage", color="status",
                         title="Progress by Status",
                         labels={"name": "Project", "progress_percentage": "Progress %"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pc = pd.DataFrame(projects)["priority"].value_counts()
            fig = px.pie(values=pc.values, names=pc.index, title="By Priority")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Projects Table")
        df2 = pd.DataFrame(projects)[["name","status","priority","progress_percentage","owner","budget"]]
        st.dataframe(df2, use_container_width=True, hide_index=True)
    else:
        st.info("No projects found. Create one from the Projects page.")


# ── PROJECTS ─────────────────────────────────────────────────────────────────
elif "Projects" in selected:
    st.header("Projects Management")
    tab1, tab2 = st.tabs(["View Projects", "Create Project"])

    with tab1:
        projects = get_projects()
        if not projects:
            st.info("No projects yet.")
        else:
            # ── Quick question chips ─────────────────────────────────────
            st.markdown("**💡 Quick AI questions**")
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

            st.divider()

            for proj in projects:
                with st.container():
                    ca, cb = st.columns([4, 1])
                    with ca:
                        st.subheader(proj["name"])
                        st.caption(proj.get("description", ""))
                        r1, r2, r3 = st.columns(3)
                        with r1: st.caption(f"Status: **{proj['status']}**")
                        with r2: st.caption(f"Priority: **{proj['priority']}**")
                        with r3: st.caption(f"Owner: {proj.get('owner','N/A')}")
                        prog = proj.get("progress_percentage", 0)
                        st.progress(prog / 100, text=f"{prog}% complete")
                        tc, rc = proj.get("task_count",0), proj.get("risk_count",0)
                        st.caption(f"Tasks: {tc} · Risks: {rc} · Budget: {proj.get('budget','N/A')}")
                    with cb:
                        st.caption(" ")  # vertical alignment spacer

                    # ── 4 AI buttons ─────────────────────────────────────
                    q, pid = render_ai_buttons(proj)
                    if q:
                        ck = f"{pid}_{q}"
                        with st.spinner("🤖 Analysing..."):
                            result = query_bot(q, pid)
                        if result:
                            st.session_state.ai_cache[ck] = result

                    # Show cached result for this project
                    proj_cache = {k: v for k, v in st.session_state.ai_cache.items()
                                  if k.startswith(f"{proj['id']}_")}
                    for ck, cv in proj_cache.items():
                        render_ai_result(cv)
                        if st.button("✖ Clear result", key=f"clr_{ck}"):
                            del st.session_state.ai_cache[ck]
                            st.rerun()
                        break  # show only latest

    with tab2:
        st.subheader("Create New Project")
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
                        st.success("Project created!")
                        st.rerun()
                    else:
                        st.error(str(data))


# ── TASKS ─────────────────────────────────────────────────────────────────────
elif "Tasks" in selected:
    st.header("Tasks Management")
    projects = get_projects()
    if not projects:
        st.info("Create a project first.")
    else:
        sel_proj = st.selectbox("Project", options=projects,
                                format_func=lambda x: x["name"],
                                key="task_proj_sel")
        tab1, tab2 = st.tabs(["View Tasks", "Create Task"])

        with tab1:
            # ── AI: what's blocking ──────────────────────────────────────
            if st.button("🤖 What's blocking this project?", key="task_ai_btn"):
                with st.spinner("🤖 Analysing..."):
                    r = query_bot(f"What tasks are blocked in {sel_proj['name']}?",
                                  sel_proj["id"])
                if r:
                    render_ai_result(r)

            proj = get_project_detail(sel_proj["id"])
            if proj and proj.get("tasks"):
                tasks = proj["tasks"]
                sf, pf = st.columns(2)
                with sf:
                    status_f = st.multiselect(
                        "Filter by status",
                        ["To-Do","In-Progress","In-Review","Completed","Blocked"],
                        default=["To-Do","In-Progress","Blocked"])
                with pf:
                    pri_f = st.multiselect(
                        "Filter by priority",
                        ["Low","Medium","High","Critical"],
                        default=["Medium","High","Critical"])
                if status_f: tasks = [t for t in tasks if t["status"] in status_f]
                if pri_f:    tasks = [t for t in tasks if t["priority"] in pri_f]

                for task in tasks:
                    with st.container():
                        tc1, tc2, tc3 = st.columns([3, 1, 1])
                        with tc1:
                            st.subheader(task["title"])
                            st.caption(task.get("description",""))
                            ta, tb, tc_, td = st.columns(4)
                            with ta: st.caption(f"Status: **{task['status']}**")
                            with tb: st.caption(f"Priority: {task['priority']}")
                            with tc_: st.caption(f"Assigned: {task.get('assigned_to','—')}")
                            with td: st.caption(f"Progress: {task['progress_percentage']}%")
                            st.progress(task["progress_percentage"] / 100)
                        with tc2:
                            new_st = st.selectbox(
                                "Status",
                                ["To-Do","In-Progress","In-Review","Completed","Blocked"],
                                index=["To-Do","In-Progress","In-Review","Completed","Blocked"]
                                      .index(task["status"])
                                      if task["status"] in ["To-Do","In-Progress",
                                                            "In-Review","Completed","Blocked"]
                                      else 0,
                                key=f"ts_{task['id']}")
                        with tc3:
                            st.caption(" ")
                            if st.button("💾 Save", key=f"tsave_{task['id']}"):
                                ok, _ = api_put(f"tasks/{task['id']}", {"status": new_st})
                                if ok: st.success("Updated!"); st.rerun()
            else:
                st.info("No tasks yet for this project.")

        with tab2:
            st.subheader("Create Task")
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
                        if ok: st.success("Task created!"); st.rerun()
                        else:  st.error(str(data))


# ── RISKS ─────────────────────────────────────────────────────────────────────
elif "Risks" in selected:
    st.header("Risk Management")
    projects = get_projects()
    if not projects:
        st.info("Create a project first.")
    else:
        sel_proj = st.selectbox("Project", options=projects,
                                format_func=lambda x: x["name"],
                                key="risk_proj_sel")
        tab1, tab2 = st.tabs(["View Risks", "Log Risk"])

        with tab1:
            if st.button("⚠️ Run AI Risk Radar", key="risk_ai_btn"):
                with st.spinner("🤖 Analysing risks..."):
                    r = query_bot(f"Risk radar for {sel_proj['name']}", sel_proj["id"])
                if r:
                    render_ai_result(r)
            st.divider()

            proj = get_project_detail(sel_proj["id"])
            if proj and proj.get("risks"):
                risks = proj["risks"]
                rm1, rm2, rm3 = st.columns(3)
                with rm1: st.metric("Total Risks", len(risks))
                with rm2: st.metric("Critical", sum(1 for r in risks if r["severity"]=="Critical"))
                with rm3: st.metric("High", sum(1 for r in risks if r["severity"]=="High"))

                sc = pd.Series([r["severity"] for r in risks]).value_counts()
                fig = px.bar(sc, x=sc.index, y=sc.values,
                             title="Risks by Severity",
                             labels={"x":"Severity","y":"Count"})
                st.plotly_chart(fig, use_container_width=True)
                st.divider()

                for risk in risks:
                    sev_icon = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}.get(risk["severity"],"⚪")
                    with st.container():
                        st.subheader(f"{sev_icon} {risk['title']}")
                        st.caption(risk.get("description",""))
                        ra, rb, rc_ = st.columns(3)
                        with ra: st.caption(f"Severity: **{risk['severity']}**")
                        with rb: st.caption(f"Probability: {risk['probability']}")
                        with rc_: st.caption(f"Status: {risk['status']}")
                        if risk.get("mitigation_plan"):
                            st.caption(f"**Mitigation:** {risk['mitigation_plan']}")
                        if risk.get("owner"):
                            st.caption(f"**Owner:** {risk['owner']}")
            else:
                st.info("No risks logged yet.")

        with tab2:
            st.subheader("Log New Risk")
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
                        if ok: st.success("Risk logged!"); st.rerun()
                        else:  st.error(str(data))


# ── DOCUMENTS ─────────────────────────────────────────────────────────────────
elif "Documents" in selected:
    st.header("📄 Project Documents & RAG Search")

    st.info(
        "Drop `.txt` or `*_BRS*` files into "
        "`documents/<project_name_lowercase>/` on the server, "
        "then click **Re-index DB** in the sidebar. "
        "The AI will automatically search these files when you ask document-related questions."
    )

    st.subheader("Try a document search")
    projects = get_projects()
    doc_proj = st.selectbox("Scope to project (optional)",
                            [{"id": None, "name": "All projects"}] + projects,
                            format_func=lambda x: x["name"],
                            key="doc_proj_sel")
    doc_q = st.text_input("Search your documents...",
                          placeholder="What are the CSS requirements in the BRS?")
    if st.button("🔍 Search documents") and doc_q:
        ns = None
        if doc_proj["id"]:
            p = next((p for p in projects if p["id"] == doc_proj["id"]), None)
            if p:
                ns = p["name"].lower().replace(" ","_")
        with st.spinner("Searching documents via RAG..."):
            result = query_bot(doc_q, doc_proj["id"], project_namespace=ns)
        if result:
            render_ai_result(result)


# ── AI BOT (dedicated chat page) ──────────────────────────────────────────────
elif "AI Bot" in selected:
    st.header("🤖 AI Project Assistant")
    st.caption("Powered by: LLM Router → SQL Agent + 3-Tier RAG · DeepSeek-V4-Pro · FAISS")

    projects = get_projects()
    sel_proj_id = None
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

    st.divider()

    # ── Suggested questions ──────────────────────────────────────────────
    st.markdown("**💡 Suggested questions**")
    sugg_cols = st.columns(3)
    suggestions = [
        ("📊 Portfolio status",   "Portfolio summary of all projects"),
        ("⚠️ Riskiest projects",  "Which projects have the highest risks?"),
        ("💰 Budget overview",    "Show me the budget of all projects"),
        ("📅 Behind schedule?",   "Which projects are behind schedule?"),
        ("👤 Team workload",      "Who is assigned the most tasks across projects?"),
        ("✨ Project summary",    f"Summarize project {sel_proj_id}" if sel_proj_id
                                   else "Summarize all projects"),
    ]
    for i, (label, q_text) in enumerate(suggestions):
        with sugg_cols[i % 3]:
            if st.button(label, key=f"sug_{i}", use_container_width=True):
                enriched_q = q_text
                if sel_proj_id and sel_proj:
                    proj_name = sel_proj["name"]
                    for phrase in ["this project", "the project", "current project"]:
                        if phrase in q_text.lower():
                            enriched_q = q_text.lower().replace(phrase, proj_name)
                            break
                st.session_state.chat_history.append({"role":"user","content":q_text})
                with st.spinner("🤖 Thinking..."):
                    result = query_bot(enriched_q, sel_proj_id, sel_namespace)

                    
                if result:
                    st.session_state.chat_history.append({
                        "role":"assistant",
                        "content":    result.get("response",""),
                        "query_type": result.get("query_type",""),
                        "confidence": result.get("confidence_score",
                                                 result.get("confidence",0)),
                        "metadata":   result.get("metadata",{}),
                    })
                st.rerun()

    st.divider()

    # ── Chat thread ───────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                qt   = msg.get("query_type","")
                conf = msg.get("confidence", 0)
                icon = _QUERY_ICONS.get(qt, "🤖")
                st.caption(f"{icon} {qt.replace('_',' ').title()} · confidence {conf:.0%}")
                meta = msg.get("metadata", {})
                if meta:
                    with st.expander("Details"):
                        st.json(meta)

    # ── Input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your projects...")
    if user_input:
    # Enrich vague "this project" references with the actual selected project name
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
                "confidence": result.get("confidence_score",
                                         result.get("confidence",0)),
                "metadata":   result.get("metadata",{}),
            })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── SETTINGS ──────────────────────────────────────────────────────────────────
elif "Settings" in selected:
    st.header("⚙️ Settings")

    st.subheader("User Profile")
    uid = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = uid

    st.subheader("API")
    st.text_input("Backend URL", value=API_BASE_URL, disabled=True)

    st.subheader("AI Engine Status")
    ai_st = get_ai_status()
    st.json(ai_st)

    st.subheader("Re-index Database")
    st.write("Run this after adding new projects, tasks, risks, or document files.")
    if st.button("🔄 Re-index now"):
        with st.spinner("Rebuilding FAISS index..."):
            res = trigger_reindex()
        if res.get("success"):
            st.success(f"Done — {res['meta']['document_count']} documents indexed")
        else:
            st.error(res.get("error","Failed"))

    st.subheader("Cache")
    if st.button("Clear data cache"):
        st.cache_data.clear()
        st.session_state.ai_cache = {}
        st.success("Cache cleared!")

    st.divider()
    st.subheader("About")
    st.write("**Project Tracker Pro v2.0**")
    st.write("Stack: Flask · Streamlit · LangChain · HuggingFace · FAISS · DeepSeek-V4-Pro")
    st.write("AI: LLM Router → SQL Agent (LLM writes SQL → executes → synthesizes) "
             "+ 3-Tier RAG (Namespace → Keyword → Global) + Use-case handlers")
