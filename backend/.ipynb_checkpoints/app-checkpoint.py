"""
Project Tracker Pro — Flask Backend v2
AI: LLM Router → SQL Agent + 3-Tier RAG + Use-Case Handlers
Integrated with a transparent DNS-over-HTTPS (DoH) monkeypatch for Hugging Face DNS resolution.
"""

import os
import sys
import json
import socket
import ssl
import urllib.request
import logging
from pathlib import Path
from datetime import datetime

# ── TRANSPARENT DNS-OVER-HTTPS (DoH) MONKEYPATCH ─────────────────────────────
# This intercepts standard system socket calls to bypass local DNS resolution 
# failures (Errno 8) specifically for Hugging Face endpoints using Cloudflare DNS.
orig_getaddrinfo = socket.getaddrinfo
dns_cache = {}

def doh_resolve(hostname):
    if hostname in dns_cache:
        return dns_cache[hostname]
    try:
        # Resolve via Cloudflare's secure DNS-over-HTTPS JSON API
        url = f"https://1.1.1.1/dns-query?name={hostname}&type=A"
        req = urllib.request.Request(url, headers={"Accept": "application/dns-json"})
        
        # Disable SSL validation temporarily for the DoH query if local proxies are in place
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
            data = json.loads(response.read().decode())
            ips = []
            if "Answer" in data:
                for ans in data["Answer"]:
                    if ans["type"] == 1:  # A record
                        ips.append(ans["data"])
            if ips:
                dns_cache[hostname] = ips
                return ips
    except Exception as e:
        pass
    return []

def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host and "huggingface" in host:
        ips = doh_resolve(host)
        if ips:
            res = []
            for ip in ips:
                try:
                    res.append((socket.AF_INET, type, proto, '', (ip, port)))
                except Exception:
                    pass
            if res:
                return res
    return orig_getaddrinfo(host, port, family, type, proto, flags)

# Apply monkeypatch before importing any HTTP client libraries
socket.getaddrinfo = custom_getaddrinfo

# ── STANDARD BACKEND CONFIGURATION ───────────────────────────────────────────
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── .env ─────────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent.parent / "config" / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    logger.info(f"✅ .env loaded from {env_path}")
else:
    logger.warning(f"⚠️  .env not found at {env_path}")

database_url = os.getenv("DATABASE_URL")
if not database_url:
    home = Path.home()
    database_url = f"sqlite:///{home}/Downloads/ProjectTracker/database/project_tracker.db"
    logger.warning(f"DATABASE_URL not set — using fallback: {database_url}")

hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
logger.info("✅ HF key loaded" if hf_key.startswith("hf_") else "⚠️  HF key missing")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JSON_SORT_KEYS"] = False
db = SQLAlchemy(app)
CORS(app)

ai_ready = False
ai_error_message = ""


def _init_ai():
    global ai_ready, ai_error_message
    try:
        from ai_engine import init_ai
        init_ai(database_url)
        ai_ready = True
        logger.info("✅ AI engine ready")
    except Exception as e:
        ai_error_message = str(e)
        logger.error(f"❌ AI engine failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MODELS
# ═════════════════════════════════════════════════════════════════════════════

class Project(db.Model):
    __tablename__ = "projects"
    id                  = db.Column(db.Integer, primary_key=True)
    name                = db.Column(db.String(100), unique=True, nullable=False)
    description         = db.Column(db.Text)
    owner               = db.Column(db.String(100))
    status              = db.Column(db.String(50), default="Planning")
    priority            = db.Column(db.String(50), default="Medium")
    progress_percentage = db.Column(db.Integer, default=0)
    start_date          = db.Column(db.DateTime, default=datetime.now)
    end_date            = db.Column(db.DateTime)
    budget              = db.Column(db.Float)
    created_at          = db.Column(db.DateTime, default=datetime.now)
    tasks  = db.relationship("Task", backref="project", lazy=True, cascade="all, delete-orphan")
    risks  = db.relationship("Risk", backref="project", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "owner": self.owner, "status": self.status, "priority": self.priority,
            "progress_percentage": self.progress_percentage,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date":   self.end_date.isoformat()   if self.end_date   else None,
            "budget": self.budget,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "task_count": len(self.tasks), "risk_count": len(self.risks),
            "tasks": [t.to_dict() for t in self.tasks],
            "risks": [r.to_dict() for r in self.risks],
        }


class Task(db.Model):
    __tablename__ = "tasks"
    id                  = db.Column(db.Integer, primary_key=True)
    project_id          = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    title               = db.Column(db.String(200), nullable=False)
    description         = db.Column(db.Text)
    status              = db.Column(db.String(50), default="To-Do")
    priority            = db.Column(db.String(50), default="Medium")
    assigned_to         = db.Column(db.String(100))
    progress_percentage = db.Column(db.Integer, default=0)
    estimated_hours     = db.Column(db.Float)
    actual_hours        = db.Column(db.Float)
    created_at          = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id, "project_id": self.project_id,
            "title": self.title, "description": self.description,
            "status": self.status, "priority": self.priority,
            "assigned_to": self.assigned_to,
            "progress_percentage": self.progress_percentage,
            "estimated_hours": self.estimated_hours, "actual_hours": self.actual_hours,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Risk(db.Model):
    __tablename__ = "risks"
    id              = db.Column(db.Integer, primary_key=True)
    project_id      = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    title           = db.Column(db.String(200), nullable=False)
    description     = db.Column(db.Text)
    severity        = db.Column(db.String(50), default="Medium")
    probability     = db.Column(db.String(50), default="Medium")
    mitigation_plan = db.Column(db.Text)
    owner           = db.Column(db.String(100))
    status          = db.Column(db.String(50), default="Open")
    created_at      = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id, "project_id": self.project_id,
            "title": self.title, "description": self.description,
            "severity": self.severity, "probability": self.probability,
            "mitigation_plan": self.mitigation_plan, "owner": self.owner,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatHistory(db.Model):
    __tablename__ = "chat_history"
    id                 = db.Column(db.Integer, primary_key=True)
    user_id            = db.Column(db.String(100))
    user_message       = db.Column(db.Text)
    bot_response       = db.Column(db.Text)
    project_context_id = db.Column(db.Integer, db.ForeignKey("projects.id"))
    query_type         = db.Column(db.String(50))
    confidence_score   = db.Column(db.Float)
    created_at         = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id, "user_id": self.user_id,
            "user_message": self.user_message, "bot_response": self.bot_response,
            "project_context_id": self.project_context_id,
            "query_type": self.query_type, "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ═════════════════════════════════════════════════════════════════════════════
# CRUD ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ai_ready": ai_ready,
                    "ai_error": ai_error_message if not ai_ready else None}), 200


@app.route("/api/projects", methods=["GET"])
def get_projects():
    try:
        projects = Project.query.all()
        return jsonify({"data": [p.to_dict() for p in projects],
                        "success": True, "total": len(projects)}), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/projects", methods=["POST"])
def create_project():
    try:
        data = request.get_json()
        if not data.get("name"):
            return jsonify({"error": "Project name required", "success": False}), 400
        if Project.query.filter_by(name=data["name"]).first():
            return jsonify({"error": "Project name already exists", "success": False}), 400
        project = Project(
            name=data["name"], description=data.get("description", ""),
            owner=data.get("owner", ""), status=data.get("status", "Planning"),
            priority=data.get("priority", "Medium"),
            progress_percentage=data.get("progress_percentage", 0),
            budget=data.get("budget"),
        )
        db.session.add(project)
        db.session.commit()
        return jsonify({"data": project.to_dict(), "success": True,
                        "message": "Project created"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/projects/<int:project_id>", methods=["GET"])
def get_project(project_id):
    try:
        project = Project.query.get_or_404(project_id)
        return jsonify({"data": project.to_dict(), "success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 404


@app.route("/api/projects/<int:project_id>", methods=["PUT"])
def update_project(project_id):
    try:
        project = Project.query.get_or_404(project_id)
        data = request.get_json()
        for field in ["name","description","owner","status","priority",
                      "progress_percentage","budget"]:
            if field in data:
                setattr(project, field, data[field])
        db.session.commit()
        return jsonify({"data": project.to_dict(), "success": True,
                        "message": "Project updated"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/projects/<int:project_id>", methods=["DELETE"])
def delete_project(project_id):
    try:
        project = Project.query.get_or_404(project_id)
        db.session.delete(project)
        db.session.commit()
        return jsonify({"success": True, "message": "Project deleted"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    try:
        project_id = request.args.get("project_id")
        tasks = Task.query.filter_by(project_id=project_id).all() if project_id \
                else Task.query.all()
        return jsonify({"data": [t.to_dict() for t in tasks],
                        "success": True, "total": len(tasks)}), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/tasks/<int:task_id>", methods=["PUT"])
def update_task(task_id):
    try:
        task = Task.query.get_or_404(task_id)
        data = request.get_json()
        for field in ["title","description","status","priority","assigned_to",
                      "progress_percentage","estimated_hours","actual_hours"]:
            if field in data:
                setattr(task, field, data[field])
        db.session.commit()
        return jsonify({"data": task.to_dict(), "success": True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/projects/<int:project_id>/tasks", methods=["POST"])
def create_task(project_id):
    try:
        Project.query.get_or_404(project_id)
        data = request.get_json()
        if not data.get("title"):
            return jsonify({"error": "Task title required", "success": False}), 400
        task = Task(
            project_id=project_id, title=data["title"],
            description=data.get("description", ""), status=data.get("status", "To-Do"),
            priority=data.get("priority", "Medium"),
            assigned_to=data.get("assigned_to", ""),
            progress_percentage=data.get("progress_percentage", 0),
            estimated_hours=data.get("estimated_hours"),
        )
        db.session.add(task)
        db.session.commit()
        return jsonify({"data": task.to_dict(), "success": True,
                        "message": "Task created"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/risks", methods=["GET"])
def get_risks():
    try:
        project_id = request.args.get("project_id")
        risks = Risk.query.filter_by(project_id=project_id).all() if project_id \
                else Risk.query.all()
        return jsonify({"data": [r.to_dict() for r in risks],
                        "success": True, "total": len(risks)}), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/projects/<int:project_id>/risks", methods=["POST"])
def create_risk(project_id):
    try:
        Project.query.get_or_404(project_id)
        data = request.get_json()
        if not data.get("title"):
            return jsonify({"error": "Risk title required", "success": False}), 400
        risk = Risk(
            project_id=project_id, title=data["title"],
            description=data.get("description", ""),
            severity=data.get("severity", "Medium"),
            probability=data.get("probability", "Medium"),
            mitigation_plan=data.get("mitigation_plan", ""),
            owner=data.get("owner", ""), status=data.get("status", "Open"),
        )
        db.session.add(risk)
        db.session.commit()
        return jsonify({"data": risk.to_dict(), "success": True,
                        "message": "Risk created"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e), "success": False}), 500


# ═════════════════════════════════════════════════════════════════════════════
# AI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/bot/query", methods=["POST"])
def bot_query():
    """
    Main AI endpoint.
    Payload: { "question": "...", "project_id": 3, "project_namespace": "pcidss" }
    """
    if not ai_ready:
        return jsonify({"error": f"AI not ready: {ai_error_message}",
                        "success": False}), 503
    try:
        data       = request.get_json()
        question   = (data.get("question") or data.get("message", "")).strip()
        project_id = data.get("project_id")
        namespace  = data.get("project_namespace")
        user_id    = data.get("user_id", "web_user")

        if not question:
            return jsonify({"error": "question is required", "success": False}), 400

        from ai_engine import process_query
        project_namespace = None
        if project_id:
            from sqlalchemy import text as sa_text
            with db.engine.connect() as conn:
                row = conn.execute(sa_text("SELECT name FROM projects WHERE id = :pid"),{"pid": project_id}).fetchone()
                if row:
                    project_namespace = row[0].lower().replace(" ", "_")

        result = process_query(question, project_id, project_namespace=project_namespace)

        chat = ChatHistory(
            user_id=user_id, user_message=question,
            bot_response=result["response"],
            project_context_id=project_id,
            query_type=result["query_type"],
            confidence_score=result["confidence_score"],
        )
        db.session.add(chat)
        db.session.commit()

        return jsonify({
            "data": {
                "question":        question,
                "response":        result["response"],
                "bot_response":    result["response"],
                "query_type":      result["query_type"],
                "confidence_score": result["confidence_score"],
                "confidence":      result["confidence_score"],
                "metadata":        result.get("metadata", {}),
            },
            "success": True,
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.exception(f"bot_query error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/bot/reindex", methods=["POST"])
def reindex():
    if not ai_ready:
        return jsonify({"error": "AI not initialised", "success": False}), 503
    try:
        from ai_engine import refresh_index
        from rag_indexer import get_index_meta
        refresh_index()
        return jsonify({"success": True, "message": "Index rebuilt",
                        "meta": get_index_meta()}), 200
    except Exception as e:
        logger.exception(f"reindex error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/bot/history", methods=["GET"])
def get_chat_history():
    try:
        project_id = request.args.get("project_id")
        limit      = int(request.args.get("limit", 20))
        q = ChatHistory.query.order_by(ChatHistory.created_at.desc())
        if project_id:
            q = q.filter_by(project_context_id=project_id)
        history = q.limit(limit).all()
        return jsonify({"data": [h.to_dict() for h in reversed(history)],
                        "success": True, "total": len(history)}), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/bot/status", methods=["GET"])
def ai_status():
    from rag_indexer import get_index_meta
    return jsonify({"ai_ready": ai_ready,
                    "ai_error": ai_error_message if not ai_ready else None,
                    "index_meta": get_index_meta()}), 200


# ═════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "success": False}), 404


@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    return jsonify({"error": "Internal server error", "success": False}), 500


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logger.info("✅ DB tables ready")
        try:
            db.session.execute(text("SELECT 1"))
            logger.info(f"✅ DB verified — {Project.query.count()} project(s)")
        except Exception as e:
            logger.error(f"❌ DB error: {e}")
        _init_ai()

    logger.info("🚀 Flask running on :5000")
    app.run(host="0.0.0.0", port=5000, debug=False)