"""
rag_indexer.py
Builds and saves a FAISS vector store with robust namespace normalization.
Sources:
  1. DB rows  — projects, tasks, risks (always present)
  2. Documents — documents/<project_namespace>/*.txt|*_BRS* (optional)

3-Tier search:
  Tier 1 — explicit project_namespace filter
  Tier 2 — keyword inference from query text
  Tier 3 — global semantic scan (no filter)
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

BACKEND_DIR    = Path(__file__).parent
VECTOR_DIR     = BACKEND_DIR / "vector_store"
FAISS_INDEX    = VECTOR_DIR / "project_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Documents root — siblings to backend/: ../documents/
DOCS_BASE_DIR  = BACKEND_DIR.parent / "documents"


def clean_namespace(name_str: str | None) -> str:
    """
    Standardizes namespaces across folder paths and database entries to match
    exact alphanumeric formats. Solves 'PCI DSS' -> 'pcidss' folder mismatches.
    """
    if not name_str:
        return ""
    # Strip spaces, dashes, underscores, and punctuation
    return re.sub(r'[^a-zA-Z0-9]', '', str(name_str)).lower()


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: DB rows → Documents
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_db_documents(db_url: str) -> list[Document]:
    engine = create_engine(db_url)
    documents: list[Document] = []

    with engine.connect() as conn:
        projects = conn.execute(text(
            "SELECT id, name, description, owner, status, priority, "
            "progress_percentage, start_date, end_date, budget FROM projects"
        )).fetchall()

        for p in projects:
            pid, name, desc, owner, status, priority, progress, start, end, budget = p
            ns_cleaned = clean_namespace(name)

            # Project doc
            documents.append(Document(
                page_content=(
                    f"PROJECT: {name}\n"
                    f"Description: {desc or 'N/A'}\n"
                    f"Owner: {owner or 'N/A'}\n"
                    f"Status: {status} | Priority: {priority}\n"
                    f"Progress: {progress}%\n"
                    f"Start: {start} | End: {end}\n"
                    f"Budget: {budget or 'N/A'}\n"
                ),
                metadata={
                    "type": "project", 
                    "project_id": pid,
                    "project_name": name,
                    "project_namespace": ns_cleaned
                }
            ))

            # Tasks
            tasks = conn.execute(text(
                "SELECT title, description, status, priority, assigned_to, "
                "progress_percentage, estimated_hours, actual_hours "
                "FROM tasks WHERE project_id = :pid"
            ), {"pid": pid}).fetchall()
            for t in tasks:
                t_title, t_desc, t_status, t_priority, t_assigned, \
                    t_prog, t_est, t_actual = t
                documents.append(Document(
                    page_content=(
                        f"TASK (Project: {name}): {t_title}\n"
                        f"Description: {t_desc or 'N/A'}\n"
                        f"Status: {t_status} | Priority: {t_priority}\n"
                        f"Assigned: {t_assigned or 'Unassigned'}\n"
                        f"Progress: {t_prog}% | Est: {t_est or 0}h | Actual: {t_actual or 0}h\n"
                    ),
                    metadata={
                        "type": "task", 
                        "project_id": pid,
                        "project_name": name,
                        "project_namespace": ns_cleaned
                    }
                ))

            # Risks
            risks = conn.execute(text(
                "SELECT title, description, severity, probability, "
                "mitigation_plan, owner, status "
                "FROM risks WHERE project_id = :pid"
            ), {"pid": pid}).fetchall()
            for r in risks:
                r_title, r_desc, r_sev, r_prob, r_mit, r_owner, r_status = r
                documents.append(Document(
                    page_content=(
                        f"RISK (Project: {name}): {r_title}\n"
                        f"Description: {r_desc or 'N/A'}\n"
                        f"Severity: {r_sev} | Probability: {r_prob}\n"
                        f"Status: {r_status}\n"
                        f"Mitigation: {r_mit or 'None defined'}\n"
                        f"Risk Owner: {r_owner or 'Unassigned'}\n"
                    ),
                    metadata={
                        "type": "risk", 
                        "project_id": pid,
                        "project_name": name,
                        "project_namespace": ns_cleaned
                    }
                ))

    logger.info(f"[RAG] DB: {len(documents)} documents fetched")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: Text / BRS files
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_file_documents() -> list[Document]:
    """
    Scan documents/<project_folder>/*.txt and *_BRS* files.
    Each folder name is normalized as the project_namespace metadata tag.
    """
    documents: list[Document] = []
    if not DOCS_BASE_DIR.exists():
        logger.info(f"[RAG] No documents directory found at {DOCS_BASE_DIR} — skipping file scan")
        return documents

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    logger.info(f"[RAG] Scanning document folders in: {DOCS_BASE_DIR}")

    for project_folder in DOCS_BASE_DIR.iterdir():
        if not project_folder.is_dir():
            continue
        
        # Clean folder name namespace tag
        namespace = clean_namespace(project_folder.name)
        logger.info(f"[RAG] Found namespace folder: {project_folder.name.upper()} normalized to '{namespace}'")

        for file_path in project_folder.iterdir():
            if file_path.suffix == ".txt" or "_BRS" in file_path.name:
                logger.info(f"[RAG]   Parsing: {file_path.name}")
                try:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    raw_docs = loader.load()
                    chunks = splitter.split_documents(raw_docs)
                    for chunk in chunks:
                        chunk.metadata["project_namespace"] = namespace
                        chunk.metadata["type"] = "document"
                        chunk.metadata["source_file"] = file_path.name
                    documents.extend(chunks)
                    logger.info(f"[RAG]   → {len(chunks)} chunks indexed for namespace '{namespace}'")
                except Exception as e:
                    logger.error(f"[RAG]   Error reading {file_path.name}: {e}")

    logger.info(f"[RAG] File docs: {len(documents)} chunks total")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# BUILD / LOAD
# ─────────────────────────────────────────────────────────────────────────────
def build_index(db_url: str) -> FAISS:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[RAG] Building index...")

    db_docs   = _fetch_db_documents(db_url)
    file_docs = _fetch_file_documents()
    all_docs  = db_docs + file_docs

    if not all_docs:
        raise ValueError("No documents to index — add projects to the DB first.")

    embeddings   = _get_embeddings()
    vector_store = FAISS.from_documents(all_docs, embeddings)
    vector_store.save_local(str(FAISS_INDEX))

    meta = {
        "last_indexed":   datetime.now().isoformat(),
        "document_count": len(all_docs),
        "db_docs":        len(db_docs),
        "file_docs":      len(file_docs),
        "model":          EMBEDDING_MODEL,
    }
    (VECTOR_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"[RAG] Index saved — {len(all_docs)} total docs")
    return vector_store


def load_index() -> FAISS | None:
    faiss_file = Path(str(FAISS_INDEX) + ".faiss")
    if not faiss_file.exists():
        return None
    embeddings = _get_embeddings()
    vs = FAISS.load_local(str(FAISS_INDEX), embeddings,
                          allow_dangerous_deserialization=True)
    logger.info("[RAG] Index loaded from disk")
    return vs


def get_index_meta() -> dict:
    meta_file = VECTOR_DIR / "index_meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    return {"last_indexed": "Never", "document_count": 0}


# ─────────────────────────────────────────────────────────────────────────────
# 3-TIER SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def tiered_search(vector_store: FAISS, query: str,
                  project_namespace: str | None = None,
                  k: int = 4) -> list[Document]:
    """
    Tier 1 — explicit namespace parameter passed in
    Tier 2 — namespace inferred from keywords in query text
    Tier 3 — global scan, no filter
    """
    namespace = None

    # Tier 1: Explicit namespace argument
    if project_namespace and project_namespace.strip():
        namespace = clean_namespace(project_namespace)
        logger.info(f"[RAG Tier 1] Explicit normalized namespace: {namespace}")

    # Tier 2: Keyword inference from query string
    if not namespace:
        q_lower = query.lower()
        if DOCS_BASE_DIR.exists():
            for folder in DOCS_BASE_DIR.iterdir():
                if folder.is_dir():
                    clean_folder_name = clean_namespace(folder.name)
                    if clean_folder_name in q_lower or folder.name.lower() in q_lower:
                        namespace = clean_folder_name
                        logger.info(f"[RAG Tier 2] Inferred namespace from query: {namespace}")
                        break

    if namespace:
        docs = vector_store.similarity_search(
            query, k=k, filter={"project_namespace": namespace}
        )
        if docs:
            return docs
        logger.info(f"[RAG] Namespace '{namespace}' returned 0 results — falling through to Tier 3")

    # Tier 3: Global Semantic fallback Scan
    logger.info("[RAG Tier 3] Global semantic scan")
    return vector_store.similarity_search(query, k=k)