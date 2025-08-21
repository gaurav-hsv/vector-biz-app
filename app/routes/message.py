# app/routes/message.py
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ..search import vector_search
from ..llm import generate_answer

# Sessions (Redis-backed)
from ..sessions_redis import get_session, append_message

# ---------------------------
# NEW: lightweight context utils
# ---------------------------
import re, json

# Pronoun / anaphora patterns we’ll rewrite using last topic
_ANAPHORA = re.compile(
    r"\b(this|that|these|those|it|them|the\s*(program|incentive|workshop|engagement|offer|scheme|activity|requirements?))\b",
    re.I,
)

# Prefer explicit metadata keys from your ingest
_TOPIC_KEYS = (
    "engagement_name", "incentive_name", "name", "title",
    "workload", "program", "product", "doc_name",
)

def _derive_topic_from_sources(sources: List[Dict[str, Any]]) -> Optional[str]:
    # 1) Try metadata keys (strongest)
    for s in sources or []:
        meta = s.get("metadata") or {}
        for k in _TOPIC_KEYS:
            v = (meta.get(k) or "").strip()
            if v and len(v) > 2:
                return v
    # 2) Fallback: extract a capitalized noun phrase ending with known terms
    for s in sources or []:
        text = (s.get("content") or "")
        m = re.search(r"\b([A-Z][A-Za-z0-9+/&\-\s]{3,}?(?:Incentive|Workshop|Program|Engagement)s?)\b", text)
        if m:
            return m.group(1).strip()
    return None

def _store_topic(session_id: str, topic: str):
    # Persist as a system message (no schema change)
    append_message(session_id, "system", "CTX_TOPIC:" + topic)

def _load_topic(session: dict) -> Optional[str]:
    msgs = (session or {}).get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "system":
            t = m.get("text") or ""
            if t.startswith("CTX_TOPIC:"):
                topic = t[len("CTX_TOPIC:"):].strip()
                if topic:
                    return topic
    return None

def _looks_like_followup_with_pronoun(msg: str) -> bool:
    t = (msg or "").strip()
    return bool(_ANAPHORA.search(t)) or (len(t.split()) <= 10 and bool(re.search(r"\b(eligibil|requirement|activities?|rate|amount|payment|timeline|scope)\w*\b", t, re.I)))

# ---------------------------
# Router
# ---------------------------
router = APIRouter()

class MessageIn(BaseModel):
    session_id: Optional[str] = None
    text: str = Field(min_length=1)
    selection: Optional[str] = None  # optional quick-pick

@router.post("/message")
def post_message(inp: MessageIn, debug: bool = Query(False, description="return debug info")):
    # Load/append user message
    session = get_session(inp.session_id)
    append_message(session["session_id"], "user", inp.text)
    session = get_session(session["session_id"])

    # Reuse last topic if user uses pronouns like "this incentive"
    last_topic = _load_topic(session)
    is_followup = bool(last_topic) and _looks_like_followup_with_pronoun(inp.text)
    effective_query = (f"{last_topic} {inp.text}".strip()) if is_followup else inp.text

    # Retrieval
    search_results = vector_search(effective_query)

    # Derive and store topic for NEXT turn (from current retrieval)
    topic = _derive_topic_from_sources(search_results.get("sources") or [])
    if topic:
        _store_topic(session["session_id"], topic)

    # LLM answer on the same effective query + sources
    result = generate_answer(effective_query, search_results['sources'])

    # Persist assistant message (helps future heuristics if needed)
    append_message(session["session_id"], "assistant", (result.get("answer") or ""))

    # Fresh session view
    session = get_session(session["session_id"])

    resp = {
        "type": "answer",
        "session_id": session["session_id"],
        "text": result.get("answer"),
        "recommendations": result.get("recommendations", []),
    }
    return resp
