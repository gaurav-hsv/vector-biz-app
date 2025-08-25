# app/routes/message.py
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ..search import vector_search
from ..llm import generate_answer, detect_query_type, get_config_by_llm, is_country_answer, explain_from_dumped_config
from ..calculations_config import CALCULATIONS_CONFIG
from ..country_config import resolve_market_from_text
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


def _patch_workshop_with_market_rate(cfg: Dict[str, Any], rate: int) -> Dict[str, Any]:
    """Append (or set) market_rate field on each workshop engagement."""
    if not isinstance(cfg, dict):
        return {}
    # shallow copy is enough for our patch usage
    out = json.loads(json.dumps(cfg))  # cheap deep copy
    workshops = out.get("workshop")
    if not isinstance(workshops, list):
        return out

    for eng in workshops:
        ffs = eng.get("form_fields")
        if not isinstance(ffs, list):
            ffs = []
        # try to find existing market_rate
        found = False
        for f in ffs:
            if str(f.get("field_name") or "").strip().lower() == "market_rate":
                f["Value"] = rate
                found = True
                break
        if not found:
            ffs.append({
                "field_name": "market_rate",
                "about": "derived from country via static market mapping",
                "label": "number",
                "Value": rate,
            })
        eng["form_fields"] = ffs
    out["workshop"] = workshops
    return out


# ---------------------------
# Router
# ---------------------------
router = APIRouter()

class MessageIn(BaseModel):
    session_id: Optional[str] = None
    text: str = Field(min_length=1)
    input_type: Optional[str] = None  # optional quick-pick
    config: Optional[Any] = None  # optional partial config to patch

@router.post("/message")
def post_message(inp: MessageIn, debug: bool = Query(False, description="return debug info")):
    # Load/append user message
    session = get_session(inp.session_id)
    append_message(session["session_id"], "user", inp.text)
    session = get_session(session["session_id"])

    if inp.input_type == "market_country":
        is_country = is_country_answer(inp.text)
        if is_country:
            # FIX: resolve returns (rate, country, market)
            market, country, rate = resolve_market_from_text(inp.text)
            cfg_in = inp.config or {}
            # Ensure we have a numeric rate (handle accidental 'A'/'B'/'C' just in case)
            rate_val = rate
            if isinstance(rate_val, str):
                MR = {"A": 163, "B": 116, "C": 70}
                rate_val = MR.get(rate_val.upper(), None)
        # (if your resolver already returns 163/116/70, this stays as-is)

            if rate_val is not None:
                cfg_patched = _patch_workshop_with_market_rate(cfg_in, rate_val)
                append_message(session["session_id"], "assistant", json.dumps(cfg_patched))
                return {
                "type": "answer",
                "session_id": session["session_id"],
                "text": "",
                "config": cfg_patched,
                "recommendations": [],
                }

    if inp.input_type == "calc_submitted":
        dump_cfg = inp.config or {}
        llm_out = explain_from_dumped_config(dump_cfg)

        append_message(session["session_id"], "assistant", llm_out.get("answer", ""))

        return {
            "type": "answer",
            "session_id": session["session_id"],
            "text": llm_out.get("answer", ""),
            "recommendations": [],
        }

    # Reuse last topic if user uses pronouns like "this incentive"
    last_topic = _load_topic(session)
    is_followup = bool(last_topic) and _looks_like_followup_with_pronoun(inp.text)
    effective_query = (f"{last_topic} {inp.text}".strip()) if is_followup else inp.text

    # Retrieval
    search_results = vector_search(effective_query)
    #print(f"Search results: {search_results}")
    sources = search_results.get("sources") or []
    # Derive and store topic for NEXT turn (from current retrieval)
    topic = _derive_topic_from_sources(sources)
    #print(f"Derived topic: {topic}")
    if topic:
        _store_topic(session["session_id"], topic)

    # LLM answer on the same effective query + sources
    user_intent = detect_query_type(inp.text)
    #print(f"Detected user intent: {user_intent}")

    if(user_intent == 'information') :
        result = generate_answer(effective_query,sources)
        # Persist assistant message (helps future heuristics if needed)
        append_message(session["session_id"], "assistant", (result.get("answer") or ""))

        session = get_session(session["session_id"])
        resp = {
        "type": "answer",
        "session_id": session["session_id"],
        "text": result.get("answer"),
        "recommendations": result.get("recommendations", []),
        }
        return resp
    
    config = get_config_by_llm(inp.text, CALCULATIONS_CONFIG,sources)

    append_message(session["session_id"], "assistant", (json.dumps(config) or ""))
    #print(f"Config selected: {json.dumps(config)}")

    # Fresh session view
    session = get_session(session["session_id"])

    resp = {
        "type": "answer",
        "session_id": session["session_id"],
        "text": "",
         "config": config,
        "recommendations": [],
    }
    return resp
    
