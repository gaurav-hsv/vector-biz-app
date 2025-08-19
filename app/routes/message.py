# app/routes/message.py
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import re

# Sessions (Redis-backed)
from ..sessions_redis import ensure_session, append_message, get_tail
from ..search import vector_search
from ..llm import embed, vec_literal, llm_decide, llm_answer

router = APIRouter()

# ---------------------------
# Request model
# ---------------------------
class MessageIn(BaseModel):
    session_id: Optional[str] = None
    text: str = Field(min_length=1)
    selection: Optional[str] = None  # optional quick-pick

# ---------------------------
# Field Synonyms (expanded)
# ---------------------------
_FIELD_SYNONYMS = {
    # Core meta
    "Solution_Play": [r"\bsolution play\b", r"\bplay\b"],
    "Engagement_Type": [r"\bengagement type\b"],
    "Incentive_Type": [r"\bincentive type\b"],
    "Workload": [r"\bworkload\b"],
    "Engagement_Name": [r"\bengagement name\b", r"\bname of engagement\b"],

    # Business fields
    "Activity_Requirement": [r"\bactivities\b", r"\bactivity\b", r"\bdeliverables?\b",
                             r"\brequirements?\b", r"\bactivity requirement(s)?\b"],
    "Customer_Qualification": [
        r"\bcustomer (eligibility|qualification|qualifications)\b",
        r"\bcustomer eligible\b",
        r"\blicense requirement(s)?\b",  # NEW
        r"\bminimum license\b",          # NEW
    ],
    "Partner_Qualification": [
        r"\bpartner (eligibility|qualification|qualifications)\b",
        r"\bpartner eligible\b"
    ],
    "Limits": [r"\blimits?\b", r"\bcap(s)?\b", r"\bmaximum\b", r"\bmax\b"],
    "Engagement_Goal": [r"\bgoal\b", r"\bobjective(s)?\b", r"\bpurpose\b", r"\bintent\b"],
    "Formula": [r"\bformula\b", r"\bcalc(?:ulation|ulate|ulating)?\b", r"\bcompute\b",
                r"\brate\b", r"\bhow (to )?calculate\b"],
    "Earning_Type": [r"\bearning (type|method|basis)\b", r"\bpayout\b", r"\bpaid\b"],
    "Maximum_Incentive_Earning_Opportunity": [
        r"\bmaximum incentive\b", r"\bmax(?:imum)? earning opportunity\b",
        r"\bmax payout\b", r"\bearning opportunit(y|ies)\b"  # NEW
    ],
       "Incentive_Categories": [
        r"\btypes of incentives\b",
        r"\bwhat incentives\b",
        r"\bincentive categories\b",
        r"\bavailable incentives\b"
    ],
    "Revenue_Threshold": [r"\brevenue threshold\b", r"\bminimum revenue\b", r"\brevenue target\b"],
    "Minimum_Hours": [r"\bminimum hours\b", r"\bmin hours\b", r"\bhours required\b"],
    "Partner_Performance_Measure": [r"\bperformance measure(s)?\b", r"\bkpi(s)?\b", r"\bmetrics?\b"],
    "Partner_Proof_of_Execution": [r"\bproof of execution\b", r"\bpoe\b", r"\bevidence\b", r"\battestation\b"],
    "Solution_Partner_Designation": [r"\bsolution partner designation\b", r"\bdesignation\b", r"\bspd\b"],  # EXPANDED
    "Specialization": [r"\bspeciali[sz]ation(s)?\b", r"\bspecialist\b"],
    "Product_Eligibility": [r"\bproduct eligibility\b", r"\beligible products?\b", r"\bsku(s)?\b"],
    "Licensing_Agreement": [r"\blicensing agreement\b", r"\blicen[cs]e\b", r"\blicensing\b", r"\bagreement\b"],
    "CPOR": [r"\bCPOR\b", r"\bcpor\b", r"\bclaim proof of registration\b"],
    "Enterprise_Segment": [r"\benterprise segment\b", r"\benterprise\b", r"\blarge\b"],
    "SMEandC_Segment": [r"\bSME&C\b", r"\bSME and C\b", r"\bSME\b", r"\bSMB\b"],
    "T_Shirt_Size": [r"\bt-?shirt size\b"],

    # Market splits
    "Incentive_Market_A": [r"\bmarket A\b", r"\bsegment A\b"],
    "Incentive_Market_B": [r"\bmarket B\b", r"\bsegment B\b"],
    "Incentive_Market_C": [r"\bmarket C\b", r"\bsegment C\b"],
    "Market_A_Definition": [r"\bmarket A definition\b"],
    "Market_B_Definition": [r"\bmarket B definition\b"],
    "Market_C_Definition": [r"\bmarket C definition\b"],
}

def _canonical_field(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    for canon, pats in _FIELD_SYNONYMS.items():
        for rx in pats:
            if re.search(rx, t):
                return canon
    return None

def _guess_engagement_from_text(user_text: str) -> Optional[str]:
    """Heuristic guess; used only to improve structured embedding prompt."""
    t = (user_text or "").strip()
    m = re.search(r"['\"]([^'\"]{3,80})['\"]", t)
    if m:
        return m.group(1).strip()
    caps = re.findall(r"(?:[A-Z][A-Za-z0-9+\-]*\s+){1,4}[A-Z][A-Za-z0-9+\-]*", t)
    if caps:
        return caps[0].strip()
    return None

def _structured_query_text(user_text: str, field: Optional[str], engagement: Optional[str]) -> str:
    """Mirrors ingestion format for better embedding alignment."""
    parts = []
    if engagement:
        parts.append(f"Engagement: {engagement}")
    if field:
        parts.append(f"Field: {field}")
    if parts:
        parts.append("Value: (answer)")
        parts.append(user_text)
        return "\n".join(parts)
    return user_text

def _extract_value_from_content(content: str) -> str:
    text = (content or "").strip()
    m = re.search(r"\bValue:\s*(.*)$", text)
    return m.group(1).strip() if m else text

def _trim_passages(hits: List[Dict[str, Any]], max_chars: int = 1200) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for h in hits:
        title = (h.get("title") or "").strip()
        content = (h.get("content") or "").strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        out.append({"title": title, "content": content})
    return out

def _fallback_recommendations_from_hits(hits: List[Dict[str, Any]], n: int = 4) -> List[str]:
    """Make grounded next-step questions from nearby results, avoid echoing."""
    recs: List[str] = []
    for h in hits[1:n+1]:
        t = (h.get("title") or "").strip()
        if not t:
            continue
        if " · " in t:
            eng, field = t.split(" · ", 1)
            q = f"{field} for {eng}?"
        else:
            q = f"More on {t}?"
        # Avoid echoing same question
        if q and q not in recs:
            recs.append(q)
        if len(recs) >= n:
            break
    return recs

# ---------------------------
# Route
# ---------------------------
@router.post("/message")
def post_message(inp: MessageIn, debug: bool = Query(False, description="return debug info")):
    sid = ensure_session(inp.session_id)
    q = inp.text.strip()
    if inp.selection:
        q = f"{q} {inp.selection.strip()}"
    append_message(sid, "user", q)

    # Signals
    detected_field = _canonical_field(q)
    eng_guess = _guess_engagement_from_text(q)

    # 1) Embed
    structured_q = _structured_query_text(q, detected_field, eng_guess)
    try:
        q_vec = embed(structured_q)
    except Exception as e:
        return {"type": "error", "session_id": sid, "error": "llm_embed_failed", "detail": str(e)[:200]}

    # 2) Search
    try:
        hits = vector_search(vec_literal(q_vec), k=30)
    except Exception as e:
        return {"type": "error", "session_id": sid, "error": "db_search_failed", "detail": str(e)[:200]}

    if not hits:
        return {"type": "not_found", "session_id": sid, "message": "No matching content in incentives data."}

    # Short-circuit direct answers
    if detected_field and hits:
        top = hits[0]
        if float(top.get("score", 0.0)) >= 0.75:
            value = _extract_value_from_content(top.get("content", ""))
            payload = {
                "type": "answer",
                "session_id": sid,
                "engagement": top.get("title", ""),
                "text": value,
                "confidence": float(top.get("score", 0.0)),
                "recommendations": _fallback_recommendations_from_hits(hits, n=4),
            }
            if debug:
                payload["hits"] = [{"title": h.get("title"), "score": float(h.get("score", 0))} for h in hits[:5]]
                payload["detected_field"] = detected_field
                payload["engagement_guess"] = eng_guess
            return payload

    passages_for_decide = _trim_passages(hits, max_chars=12000)
    tail = get_tail(sid, n=6)

    # 3) Decide
    try:
        decision = llm_decide(user_text=q, passages=passages_for_decide, tail_messages=tail)
    except Exception as e:
        decision = {
            "mode": "clarify",
            "why": f"decision_error: {str(e)[:120]}",
            "pick": None,
            "confidence": 0.0,
            "followup": {"message": "Could you clarify your question in one sentence?", "options": []},
        }

    mode = decision.get("mode")

    # 4) Act
    if mode == "answer":
        pick = decision.get("pick")
        if not isinstance(pick, int) or pick < 1 or pick > len(hits):
            pick = 1
        chosen_full = hits[pick - 1]
        try:
            out = llm_answer(user_text=q, passage=chosen_full)
        except Exception as e:
            return {"type": "error", "session_id": sid, "error": "llm_answer_failed", "detail": str(e)[:200]}

        if out.startswith("CLARIFY:"):
            msg = out.replace("CLARIFY:", "", 1).strip()
            payload = {"type": "clarify", "session_id": sid, "message": msg}
        else:
            text = out.replace("ANSWER:", "", 1).strip()
            payload = {
                "type": "answer",
                "session_id": sid,
                "engagement": chosen_full.get("title", ""),
                "text": text,
                "confidence": float(decision.get("confidence", 0.0)),
                "recommendations": decision.get("recommendations", _fallback_recommendations_from_hits(hits, 4))
            }
        if debug:
            payload["decision"] = decision
            payload["hits"] = [{"title": h.get("title"), "score": float(h.get("score", 0))} for h in hits[:5]]
        return payload

    elif mode in ("clarify", "not_understood"):
        follow = decision.get("followup", {}) or {}
        msg = (follow.get("message") or "Could you clarify your question?").strip()
        opts = follow.get("options") or []
        payload = {"type": "clarify", "session_id": sid, "message": msg}
        if opts:
            payload["options"] = opts
        if debug:
            payload["decision"] = decision
            payload["hits"] = [{"title": h.get("title"), "score": float(h.get("score", 0))} for h in hits[:5]]
        return payload

    payload = {
        "type": "clarify",
        "session_id": sid,
        "message": "Could you clarify your question in one sentence?",
        "recommendations": decision.get("recommendations", []),
    }
    if debug:
        payload["decision"] = decision
        payload["hits"] = [{"title": h.get("title"), "score": float(h.get("score", 0))} for h in hits[:5]]
    return payload
