# app/llm.py
import json
from typing import Dict, Any, List, Tuple

from openai import OpenAI
from openai._exceptions import APIConnectionError, APIStatusError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from .config import settings

# ---- OpenAI client (explicit key + timeout) ----
if not settings.OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Check your .env or environment.")

client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=getattr(settings, "LLM_TIMEOUT_S", 20))

# ---- Helpers ----
def _truncate(txt: str, limit: int = 1200) -> str:
    if not txt:
        return ""
    return txt if len(txt) <= limit else txt[:limit] + "..."

# ---- Embeddings with retries ----
@retry(
    reraise=True,
    stop=stop_after_attempt(getattr(settings, "LLM_MAX_RETRIES", 3)),
    wait=wait_exponential_jitter(initial=0.5, max=4.0),
    retry=retry_if_exception_type((APIConnectionError, APIStatusError, RateLimitError)),
)
def embed(text: str) -> List[float]:
    """Return embedding vector for the given text (dimension must match your pgvector schema)."""
    resp = client.embeddings.create(model=settings.EMBED_MODEL, input=text)
    return resp.data[0].embedding

def vec_literal(vec: List[float]) -> str:
    """Format Python list[float] into pgvector literal: [v1,v2,...]"""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

# ---- Unified DECISION call (answer / clarify / not_understood) ----
@retry(
    reraise=True,
    stop=stop_after_attempt(getattr(settings, "LLM_MAX_RETRIES", 3)),
    wait=wait_exponential_jitter(initial=0.5, max=4.0),
    retry=retry_if_exception_type((APIConnectionError, APIStatusError, RateLimitError)),
)
def llm_decide(
    *,
    user_text: str,
    passages: List[Dict[str, str]],
    tail_messages: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    Let the model decide: "answer" (pick a passage), or "clarify" (ask one follow-up with optional options),
    or "not_understood" (request a clearer rephrase). No hard-coded rules.
    Returns STRICT JSON:
    {
      "mode": "answer" | "clarify" | "not_understood",
      "why": "short reason",
      "pick": <int|null>,          # 1-based index into passages if answering
      "confidence": <float>,       # 0..1
      "followup": { "message": "...", "options": ["..."] }  # present for clarify/not_understood
    }
    """
    # Build a compact context for the model
    numbered_blocks = []
    for i, p in enumerate(passages, start=1):
        t = (p.get("title") or "").strip()
        c = _truncate(p.get("content") or "", 1200)
        numbered_blocks.append(f"{i}) {t}\n{c}\n")
    passages_text = "\n".join(numbered_blocks) if numbered_blocks else "None"

    tail_txt = ""
    if tail_messages:
        # include last 2 messages only (short context)
        last = tail_messages[-2:]
        lines = []
        for m in last:
            role = m.get("role", "user")
            txt = _truncate(m.get("text", ""), 300)
            lines.append(f"{role}: {txt}")
        tail_txt = "\n".join(lines)

    system_msg = (
    "You are a production-grade Microsoft Partner Business apps Incentives assistant. "
    "Your domain is STRICTLY Business Applications (BizzApps). "
    "Assume the user is always a Microsoft reseller partner asking about anything related to incentives/engagements. "
    "You must be decisive, minimal, and precise. "
    "GROUNDING RULES: "
    "- Use ONLY the provided passages. "
    "- If the fact is not present, DO NOT invent or generalize. "
    "- If one detail is missing, return mode=clarify with ONE clear follow-up question. "
    "- If question is invalid or out of scope, return mode=not_understood. "
    "JSON RULES: "
    "- Return STRICT JSON only, never text outside JSON. "
    "- JSON must exactly match schema. "
    "INTERACTION RULES: "
    "- If answer is possible, set mode=answer, pick the passage index, and give confidence. "
    "- Always suggest 3–4 concise recommd-up questions that a partner may naturally ask next. "
    """"""
    """RECOMMENDATIONS
- recomendation question purpose to continue the converations based on user original message.
-Return 3–5 short, non-duplicative prompts that continue the conversation based on the ORIGINAL_USER_MESSAGE and what you just showed.
- Style: CTA phrasing the user can click, e.g., “Want to …”, “Interested in …”, “Need to …”, “See …”, “Compare …”, “Check …”, “Confirm …”.
- for example
    -  "Want to check your customer’s eligibility?",
    -  "Interested in incentive earnings for this engagement?"""
)

    user_msg = f'''User:
{user_text}

Recent context (last messages):
{tail_txt or "None"}

Passages (numbered):
{passages_text}

Return EXACTLY this JSON schema:
{{
  "mode": "answer" | "clarify" | "not_understood",
  "why": "short reason",
  "pick": <int or null>,               
  "confidence": <number 0..1>,
  "followup": {{
    "message": "one short clarification/rephrase",
    "options": ["q1","q2","q3","q4"]   # 3–4 natural follow-up partner questions, omit if not useful
  }},
  "recommendations": ["q1","q2","q3","q4"]
}}
Rules:
- Use ONLY the passages (no external facts).
- If a precise answer is possible from ANY passage, set mode=answer and the correct 1-based 'pick'.
- If one short missing detail blocks precision, set mode=clarify with exactly ONE short question (options optional).
- If unclear/out-of-scope, set mode=not_understood with a better rephrase in followup.message.
- If mode=answer, 'recommendations' MUST include 3–5 grounded next questions relevant to Business Apps reseller partners, derived from the same passages only.
'''

    comp = client.chat.completions.create(
        model=getattr(settings, "LLM_MODEL_RERANK", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = comp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # fallback minimal clarify if model didn't return JSON
        data = {
            "mode": "clarify",
            "why": "model returned non-JSON",
            "pick": None,
            "confidence": 0.0,
            "followup": {"message": "Could you clarify your question in one sentence?", "options": []},
        }

    # sanitize required fields
    mode = data.get("mode")
    if mode not in ("answer", "clarify", "not_understood"):
        data["mode"] = "clarify"
        data.setdefault("followup", {})
        data["followup"].setdefault("message", "Could you clarify your question in one sentence?")
        data["pick"] = None
        data["confidence"] = float(data.get("confidence", 0.0) or 0.0)

    # ensure followup object exists for clarify/not_understood
    if data["mode"] in ("clarify", "not_understood"):
        data.setdefault("followup", {})
        data["followup"].setdefault("message", "Could you clarify your question in one sentence?")
        data["followup"].setdefault("options", [])

    # ensure pick is int or None
    if data.get("pick") is not None:
        try:
            data["pick"] = int(data["pick"])
        except Exception:
            data["pick"] = None

    # confidence
    try:
        data["confidence"] = float(data.get("confidence", 0.0))
    except Exception:
        data["confidence"] = 0.0

    # short why
    data["why"] = (data.get("why") or "").strip()[:200]

    return data

# ---- Answer call (final short answer) ----
@retry(
    reraise=True,
    stop=stop_after_attempt(getattr(settings, "LLM_MAX_RETRIES", 3)),
    wait=wait_exponential_jitter(initial=0.5, max=4.0),
    retry=retry_if_exception_type((APIConnectionError, APIStatusError, RateLimitError)),
)
def llm_answer(*, user_text: str, passage: Dict[str, str]) -> str:
    """
    Produce ≤2 line final answer from the single chosen passage.
    If truly blocked, return:  CLARIFY: "<one short question>"
    Otherwise:                ANSWER: "<final text>"
    """
    title = (passage.get("title") or "").strip()
    content = passage.get("content") or ""

    system_msg = (
        "Answer in <=2 sentences. Copy figures exactly. No preface, no theory. "
        "If truly unsure, output exactly: CLARIFY: \"<one short question>\""
    )

    user_msg = f"""User asked: "{user_text}"
Use ONLY this passage (title: {title}):
{content}

Output either:
CLARIFY: "<question>"
or
ANSWER: "<final 1–2 sentences>"
"""

    comp = client.chat.completions.create(
        model=getattr(settings, "LLM_MODEL_ANSWER", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    txt = (comp.choices[0].message.content or "").strip()
    # normalize unexpected output
    if not (txt.startswith("ANSWER:") or txt.startswith("CLARIFY:")):
        # Treat as answer by default
        txt = "ANSWER: " + txt
    return txt
