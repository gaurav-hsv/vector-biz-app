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
    Decide: answer / clarify / not_understood with strong schema guidance.
    Returns STRICT JSON per schema.
    """
    import json, re

    def _truncate(txt: str, limit: int = 1200) -> str:
        if not txt:
            return ""
        return txt if len(txt) <= limit else txt[:limit] + "..."

    # Build numbered passages (as before)
    numbered_blocks = []
    # Also build a compact CATALOG = idx | engagement | field | value peek (schema-aware)
    catalog_rows = []
    for i, p in enumerate(passages, start=1):
        t = (p.get("title") or "").strip()
        c = _truncate(p.get("content") or "", 1200)

        # Try to extract engagement/field/value from structured content
        # Expected ingestion format:
        #   Engagement: <name>\nField: <field_key>\nValue: <text>
        eng = ""
        fld = ""
        val = ""
        m_eng = re.search(r"\bEngagement:\s*([^\n]+)", c)
        m_fld = re.search(r"\bField:\s*([^\n]+)", c)
        m_val = re.search(r"\bValue:\s*(.*)$", c, re.DOTALL)
        if m_eng: eng = m_eng.group(1).strip()
        if m_fld: fld = m_fld.group(1).strip()
        if m_val: val = _truncate(m_val.group(1).strip(), 160)

        # Fallbacks from title if needed
        if not eng and " · " in t:
            eng = t.split(" · ", 1)[0].strip()
        if not fld and " · " in t:
            fld = t.split(" · ", 1)[1].strip()

        numbered_blocks.append(f"{i}) {t}\n{c}\n")
        catalog_rows.append(f"{i:>2} | {eng or '-'} | {fld or '-'} | {val or '-'}")

    passages_text = "\n".join(numbered_blocks) if numbered_blocks else "None"
    catalog_text = "idx | engagement | field | value\n" + "\n".join(catalog_rows) if catalog_rows else "None"

    # Short tail context (keep as in your original)
    tail_txt = ""
    if tail_messages:
        last = tail_messages[-2:]
        lines = []
        for m in last:
            role = m.get("role", "user")
            txt = _truncate(m.get("text", ""), 300)
            lines.append(f"{role}: {txt}")
        tail_txt = "\n".join(lines) if lines else ""

    # ---------------- SYSTEM PROMPT (schema-aware) ----------------
    system_msg = (
        "You are a production-grade Microsoft Partner Business Applications Incentives assistant.\n"
        "Your scope is STRICTLY BizzApps incentives & engagements. Be decisive, minimal, precise.\n"
        "\n"
        "HARD GROUNDING RULES:\n"
        "- Use ONLY the provided passages.\n"
        "- Never introduce external facts, rates, bands, or numbers.\n"
        "- If exactly ONE missing input blocks precision, use mode='clarify' and ask ONE short question.\n"
        "- For payout: NEVER ask for a 'cap' or 'market rate'. If geography matters, ask only for 'country'.\n"
        "- If invalid/out-of-scope, use mode='not_understood' with a better rephrase.\n"
        "\n"
        "FIELD GUIDE (what each field means & how to answer):\n"
        "• engagement_type: High-level category (e.g., Pre-sales, CSP Incentive). If asked for 'types of incentives', list engagement_type values.\n"
        "• incentive_type: Program type within an engagement (e.g., Pre-sales activities, Workshop, 1:Many Briefings, Transaction).\n"
        "• workload / solution_play / specialization / solution_partner_designation: Return the exact value(s). No extra text.\n"
        "• activity_requirement: Bullet-like requirements. Provide the exact requirement(s). If multiple relevant items exist, list 2–4 succinct bullets.\n"
        "• customer_qualification: Customer-side eligibility. If explicit 'no requirement' or equivalent is present, state it plainly.\n"
        "• partner_qualification: Partner-side eligibility. Same rule as above.\n"
        "• product_eligibility / licensing_agreement: State exactly what's written (eligible SKUs, agreements). If none, state 'No requirement specified'.\n"
        "• limits / maximum_incentive_earning_opportunity / revenue_threshold / minimum_hours: Return the explicit value. If not specified, say 'Not specified'.\n"
        "• formula: When user asks to calculate payout/earnings or provides numeric context (ACV/ARR/%, seats, hours):\n"
        "  - Prefer the passage with formula.\n"
        "  - If all inputs present → answer directly (calculation happens later in the answer step).\n"
        "  - If one input missing → ask one short clarification (country/ACV/hours/seats). Do NOT ask about caps.\n"
        "• cpor: If stated as required/not required, answer directly. No extra clarification.\n"
        "• partner_performance_measure / partner_proof_of_execution: Return exactly what's specified; if absent, say 'Not specified'.\n"
        "• market_a/b/c & market_*_definition: If referenced, use exactly the given definitions; if geography needed, ask only for 'country'.\n"
        "\n"
        "DECISION POLICY:\n"
        "- If the user asks for 'types of incentives' or 'incentive categories':\n"
        "  • Prefer engagement_type values (e.g., Pre-sales, CSP Incentive). If incentive_type is also present, include both succinctly.\n"
        "- For numeric/transactional questions (ACV/ARR/revenue/%/rate/hours/seats/payout/earnings):\n"
        "  • If any passage is a formula passage, set mode='answer' and pick that formula passage.\n"
        "- For eligibility/license questions:\n"
        "  • Distinguish customer_qualification vs partner_qualification. Do not mix them.\n"
        "  • If text says no requirement, answer directly; do not clarify.\n"
        "- If question references or implies a specific engagement mentioned in recent context or passages, prefer that engagement's passage.\n"
        "- If plural query ('types of…', 'which engagements…'), avoid single narrow answers—return the relevant set from the passages.\n"
        "\n"
        "JSON OUTPUT RULES:\n"
        "- Return STRICT JSON only (no extra text).\n"
        "- Schema: {mode, why, pick, confidence, followup:{message,options}, recommendations[]}.\n"
        "- pick is 1-based index into Passages if mode='answer'; null otherwise.\n"
        "- recommendations: 3–5 unique CTA-style, grounded in the SAME passages (no external facts).\n"
    )

    # ---------------- USER PROMPT ----------------
    user_msg = f"""
ORIGINAL USER MESSAGE:
{user_text}

RECENT CONTEXT (last messages):
{tail_txt or "None"}

PASSAGES (numbered, full text):
{passages_text}

CATALOG (parsed overview: idx | engagement | field | value):
{catalog_text}

Return EXACTLY this JSON schema:
{{
  "mode": "answer" | "clarify" | "not_understood",
  "why": "short reason",
  "pick": <int or null>,               
  "confidence": <number 0..1>,
  "followup": {{
    "message": "one short clarification/rephrase",
    "options": ["q1","q2","q3","q4"]
  }},
  "recommendations": ["q1","q2","q3","q4"]
}}
STRICT RULES:
- Use ONLY the passages and the CATALOG.
- If any passage precisely answers the question, set mode="answer" and the correct 1-based 'pick'.
- If ONE short missing detail blocks precision, set mode="clarify" with exactly ONE short question (options optional).
- If unclear/out-of-scope, set mode="not_understood" with a better rephrase in followup.message.
- If user asks for 'types of incentives' or 'incentive categories', prefer engagement_type values; include incentive_type if clearly present.
- For numeric or payout calculation intent, pick the formula passage if present.
- Keep recommendations 3–5, CTA-style, deduped, grounded in the chosen/nearby passages only.
"""

    # --------------- COMPLETION ---------------
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
        data = {
            "mode": "clarify",
            "why": "model returned non-JSON",
            "pick": None,
            "confidence": 0.0,
            "followup": {"message": "Could you clarify your question in one sentence?", "options": []},
            "recommendations": [],
        }

    # --------------- SANITIZE ---------------
    mode = data.get("mode")
    if mode not in ("answer", "clarify", "not_understood"):
        data["mode"] = "clarify"
        data.setdefault("followup", {})
        data["followup"].setdefault("message", "Could you clarify your question in one sentence?")
        data["pick"] = None
        data["confidence"] = float(data.get("confidence", 0.0) or 0.0)

    if data["mode"] in ("clarify", "not_understood"):
        data.setdefault("followup", {})
        data["followup"].setdefault("message", "Could you clarify your question in one sentence?")
        data["followup"].setdefault("options", [])

    if data.get("pick") is not None:
        try:
            data["pick"] = int(data["pick"])
        except Exception:
            data["pick"] = None

    try:
        data["confidence"] = float(data.get("confidence", 0.0))
    except Exception:
        data["confidence"] = 0.0

    data["why"] = (data.get("why") or "").strip()[:200]

    # Deduplicate & trim recommendations
    recs = data.get("recommendations", [])
    if isinstance(recs, list):
        seen, cleaned = set(), []
        for q in recs:
            qn = (q or "").strip()
            if not qn:
                continue
            low = qn.lower()
            if low not in seen:
                cleaned.append(qn)
                seen.add(low)
        data["recommendations"] = cleaned[:5]

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

    blob = f"{title}\n{content}".lower()
    is_formula = ("formula" in blob)  # minimal, robust enough with your data

    if is_formula:
        system_msg = ("""
You are a precise, grounded payout calculator for Microsoft Business Apps incentives.
SCOPE & GROUNDING:
    -Use ONLY the provided snippet(s); do not import external facts.
CALC RULES:
        -If a formula is present and inputs are provided, compute payout.
        -CAP: Never ask the user for a cap. If a 'Maximum Incentive Earning Opportunity' appears in the provided text, treat it as the hard cap; if not present, assume no cap.
        -MARKET RATE: Never ask for 'market rate'. If geography matters, ask only for 'country'. When country is known and Market_A/B/C *definitions* appear in the provided text, map the country to A/B/C and use the corresponding Incentive_Market_[A|B|C] value.
OUTPUT:
                      ANSWER: You will earn **{currency} <amount>**.
           • Breakdown: one short line showing the formula with substituted numbers and the result; if a cap applies, show the min() step.
           FORMAT:
            • Always include currency; format numbers with thousands separators and 2 decimals.
            WHEN BLOCKED:
             • If exactly one input is missing (e.g., country, ACV, hours),             
                      """

        )
        # tip: lower temperature for deterministic math
        answer_model = getattr(settings, "LLM_MODEL_ANSWER", "gpt-4o-mini")
        temperature = 0.0

        user_msg = f"""User message:
{user_text}

PASSAGE (only source of rules; do not use anything else):
{content}

Output either:
CLARIFY: <one missing input question>
or
ANSWER: Payout: <amount and currency if present>. <≤1-line breakdown of how computed>
"""
    else:
        system_msg = (
        """You are a Microsoft Partner Incentives assistant.
Your scope is STRICTLY Business Applications incentives.
Answer ONLY from the provided passage.
Do not invent or add external facts.
If missing, return CLARIFY with exactly one short question."""
        )
        answer_model = getattr(settings, "LLM_MODEL_ANSWER", "gpt-4o-mini")
        temperature = 0.2

        user_msg = f"""User asked: "{user_text}"
Use ONLY this passage (title: {title}):
{content}

Output either:
CLARIFY: "<question>"
or
ANSWER: "<final 1–2 sentences>"
"""

    comp = client.chat.completions.create(
    model=answer_model,
    temperature=temperature,
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    )
    txt = (comp.choices[0].message.content or "").strip()
    if not (txt.startswith("ANSWER:") or txt.startswith("CLARIFY:")):
        txt = "ANSWER: " + txt
    return txt

