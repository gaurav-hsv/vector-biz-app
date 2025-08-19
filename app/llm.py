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
    Decide: answer / clarify / not_understood with partner-friendly, schema-aware guidance.
    Returns STRICT JSON per schema.
    """
    import json, re

    def _tr(txt: str, limit: int = 1200) -> str:
        if not txt: return ""
        return txt if len(txt) <= limit else txt[:limit] + "..."

    # Build numbered passages + a compact catalog (idx | engagement | field | value-ish)
    numbered_blocks, catalog_rows = [], []
    # Hints: collect incentive categories / types for clarify options
    hint_categories, hint_types = set(), set()
    for i, p in enumerate(passages, start=1):
        t = (p.get("title") or "").strip()
        c = _tr(p.get("content") or "", 1200)

        # Try to parse light structure from naturalized chunks/titles
        eng = ""
        fld = (p.get("field") or "").strip()
        if " · " in t:
            eng = t.split(" · ", 1)[0].strip()

        # Mine categories/types from synthesized sentence if present
        # e.g., "Types of incentives include categories such as Pre-sales and CSP Incentive,
        # and incentive types such as Workshops, Transactions..."
        m_cat = re.search(r"categories (?:such as )?(.+?)(?:\.|, and incentive|$)", c, re.I)
        if m_cat:
            for x in re.split(r",| and ", m_cat.group(1)):
                x = x.strip()
                if x: hint_categories.add(x)
        m_typ = re.search(r"incentive types (?:such as )?(.+?)(?:\.|$)", c, re.I)
        if m_typ:
            for x in re.split(r",| and ", m_typ.group(1)):
                x = x.strip()
                if x: hint_types.add(x)

        # Try to peek a short value from the content for the catalog
        val = c.split("\n", 1)[0][:160]
        numbered_blocks.append(f"{i}) {t}\n{c}\n")
        catalog_rows.append(f"{i:>2} | {eng or '-'} | {fld or '-'} | {val or '-'}")

    passages_text = "\n".join(numbered_blocks) if numbered_blocks else "None"
    catalog_text = "idx | engagement | field | value\n" + "\n".join(catalog_rows) if catalog_rows else "None"

    # Recent tail (2 msgs)
    tail_txt = ""
    if tail_messages:
        last = tail_messages[-2:]
        tail_txt = "\n".join([f"{m.get('role','user')}: {_tr(m.get('text',''),300)}" for m in last])

    # Detect earnings intent up-front (maximize, increase payout, etc.)
    earnings_intent = bool(re.search(
        r"\b(increase|maximi[sz]e|boost|grow|get more|raise)\b.*\b(earn|earnings|payout|revenue)\b|"
        r"\b(earnings?|payout|compensation)\b",
        user_text, flags=re.I
    ))

    # Build hint options for clarify (caps at 6 each)
    cat_opts = sorted(list(hint_categories))[:6]
    type_opts = sorted(list(hint_types))[:6]
    hint_block = ""
    if cat_opts or type_opts:
        hint_block = "OPTION_HINTS:\n"
        if cat_opts:  hint_block += f"- Categories: {', '.join(cat_opts)}\n"
        if type_opts: hint_block += f"- Types: {', '.join(type_opts)}\n"

    # SYSTEM — “exec demo” clarity
    system_msg = (
        "You are a production-grade Microsoft Business Applications Partner Incentives assistant.\n"
        "Operate with extreme clarity and restraint. Be decisive. No fluff.\n"
        "\n"
        "GROUNDING:\n"
        "- Use ONLY the provided passages & catalog. No external facts or numbers.\n"
        "- If exactly ONE input is missing → mode='clarify' with ONE short question.\n"
        "- For payout: NEVER ask about 'cap' or 'market rate'. If geography matters, ask only for 'country'.\n"
        "- If invalid/out-of-scope → mode='not_understood' with a better rephrase.\n"
        "\n"
        "FIELD GUIDE (plain-English; NEVER show raw keys/braces/JSON):\n"
        "• engagement_type → “The incentive categories are …”.\n"
        "• incentive_type  → “Available incentive types include …”.\n"
        "• activity_requirement → 1–4 crisp bullets; partner-facing.\n"
        "• customer_qualification / partner_qualification → state requirement; if none, say “No [customer/partner] requirement.”\n"
        "• product_eligibility / licensing_agreement → exact eligibility; if none, “No specific … requirement.”\n"
        "• limits / maximum_incentive_earning_opportunity / revenue_threshold / minimum_hours → explicit value; else “Not specified.”\n"
        "• formula → for compute/payout intent, prefer the formula passage.\n"
        "• cpor / solution_partner_designation / specialization / solution_play → state succinctly.\n"
        "\n"
        "PREFERENCE RULES:\n"
        "- If the user asks for 'types of incentives' or 'categories', prefer a categories/types passage.\n"
        "- If numeric/transactional or 'calculate' intent → pick a formula passage.\n"
        "- If 'increase/maximize earnings' intent:\n"
        "  • If an 'overview' passage exists for the named engagement → pick that (it aggregates levers).\n"
        "  • Else prefer passages that describe levers: activity_requirement, partner_performance_measure, proof_of_execution, limits, engagement_goal.\n"
        "  • If engagement is not specified → mode='clarify' with a single question asking which engagement/type. Provide options if OPTION_HINTS exist.\n"
        "- For plural queries (types/which), avoid a single narrow answer.\n"
        "\n"
        """RECOMMENDATIONS
- recomendation question purpose to continue the converations based on user original message.
-Return 3–5 short, non-duplicative prompts that continue the conversation based on the ORIGINAL_USER_MESSAGE and what you just showed.
- Style: CTA phrasing the user can click, e.g., “Want to …”, “Interested in …”, “Need to …”, “See …”, “Compare …”, “Check …”, “Confirm …”.
- for example
    -  "Want to check your customer’s eligibility for current incentives?"}",
    -  "Interested in incentive earnings for this engagement?
make sure in every recommendation to use the engagement name from the chosen passage.    
"""

    )

    user_msg = f"""
ORIGINAL USER MESSAGE:
{user_text}

EARNINGS_INTENT: {str(earnings_intent)}

RECENT CONTEXT (last messages):
{tail_txt or "None"}

{hint_block or ""}PASSAGES (numbered):
{passages_text}

CATALOG (idx | engagement | field | value):
{catalog_text}

Return EXACTLY this JSON:
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
- If a precise answer exists, set mode='answer' and the correct 1-based pick.\n
- If ONE short detail blocks precision, set mode='clarify' with exactly one question. Use OPTION_HINTS to propose 2–4 clickable options.\n
- Keep recommendations CTA-style, partner-friendly, 3–5 items, grounded in the chosen/nearby passages.\n
- NEVER output raw field keys, braces, equals-sign lists, or JSON-like fragments to the user.\n
"""

    comp = client.chat.completions.create(
        model="gpt-4o",
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

    # sanitize
    mode = data.get("mode")
    if mode not in ("answer", "clarify", "not_understood"):
        data["mode"] = "clarify"
        data.setdefault("followup", {"message": "Could you clarify your question in one sentence?", "options": []})
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

    # dedupe/trim recommendations
    recs = data.get("recommendations", [])
    if isinstance(recs, list):
        seen, cleaned = set(), []
        for q in recs:
            qn = (q or "").strip()
            if qn and qn.lower() not in seen:
                cleaned.append(qn)
                seen.add(qn.lower())
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
    Produce ≤2 line partner-friendly answer from the chosen passage.
    If blocked: CLARIFY: "<one short question>"
    Otherwise:  ANSWER: "<final text>"
    """
    import re

    title   = (passage.get("title") or "").strip()
    content = (passage.get("content") or "")
    field   = (passage.get("field") or "").lower()

    blob = f"{title}\n{content}".lower()
    is_formula = ("formula" in blob)

    # Detect earnings intent to shape tone (e.g., “how can I increase my earnings”)
    earnings_intent = bool(re.search(
        r"\b(increase|maximi[sz]e|boost|grow|get more|raise)\b.*\b(earn|earnings|payout)\b|"
        r"\b(earnings?|payout)\b",
        (user_text or ""), flags=re.I
    ))

    if is_formula:
        system_msg = (
            "You are a precise payout calculator for Microsoft Business Apps incentives.\n"
            "GROUNDING: Use ONLY the provided passage. No external facts.\n"
            "RULES:\n"
            "- If formula inputs are present → compute payout.\n"
            "- CAP: Never ask for a cap. If 'Maximum Incentive Earning Opportunity' is present, enforce it; otherwise no cap.\n"
            "- MARKET: Never ask for 'market rate'. If geography matters, ask only for 'country'.\n"
            "- If exactly one input is missing (ACV/ARR/%, hours, seats, country) → CLARIFY once.\n"
            "FORMAT:\n"
            "• ANSWER: Payout = <currency amount>. Short breakdown.\n"
            "• Always include currency; 2-decimal, thousands separators.\n"
            "• Or CLARIFY: <one short question>.\n"
        )
        model = "gpt-4o"
        temperature = 0.0
        user_msg = f"""User message:
{user_text}

PASSAGE (formula only):
{content}

Respond with either:
CLARIFY: <one missing input question>
or
ANSWER: Payout = <amount and currency>. <≤1-line breakdown>
"""
    else:
        # Partner-facing, with special handling for “increase earnings”
        system_msg = (
            "You are a Microsoft Business Applications Partner Incentives assistant.\n"
            "Rewrite the passage into a concise, human-friendly answer for a reseller partner.\n"
            "GROUNDING: Use ONLY this passage. No external facts.\n"
            "If missing/unclear, return CLARIFY with exactly one short question.\n"
            "\n"
            "IF EARNINGS INTENT IS TRUE:\n"
            "- Give an 'Earnings levers' style answer for THIS engagement, derived ONLY from the passage:\n"
            "  • What activities qualify to earn.\n"
            "  • Any proof/performance/goal signals that drive payout.\n"
            "  • Note limits/caps/minimums if present.\n"
            "- Keep it to 1–2 sentences or 2–3 bullets. No speculation beyond the passage.\n"
            "FORMAT: return either ANSWER: or CLARIFY: prefix.\n"
             "STYLE:\n"
             "- No hedging, no filler, no ellipses (...).\n"
             "complete sentences, crisp and informative.\n"
             "- Prefer plain language; keep any figures or counts exactly as written in the passage.\n"
             "- If the passage lists activities/eligibility/requirements, summarize the essentials clearly.\n"
             "FAILSAFE:\n"
            "- If the passage truly lacks the needed fact, return CLARIFY with exactly one short question.\n"
        )
        model = "gpt-4o"
        temperature = 0.2
        user_msg = f"""User asked: "{user_text}"
EARNINGS_INTENT: {str(earnings_intent)}
Use ONLY this passage (field: {field}, title: {title}):
{content}

Respond with either:
CLARIFY: "<one short missing input question>"
or
ANSWER: "<partner-friendly If earnings intent, phrase as 'Earnings levers' for this engagement using only what's here>"
"""

    comp = client.chat.completions.create(
        model=model,
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
