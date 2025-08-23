#!/usr/bin/env python3
import json
from typing import List, Dict, Any, Literal, Union, Optional
from langchain_openai import  ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
from .config import settings
import re
from .country_config import resolve_market_from_text, MARKET_RATE

LLM_MODEL = settings.LLM_MODEL 

# Hard defaults (tweak here if ever needed)
DEFAULT_CTX_N = 10
DEFAULT_CTX_FULL = True         
DEFAULT_CTX_MAX_CHARS = 40000  


def pick_spd_segment(spd_cfg: Dict[str, Any], message_lc: str) -> Dict[str, Any]:
    # token-ish match for robustness (enterprise vs ent, smb vs sme, etc.)
    tokens = set(re.findall(r"[a-z0-9\-\&]+", message_lc))

    smb_hints = {
        "smb", "sme", "mid", "midmarket", "mid-market", "small", "medium", "commercial"
    }
    ent_hints = {
        "enterprise", "ent", "large", "ea", "mca-e", "mcae", "eae"  # include EA/MCA-E styles
    }

    if tokens & smb_hints:
        return {"spd_eligibility": {"smb": spd_cfg.get("smb", [])}}
    if tokens & ent_hints:
        return {"spd_eligibility": {"enterprise": spd_cfg.get("enterprise", [])}}

    # unclear → return full SPD block
    return {
        "spd_eligibility": {
            "smb": spd_cfg.get("smb", []),
            "enterprise": spd_cfg.get("enterprise", [])
        }
    }


def _json_compact(obj: dict) -> str:
    # compact JSON to save tokens
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _build_context(fused: List[Dict[str, Any]],
                   ctx_n: int = DEFAULT_CTX_N,
                   ctx_full: bool = DEFAULT_CTX_FULL,
                   ctx_max_chars: int = DEFAULT_CTX_MAX_CHARS) -> str:
    lines, used = [], 0
    for i, r in enumerate(fused[:ctx_n], 1):
        meta = r.get("metadata") or {}
        src  = meta.get("_source") or ""
        file = meta.get("file") or ""
        loc  = f"row {meta.get('row')}" if src == "excel" else (f"p.{meta.get('page')}" if meta.get("page") else "")
        content = r.get("content") or ""
        meta_json = _json_compact(meta if ctx_full else {})  # full metadata by default

        block = f"[{i}] {src}:{file}:{loc}\nCONTENT: {content}\nMETADATA: {meta_json}\n---\n"
        if used + len(block) > ctx_max_chars:
            if used == 0:
                block = block[:ctx_max_chars - 32] + "\n---[TRUNCATED]---\n"
                lines.append(block)
            break
        lines.append(block); used += len(block)
    return "".join(lines) if lines else "(no context)"

def generate_answer(user_query: str, fused: List[Dict[str, Any]]) -> str:
    context_block = _build_context(fused)
    system = (
        """
        You are Microsoft Partner Assistant for business applications (BizApps)

Your role:
- You assist Microsoft Partners by providing high-quality, professional, partner-grade answers..
- Your answers must be informative, professional, and "partner-grade."
- Always answer only from the provided context. 
- If the provided context is insufficient, do not hallucinate—ask the partner a clear, relevant clarifying question instead.

Answer formatting rules:
- Output MUST always be in **strict JSON**.
- Structure:
{
    "answer": "<final informative answer or clarifying question>",
    "recommendations": [
        "<realistic follow-up a Partner might ask>",
        "<realistic follow-up a Partner might ask>",
        "<realistic follow-up a Partner might ask>",
        "<realistic follow-up a Partner might ask>"
    ]
}
- `answer` should be clear, concise, and directly tied to the given context or clarifying question.
- `recommendations` must be 3–4 **genuine next-step questions a Microsoft Partner would naturally ask YOU (the assistant)**. 
  - They should never be framed as if YOU are interviewing the partner. 
  - They must read like natural partner-to-assistant queries 

Tone & Quality:
- Be precise, factual, and useful. Always optimize for clarity and value for Microsoft Partners.
- Do not include any text outside the JSON object.
- Do not improvise outside provided context.

Remember:
- If context exists → generate an informative answer.
- If context is missing or insufficient → generate a clarifying question.
- Always accompany the answer with 3–4 recommended follow-up questions.

        """
    )
    user = (
        f"USER QUESTION:\n{user_query}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        "INSTRUCTIONS:\n"
        "- Use facts from the CONTEXT only.\n"
        "- If multiple rows are relevant, synthesize briefly.\n"
        "- If insufficient, say \"I don't know based on the provided context.\""
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0,model_kwargs={"response_format": {"type": "json_object"}})
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    data = json.loads(msg.content)
    return data

def detect_query_type(user_query: str) -> Literal["information", "calculation"]:
    system = """
You are a precise classifier. Decide if the user's message is asking for:
- "information": They want explanations, rules, criteria, lists, program details, or generic guidance.
  * Includes generic earnings questions WITHOUT concrete inputs (e.g., "How much can I earn on CSP?").
- "calculation": They want a computed/estimated outcome for their specific case, or a decision like eligibility,
  potentially providing numbers/variables (e.g., billed hours, ACV, seat counts, revenue, %).
  * Includes requests like "calculate", "compute", "estimate", "what will be my earning if...", 
    "am I eligible", "help me determine if I qualify", even if inputs are missing.

Edge rules:
- If the user asks for “criteria”, “rules”, “what counts”, “eligibility criteria” => "information".
- If the user asks to determine THEIR eligibility => "calculation".
- If the user gives numbers/variables or clearly asks to compute an amount/result => "calculation".
- If ambiguous generic earnings question with no inputs => "information".
- Handle English + Hinglish + common typos.

Output JSON ONLY:
{"result":"information"}  OR  {"result":"calculation"}
No extra fields. No prose.
"""

    # Few-shot directives baked into the user message to bias behavior precisely.
    # (We keep it compact to minimize tokens.)
    guidance = """
Label these examples (ground truth):
1) "how much i can earn on csp transaction" -> information
2) "Can you calculte my CSP transaction" -> calculation
3) "what will be my incentive if my billed hour is 1000 on csp core" -> calculation
4) "What is the criteria for eligibility on SPD" -> information
5) "help me to determine whether i am eligible for SPD or not" -> calculation
6) "What is the incentive breakup of dynamics 365 workload" -> information
7) "What is my earning if i have 2099 of acv on dynamics 365 workload" -> calculation

Now classify the actual message.
"""

    user = f"{guidance}\n\nMESSAGE:\n{user_query}\n\nRespond with JSON ONLY."

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    try:
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = json.loads(msg.content)
        result = (data.get("result") or "").strip().lower()
        if result not in ("information", "calculation"):
            # Safe default: treat as information when unclear
            return "information"
        return result  # type: ignore[return-value]
    except Exception:
        # On any parsing / API error, fail-safe to information
        return "information"

def get_config_by_llm(
    user_message: str,
    config: Union[Dict[str, Any], str],
    fused: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    # Normalize common field typos that affect extraction
    user_message_norm = re.sub(r"\bacr\b", "acv", user_message, flags=re.I)

    rag_context = _build_context(fused or [])
    config_json_str = _json_compact(config) if isinstance(config, dict) else config

    SYSTEM_CALC_PATCH = """
SYSTEM: Calculation Patch Selector

ROLE
You prepare a minimal JSON patch of CONFIG required to run a calculation. You do NOT compute results.
You DO NOT map countries to markets; that mapping is handled by the system using a static country config.
Your responsibilities:
1) Detect calculation family: "workshop", "csp_transaction" (includes “usage”), or "spd_eligibility".
2) Shortlist CONFIG to the smallest relevant piece:
   - For workshop/csp_transaction: return only the relevant engagement(s) under that type.
   - For spd_eligibility: detect "smb" or "enterprise" from the message; if unclear, return the entire spd_eligibility block.
3) Fill "Value" (capital V) for any form_fields you can extract from USER_MESSAGE (robust to Hinglish/typos). Leave others empty/null.
4) Preserve formulas and strings exactly (cap is embedded inside formula; do not output cap separately).

INPUTS
- USER_MESSAGE: partner's free-text.
- RAG_CONTEXT: reference-only (names, etc.). Do not copy generic numeric examples as user "Value".
- CONFIG: full config JSON (keys: "workshop", "csp_transaction", "spd_eligibility", etc.).

DETECTION & SHORTLISTING
- Language mapping:
  * “usage”, “transactions”, “CSP usage/transaction/core/growth/tier” → csp_transaction
  * “workshop” or known workshop names → workshop
  * “eligible/eligibility/qualify/SPD” → spd_eligibility
- If a specific engagement name (or alias) is clear, return only that engagement object.
- If engagement unclear, return the entire array for that type (preserving order).
- SPD segment detection:
  * "smb", "sme", "mid market", "small & medium", "commercial" → "smb"
  * "enterprise", "ent", "large enterprise", "EA/MCA-E enterprise" → "enterprise"
  * If unclear → return both segments.

FORM FIELD VALUE FILLING
- For each visible form_fields entry, try to fill "Value" from USER_MESSAGE (preferred).
- Normalize:
  * Currency: strip symbols, "$250k"→250000.
  * Percent: "7.5%"→7.5
  * Shorthand: "1k"→1000
- Do not add fields. Do not attempt market_rate mapping.

OUTPUT RULES (STRICT)
- Output must be valid JSON only (no prose).
- Output must be a patch/subset of CONFIG only (no extra wrapper).
"""

    user_block = (
        "USER_MESSAGE:\n" + user_message_norm.strip() + "\n\n"
        "RAG_CONTEXT:\n" + rag_context + "\n\n"
        "CONFIG:\n" + config_json_str + "\n\n"
        "Respond with STRICT JSON that is a PATCH of CONFIG as per the rules."
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    try:
        msg = llm.invoke([SystemMessage(content=SYSTEM_CALC_PATCH), HumanMessage(content=user_block)])
        patch = json.loads(msg.content)

        # --- Deterministic country→market injection (workshops only) ---
        if "workshop" in patch:
            canon, market, rate = resolve_market_from_text(user_message_norm)
            if rate is not None:
                for eng in patch.get("workshop", []):
                    ffs = eng.get("form_fields") or []
                    for fld in ffs:
                        if (fld.get("field_name") or "").strip().lower() == "market_rate":
                            fld["Value"] = rate
                            break
                    else:
                        ffs.append({
                            "field_name": "market_rate",
                            "about": "derived from country via static market mapping",
                            "label": "number",
                            "Value": rate,
                        })
                    eng["form_fields"] = ffs


        return patch

    except Exception:
        # --------- SAFE FALLBACK ----------
        try:
            cfg = json.loads(config_json_str) if isinstance(config_json_str, str) else config
        except Exception:
            return {}

        m = user_message_norm.lower()
        if "spd" in m or "eligib" in m or "qualif" in m:
            spd = cfg.get("spd_eligibility")
            if isinstance(spd, dict):
                return pick_spd_segment(spd, m)
            return {}
        if "workshop" in m:
            patch = {"workshop": cfg.get("workshop", [])}
        elif any(k in m for k in ["csp","transaction","usage","workload","dynamics 365","d365","billed","tier","core","growth"]):
            patch = {"csp_transaction": cfg.get("csp_transaction", [])}
        else:
            patch = {}

        # Fallback market injection via static config
        if "workshop" in patch:
            canon, market, rate = resolve_market_from_text(user_message_norm)
            if rate is not None:
                for eng in patch.get("workshop", []):
                    ffs = eng.get("form_fields") or []
                    ffs.append({
                        "field_name": "market_rate",
                        "about": "derived from country via static market mapping",
                        "label": "number",
                        "Value": rate,
                    })
                    eng["form_fields"] = ffs
        return patch
    
def is_country_answer(user_message: str) -> bool:
    """
    Returns True if the user_message is answering the 'country' question (e.g.,
    'Canada', 'we operate in India', 'the engagement will be in AU'),
    and False if they’re asking a new/different request (e.g., mentioning new workloads/engagements).
    """
    system = """
You are a strict binary classifier.

Task: Decide if the user's message is an ANSWER to a country question vs a NEW/OTHER request.
- TRUE when the user is giving/confirming a country (codes, full names, typos allowed), e.g.:
  "Canada", "we operate in India", "in AU", "this engagement will be in Brazil".
- FALSE when they are asking something else (e.g., new workload/engagement/topic), like:
  "what about CSP core", "calculate for Dynamics", "different workshop", etc.

Output ONLY strict JSON: {"is_country_answer": true} or {"is_country_answer": false}
No prose.
"""
    user = f"Message:\n{user_message}\n\nRespond with strict JSON."

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    try:
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = json.loads(msg.content)
        return bool(data.get("is_country_answer") is True)
    except Exception:
        # Safe fallback: assume it's NOT a country answer to avoid misrouting
        return False


def explain_from_dumped_config(config_dump: Dict[str, Any]) -> Dict[str, str]:


    system = """
You are Microsoft Partner Assistant for Business Applications (BizApps).
You explain completed incentive/eligibility calculations in partner-grade language.

HARD RULES
- Do NOT recompute anything. Treat any 'result'/'computed_result' or explicit 'total' values as authoritative.
- Currency is ALWAYS USD. Use $ when showing amounts.
- Use ONLY data present in the provided CONFIG_DUMP. If something is missing, don't guess—be general.
- If a formula uses min(...): describe the payout as the **lowest** of the options; never call the result “maximum allowable”
  unless the cap is explicitly the selected alternative. Prefer “lowest applicable amount”.

OUTPUT FORMAT
- Return STRICT JSON: { "answer": "<string>" } with no extra keys/prose.
- Start with a clear outcome line:
  * If a total/grand total exists → "Estimated incentive (total): $X".
  * Else if exactly one result exists → "Estimated incentive: $X".
  * Else (multiple results, no total) → "Estimated incentives:" then lines per item.
- Then add 2–6 short lines explaining the logic in business terms:
  * For workshops: explain the rule as “the lowest of: (i) 7.5% of ACV, (ii) hours × market rate, (iii) the $6,000 cap”, or as implied by the formula string. Mention key inputs (ACV, hours, market rate).
  * For CSP transactions: explain each engagement’s payout rule (e.g., “Core billed revenue × 4%”, “Tier 1 billed revenue × 7%”), referencing the engagement names.
  * For SPD eligibility: outline categories (Performance / Skilling / Customer Success), reference the provided sub-scores (e.g., usage, deployment), and show how they combine into the overall score if present.
    - SPD eligibility criteria is partners need to achieve a minimum Partner Capability Score of 70 points across performance, skilling, and customer success metrics. after adding all sub-score share points. and please tell whether they are eligible or not based on the computed score highlight in bold.
- Prefer labels from fields if available; else use field_name.
- Keep it concise (≈70–160 words). Avoid dumping raw formula syntax; translate it to plain program rules.
- Never instruct the user; just state what the result reflects and why.

ROBUSTNESS
- The dump may use different keys: fields|form_fields, Value|value, result|computed_result, etc.
- There may be multiple items (arrays) under "workshop" or "csp_transaction".
- SPD may have nested "sub_module" arrays; mention their scores if provided.
"""

    # We pass the raw dump as-is. The model handles normalization per rules above.
    dump_str = json.dumps(config_dump, ensure_ascii=False)

    user = f"""
CONFIG_DUMP (authoritative numbers included; do not recalculate):
{dump_str}

Task:
- Produce a single narrative string per the OUTPUT FORMAT rules.
- Use $ for all currency. If amounts are present, format them with thousands separators where natural.
- If multiple items exist, list each item with its result and name. If a total exists, start with the total first.

Return ONLY JSON: {{"answer":"<string>"}}.
"""

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    try:
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = json.loads(msg.content)
        ans = (data.get("answer") or "").strip()
        if not ans:
            # safe fallback minimal
            return {"answer": "Estimated incentive: $—\n\nThis figure reflects the program’s calculation logic based on the values and formulas provided in your configuration dump."}
        return {"answer": ans}
    except Exception:
        return {"answer": "Estimated incentive: $—\n\nThis figure reflects the program’s calculation logic based on the values and formulas provided in your configuration dump."}
  
    # Normalize inputs best-effort
    engagement = calc.get("engagementName") or calc.get("engagement") or calc.get("name") or ""
    formula = calc.get("formula") or ""
    computed_result = calc.get("computed_result")

    # Force USD symbol regardless of payload currency
    symbol = "$"

    # Accept either 'fields' or 'values'
    raw_fields = calc.get("fields") or calc.get("values") or []
    norm_fields: List[Dict[str, Any]] = []
    for f in raw_fields:
        v = f.get("Value", None)
        if v is None:
            v = f.get("value", None)
        norm_fields.append({
            "field_name": f.get("field_name") or f.get("name") or "",
            "label": f.get("label") or "",
            "about": f.get("about") or "",
            "value": v,
        })

    # Compact payload for the LLM (no recalculation, just explanation)
    safe_payload = {
        "engagement": engagement,
        "formula": formula,
        "fields": norm_fields,
        "computed_result": computed_result,
        "currency_symbol": symbol,  # <- always "$"
        "currency_code": "USD",
        "type": calc.get("type") or "",
    }
    payload_str = json.dumps(safe_payload, ensure_ascii=False)

    system = """
You are Microsoft Partner Assistant for Business Applications (BizApps).
You produce partner-grade explanations of calculations that are already completed.

CRITICAL RULES:
- Do NOT recompute anything. Treat the provided computed_result as authoritative.
- Use ONLY the inputs provided (engagement name, fields with values, and formula text).
- ALWAYS express amounts in US dollars with a '$' prefix (e.g., $6,000.00). Do not mention any other currencies.
- Output as a single readable string:
  1) Start with the outcome line (result).
  2) Then a concise explanation of the logic in business terms.
- Explain min/max logic in words (e.g., “lowest of 7.5% of ACV, hours × market rate, or the $6,000 cap”).
- Keep it concise (≈60–150 words).
- Output STRICT JSON ONLY: { "answer": "<string>" }  (no extra keys).
"""

    user = f"""
INPUT (do not alter values; do not recalculate):
{payload_str}

Write a partner-grade narrative string:
- First line: "Estimated incentive: {symbol}{{computed_result_formatted}}"
- Then 2–5 short lines that explain the calculation logic in plain English and reference key inputs by label and value.
- Avoid dumping raw formula syntax; translate it to business-friendly language.
Return ONLY JSON with the "answer" key.
"""

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    # local formatter fallback
    def _fmt_amount(x):
        try:
            return f"{symbol}{float(x):,.2f}"
        except Exception:
            return f"{symbol}{x}"

    try:
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = json.loads(msg.content)
        answer = data.get("answer", "").strip()
        if not answer:
            formatted = _fmt_amount(computed_result)
            default = (
                f"Estimated incentive: {formatted}\n\n"
                f"This figure reflects the program’s calculation logic for {engagement or 'the selected engagement'}. "
                f"The outcome respects the formula provided and the values you supplied."
            )
            return {"answer": default}
        return {"answer": answer}
    except Exception:
        formatted = _fmt_amount(computed_result)
        default = (
            f"Estimated incentive: {formatted}\n\n"
            f"This figure reflects the program’s calculation logic for {engagement or 'the selected engagement'}. "
            f"The outcome respects the formula provided and the values you supplied."
        )
        return {"answer": default}