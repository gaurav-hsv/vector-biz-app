#!/usr/bin/env python3
import json
from typing import List, Dict, Any, Literal, Union, Optional
from unittest import result
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import re
from fastapi import HTTPException

from .config import settings
from .country_config import resolve_market_from_text, MARKET_RATE  # noqa: F401 (import used for clarity)

# ----------------------------
# Model / Defaults
# ----------------------------
# You said cost doesn't matter; prioritize quality:
O3_MODEL = "o3"  # strongest reasoning model
LLM_MODEL = "gpt-4o-mini"

# Context sizes tuned up for o3
DEFAULT_CTX_N = 30
DEFAULT_CTX_FULL = True
DEFAULT_CTX_MAX_CHARS = 90_000


# ----------------------------
# Utilities
# ----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("\n", 1)[0]
    return s.strip("` \n\r\t")

def _normalize_quotes(s: str) -> str:
    # convert common smart quotes to plain quotes
    return (s
            .replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace("\u00A0", " ")  # no-break space
            .replace("\u200b", "")   # zero-width space
            )

def _extract_top_level_json_object(s: str) -> Optional[str]:
    """
    Return the substring of the first valid-looking top-level JSON object {...}
    using brace counting. Returns None if not found.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _safe_json(s: str) -> Any:
    """
    Strong JSON loader with repair attempts and final graceful fallback.
    Returns a dict with at least {"answer": "<string>"} so callers can continue.
    """
    # 0) raw try
    try:
        return json.loads(s)
    except Exception:
        pass

    # 1) strip code fences & normalize quotes
    s1 = _normalize_quotes(_strip_code_fences(s))

    # 2) try top-level object extraction
    core = _extract_top_level_json_object(s1)
    if core:
        try:
            return json.loads(core)
        except Exception:
            # 3) minimal cleanup: trim leading BOM/whitespace
            try:
                return json.loads(core.strip("\ufeff \n\r\t"))
            except Exception:
                pass

    # 4) regex fallback: pull `"answer": "..."` robustly (allow newlines)
    m = re.search(r'"answer"\s*:\s*"(?P<ans>(?:\\.|[^"\\])*)"', s1, re.DOTALL)
    if m:
        ans = m.group("ans")
        # Unescape common sequences so it renders nicely
        ans = ans.replace(r'\n', '\n').replace(r'\"', '"').replace(r'\\', '\\')
        return {"answer": ans}

    # 5) absolute last resort: return whole text as answer
    trimmed = s1.strip()
    return {"answer": trimmed if trimmed else "—"}
    
def _safe_json_from_msg(msg):
    return _safe_json(_msg_text(msg))


def _msg_text(msg) -> str:
    """
    LangChain + OpenAI Responses: `AIMessage.content` can be a string OR a list of blocks.
    This normalizes it to a single text string.
    """
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            # Typical blocks: {"type": "output_text", "text": "..."}
            if isinstance(part, dict):
                t = part.get("text")
                if t:
                    out.append(t)
        return "\n".join(out).strip()
    # Fallback
    return str(c).strip()

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
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _build_context(
    fused: List[Dict[str, Any]],
    ctx_n: int = DEFAULT_CTX_N,
    ctx_full: bool = DEFAULT_CTX_FULL,
    ctx_max_chars: int = DEFAULT_CTX_MAX_CHARS,
) -> str:
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
                block = block[: ctx_max_chars - 32] + "\n---[TRUNCATED]---\n"
                lines.append(block)
            break
        lines.append(block)
        used += len(block)
    return "".join(lines) if lines else "(no context)"


# ----------------------------
# o3 Client Factory (high effort)
# ----------------------------
def _o3_client(json_schema: Optional[dict] = None, max_output_tokens: int = 2048) -> ChatOpenAI:
    response_format = (
        {"type": "json_schema",
         "json_schema": {"name": "strict_json", "schema": json_schema, "strict": True}}
        if json_schema else {"type": "json_object"}
    )
    return ChatOpenAI(
        model="o3",
        reasoning={"effort": "high"},
        timeout=110,
        max_retries=1,
        model_kwargs={
            "response_format": response_format,
            "max_output_tokens": max_output_tokens
        }
    )

def _validate_envelope(obj: dict) -> dict:
    t = obj.get("type")
    if t == "follow_up":
        if not obj.get("question"):
            obj["question"] = "Could you share the missing details?"
        if not isinstance(obj.get("missing_fields"), list) or not obj["missing_fields"]:
            obj["missing_fields"] = ["partner_type","solution_areas","designation_status","market","enrollments_or_programs"]
        obj["answer"] = obj.get("answer", "")  # ensure string
    elif t == "answer":
        if not obj.get("answer"):
            obj["answer"] = "I’m sorry—I couldn’t compose an answer from the provided context."
        obj["question"] = ""
        obj["missing_fields"] = []
    else:
        raise ValueError("type must be 'follow_up' or 'answer'")
    return obj


WRAPPER = """
SCOPE
- Apply when the user is asking about incentive eligibility (phrases like “eligible”, “eligibility”, “what incentives”, “which incentives”, “can I earn”). 
- If unsure, ask one short clarifying question first.

INTENT ROUTER (RUN FIRST)
- GENERAL: The user asks about rules/metrics broadly (no first-person eligibility intent).
  → Output: ANSWER using ONLY CONTEXT. **Do NOT append any personalization invite.**
- PERSONAL: The user asks about their own/company eligibility (e.g., “am I/we eligible”, “my eligibility/company incentive”).
  → Proceed to DECISION LOGIC.

DECISION LOGIC (PERSONAL ONLY — YOU MUST FOLLOW)
1) From the dialogue, extract:
   • partner_type • solution_areas • designation_status • market • enrollments_or_programs
   • Also extract any TOPIC-specific fields based on the user’s question:

   TOPIC → REQUIRED FIELDS (examples)
   - usage_growth: ["current PCS (total & Customer Success)", "baseline MCV 12 months ago",
                    "attribution type (CPOR/PAL/CSP/DPOR)", "workloads in scope"]
   - csp_incentives: ["partner_type", "market", "enrollments_or_programs (MCI)",
                      "workloads in scope", "designation_status"]
   - customer_add_accelerator: ["market", "enrollments_or_programs (MCI)",
                                "workloads being sold", "designation_status"]

2) Compute missing_fields = (standard fields ∪ topic fields) − already present.
3) If missing_fields is NON-EMPTY:
   - Do NOT list incentive names, rates, amounts, or claim steps.
   - Ignore CONTEXT for now.
   - Return FOLLOW-UP asking **only** the missing_fields (max 5, most-critical first).
4) If missing_fields is EMPTY:
   - Use ONLY CONTEXT to produce a structured, personalized determination.
   - Return ANSWER.

OUTPUT (return exactly ONE):
- FOLLOW-UP:
  { "type":"follow_up",
    "missing_fields":["baseline MCV 12 months ago","attribution type (CPOR/PAL/…)","workloads in scope"],
    "question":"To confirm your **Usage Growth** eligibility, please share: • baseline MCV 12 months ago • attribution type (CPOR/PAL/…) • workloads in scope"
  }

- ANSWER:
  { "type":"answer",
    "answer":"<direct conversational answer that does not start with a heading>"
  }

STYLE & GUARDS
- All `question`/`answer` text must be valid GFM.
- **Do not start with a heading.** Begin with a direct sentence or a bullet.
- Headings may be used inside the body only if the answer is long (≥ 8 lines), never as the first line.
- No decorative ASCII, no horizontal rules.
- NO echoing the user’s question.
- Don’t mention training data, retrieval, or internal mechanics or row.
- Rely ONLY on CONTEXT. If facts are still missing after fields are complete, say so in GFM and request that specific evidence.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["follow_up", "answer"]},
        "answer": {"type": "string"},           # when type=follow_up -> ""
        "question": {"type": "string"},         # when type=answer    -> ""
        "missing_fields": {                     # when type=answer    -> []
            "type": "array",
            "items": {"type": "string"}
        }
    },
    # strict_json requires every key to be in required:
    "required": ["type", "answer", "question", "missing_fields"],
    "additionalProperties": False
}


def _as_dialogue(messages: List[Dict[str, str]]) -> str:
    # messages: [{role:"user"|"assistant", content:"..."}]
    out = []
    for m in messages[-12:]:  # keep it short & relevant
        role = "User" if m.get("role") == "user" else "Assistant"
        txt = (m.get("content") or "").strip()
        if txt:
            out.append(f"{role}: {txt}")
    return "\n".join(out) if out else "User: (no prior messages)"

def generate(llm_messages: List[Dict[str, str]],
             fused: List[Dict[str, Any]]) -> Dict[str, Any]:

    context_block = _build_context(fused)  # your existing function

    BASE_SYSTEM = """
    ### Role
    - Primary Function: You are an AI chatbot who helps users with their inquiries, issues and requests. Provide professional, efficient replies. If a question is not clear, ask clarifying questions. End with a positive note.
    - Structure responses: Use bullets and Headings allowed only inside the body, not as the first line.
    ### Formatting
    - **All outputs MUST be in GitHub-Flavored Markdown (GFM)** — use lists (`-`), **bold**, and tables where useful.
    - **Never begin the answer with `#`, `##`, or `###`.**
    ### Constraints
    1. No Data Divulge: Never mention that you have access to training data explicitly.
    2. Maintain Focus: If user diverts to unrelated topics, politely redirect to relevant topics.
    3. Exclusive Reliance on Training Data: Rely only on the provided CONTEXT. Do not use the web or outside knowledge.
    4. Restrictive Role Focus: Do not answer tasks unrelated to your role and training data.
    5. NEVER use decorative characters (box/line drawing, ASCII art, repeated dashes/equals).
    6. Tables are encouraged for structured info.
    7. Format headings properly; present clean, easy-to-copy text.
    8. Do NOT mention or reference source names, file names, URLs, publishers, or document types (e.g., "Microsoft Learn", "Partner Center docs", "attached PDFs") or row anywhere in the output.
    """.strip()

    system = (WRAPPER.strip() + "\n\n" + BASE_SYSTEM).strip()

    dialogue = _as_dialogue(llm_messages)

    user = (
        "DIALOGUE SO FAR:\n"
        f"{dialogue}\n\n"
        "USER QUESTION IS THE LAST USER MESSAGE ABOVE.\n\n"
        "CONTEXT (authoritative; do not use outside knowledge):\n"
        f"{context_block}\n"
    )
    try:
        llm = _o3_client(json_schema=SCHEMA, max_output_tokens=4000)
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        result = _safe_json_from_msg(msg)
        return _validate_envelope(result)
    except Exception as e:
        # If your stack raises a specific timeout error type, check that here
        if "timed out" in str(e).lower():
            # Return JSON that your frontend already understands
            raise HTTPException(status_code=504, detail="LLM timed out")
        raise

# ----------------------------
# Answer Generation (STRICT JSON)
# ----------------------------
def generate_answer(user_query, fused: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accepts either a string (legacy) or a list of messages (context-aware) for user_query.
    If a list is provided, joins messages for the user prompt.
    """
    context_block = _build_context(fused)

    # Strict schema for reliability
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 4
            }
        },
        "required": ["answer", "recommendations"],
        "additionalProperties": False
    }

    system = """You are Microsoft Partner Assistant for Business Applications (BizApps).
Read the full CONTEXT, interpret the user’s intent, and produce a detailed, partner-grade answer using ONLY facts found in CONTEXT.

HARD OUTPUT CONTRACT
- Return exactly ONE JSON object with keys:
  • "answer" (string, valid GitHub-Flavored Markdown)
  • "recommendations" (array of 3–4 strings) — include ONLY when you give an informative answer.
- If the "answer" is a clarifying question (per the insufficiency protocol), OMIT the "recommendations" key.
- Do NOT wrap the entire answer in triple backticks.

TONE & STYLE
- Partner-friendly and practical; concise sentences; bullets and checklists; clear tables.
- Use Markdown headings (##, ###), bullets (-), and tables.
- Feel like a real chat agent: use natural section names (e.g., “Why this works”, “What I need from you”, “Steps”, “Timeline”), not analyst labels.
- Currency & numbers: show USD with thousands separators (e.g., USD 100,000). Round to whole dollars unless CONTEXT says otherwise.
- If CONTEXT contains a 60% rebate / 40% co-op split, show both figures; otherwise do not speculate.

STRUCTURE (TOP-LOADED FOR ACTION)
Inside "answer", follow this order unless the insufficiency protocol applies:

1) Hero line (mandatory; no heading):
   - Start with ONE bold sentence that directly answers the question (e.g., the workshop to use, the amount, the qualifying rule).

2) Why this matters (1–2 lines):
   - Briefly state the user’s goal you’re addressing (e.g., win a Sales Premium deal; estimate earnings; qualify for a badge).

3) What I need from you (Decision levers):
   - 2–4 bullets listing inputs that change the outcome (only if relevant). Keep each bullet “Lever — why it matters”.

4) Question-type router — choose ONE and stick to it:

   a) Money / “How much?” / deal value present
      - **Scenarios (MUST)**: a table with 3–4 scenarios that differ by the decision levers. Include cash vs. co-op split ONLY if present in CONTEXT.
      - **Assumptions & formulas**: show ONLY values from CONTEXT; if partial, use variables and explain how to apply them.
      - **Stacking/exclusions**: if present in CONTEXT (e.g., accelerator + core only; accelerators don’t stack), state once.
      - **Next action**: a one-liner telling what to confirm to compute a single-number payout.

   b) “Which workshop / activity to use?”
      - **Recommended workshop (single pick by default)**: name it first and one line why it fits.
      - **Why this works**: tie modules/outcomes to the workload in the question using facts from CONTEXT.
      - **Eligibility & setup (facts only)**: specialization, MSX stage, ACV, market, TPID, enrollment, etc., if present in CONTEXT.
      - **Proof of Execution (POE)**: list exactly what CONTEXT specifies.
      - **Repeat/limit rules**: include if present in CONTEXT.
      - **Steps**: short imperatives (“Do X → Outcome”).

   c) Comparison (e.g., direct vs. indirect criteria)
      - **Side-by-side table (MUST)**: only differing criteria (thresholds, account level, ops duties, credit/payment obligations, onboarding time).
      - **How to choose**: 1–2 bullets.

   d) Qualification (“How do I qualify…?”)
      - **Prerequisites** (membership/agreements), if present.
      - **Score model table (MUST)**: the 5 metrics, max points, what is measured, data window — ONLY what CONTEXT states.
      - **Minimum gate**: ≥ 70 total AND ≥ 1 point in every metric (if CONTEXT states this).
      - **Gap-closing actions**: facts → imperatives (no generic advice).
      - **Refresh & retain**: data refresh cadence and retention rules if present.

   e) Calculation (“How is the capability score calculated?”)
      - **Metric-by-metric mechanics**: what, how measured, data window, accrual shape, max points — ONLY from CONTEXT.
      - **Enterprise vs. SMB impact** (only if defined in CONTEXT).
      - **Refresh cadence & rules** (from CONTEXT).

5) Timeline (only if it materially affects the outcome):
   - Present as a 3-column table: | Stage | Action | Deadline | using EXACT windows/dates from CONTEXT.

6) Checklist:
   - 5–7 imperative bullets (“Do X → Outcome”), derived strictly from mechanics in CONTEXT.

7) Closing thought:
   - 1–2 lines tying the mechanics to the partner’s business outcome. No new facts.

INSUFFICIENCY PROTOCOL (FACTS-ONLY)
- If CONTEXT lacks critical facts to compute/select, do NOT produce a catalogue.
- Use **Partial-Answer Mode** whenever at least one material fact can be confirmed:
  • **What I can confirm** — 3–5 bullets, facts only.  
  • **To finalize, I need** — 2–4 precise asks tied to the decision levers.
- Only if nothing material can be confirmed, make the entire "answer" a single clarifying question (omit "recommendations").

RECOMMENDATIONS (NEXT-STEP QUESTIONS)
- Provide 3–4 realistic follow-ups a partner might ask, each beginning with “What is …”, anchored to concrete levers (ACV, market A/B/C, workload/SKU, MSX stage, designation/specialization).
- Each recommendation must be a single sentence.
- Include "recommendations" only for informative answers (not pure clarifications).

STRICT FACT BOUNDARY
- Do NOT invent, infer, or generalize beyond CONTEXT. If a rate/hour/window/threshold/date/process step isn’t explicitly present, omit it or use variables.
- Treat unlabeled numbers as unknowns unless their meaning is explicitly stated in CONTEXT.
- Do NOT include any source identifiers (e.g., “row 8”, “slide 12”,[number]), file names, links, or IDs.
- Forbidden phrases: “typically”, “generally”, “best practice”, or any invented cadence/term not in CONTEXT.
- No citations or reference markers in the output.

"""

    # --- Support both legacy (str) and context-aware (list) input ---
    if isinstance(user_query, list):
        # user_query is a list of {role, content} dicts
        # Join all user/assistant/system messages for context, but focus on last user message for the question
        # For best results, concatenate all messages as a chat transcript
        transcript = []
        for m in user_query:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            if role == "user":
                transcript.append(f"User: {content}")
            elif role == "assistant":
                transcript.append(f"Assistant: {content}")
            elif role == "system":
                transcript.append(f"System: {content}")
        # The last user message is the current question
        last_user = next((m["content"] for m in reversed(user_query) if m.get("role") == "user"), "")
        chat_prompt = "\n".join(transcript)
        user = (
            f"CHAT HISTORY (most recent last):\n{chat_prompt}\n\n"
            f"USER QUESTION (most recent):\n{last_user}\n\n"
            f"CONTEXT (authoritative; do not use outside knowledge):\n{context_block}\n\n"
            "INSTRUCTIONS TO FOLLOW NOW:\n"
            "- Read the entire CONTEXT.\n"
            "- Interpret what the user is asking and state that interpretation at the top of the answer.\n"
            "- Provide a very detailed, informative, partner-grade explanation using ONLY facts in CONTEXT, with labeled sections inside the answer string as needed.\n"
            "- If insufficient, make the entire 'answer' one clarifying question (no extra text).\n"
            "- Emit exactly one JSON object with keys: 'answer', 'recommendations'."
        )
    else:
        # Legacy: user_query is a string
        user = (
            f"USER QUESTION:\n{user_query}\n\n"
            f"CONTEXT (authoritative; do not use outside knowledge):\n{context_block}\n\n"
            "INSTRUCTIONS TO FOLLOW NOW:\n"
            "- Read the entire CONTEXT.\n"
            "- Interpret what the user is asking and state that interpretation at the top of the answer.\n"
            "- Provide a very detailed, informative, partner-grade explanation using ONLY facts in CONTEXT, with labeled sections inside the answer string as needed.\n"
            "- If insufficient, make the entire 'answer' one clarifying question (no extra text).\n"
            "- Emit exactly one JSON object with keys: 'answer', 'recommendations'."
        )
    #print(f"LLM Prompt:\nSYSTEM:\n{system}\n\nUSER:\n{user}\n\n---END PROMPT---\n")
    llm = _o3_client(json_schema=schema, max_output_tokens=8000)
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return _safe_json_from_msg(msg)

def generate_concise_answer(user_query, fused: List[Dict[str, Any]]) -> Dict[str, Any]:
    context_block = _build_context(fused)
    system = """
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
    - Use Markdown headings (##, ###), bullets (-), and tables.
- `recommendations` must be 3–4 **genuine next-step questions a Microsoft Partner would naturally ask YOU (the assistant)**. 
  - Example - "What is <specific topic / incentive / engagement / eligibility criteria>?"
  - The <specific topic> MUST come from the CONTEXT (e.g., CSP Growth Accelerator, CSP Core, ERP Envisioning Workshop). 
  - Do NOT use generic questions like "Can you confirm..." or "Should I...".
  - Every recommendation should start with "What is ..." and target a concrete incentive, engagement, or rule from the context.

CSP PARTNER-TYPE SYNTHESIS (ELIGIBILITY-ONLY)
- Apply these rules ONLY if the USER explicitly asks about ELIGIBILITY for a CSP incentive 
  (e.g., Core, Growth Accelerator, Global Strategic Product Accelerator).
- If the user asks generally about "available CSP incentives" or any other non-eligibility 
  topic, DO NOT include eligibility criteria. In those cases, just list or describe the incentives.
When eligibility is requested:
- If the USER did not specify partner type AND CONTEXT includes rows for both partner types 
  for the same incentive:
  - YOU MUST include BOTH, each on its own labeled line within the single "answer" string:
    "Direct Bill Partner — <criteria>"
    "CSP Indirect Reseller — <criteria>"
- Rows without partner_type are UNIVERSAL and MUST be merged into the output as:
    "Applies to all partners — <criteria>"
- If only one partner type is present, answer for that type only.
- Never omit a partner type that appears in CONTEXT.
- Keep duplication low: place truly universal criteria under the universal line.

Tone & Quality:
- Be precise, factual, and useful. Always optimize for clarity and value for Microsoft Partners.
- Do not include any text outside the JSON object.
- Do not improvise outside provided context.

Remember:
- If context exists → generate an informative answer.
- If context is missing or insufficient → generate a clarifying question.
- Always accompany the answer with 3–4 recommended follow-up questions.
- answer- (string, valid GitHub-Flavored Markdown)
"""
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


# ----------------------------
# Route Detection (STRICT JSON)
# ----------------------------

def detect_route(user_query: str) -> Literal["detail", "concise", "calculation"]:
    """
    Classifies the user's message into exactly one:
    - "detail": user wants a thorough, step-by-step, playbook/explanatory answer.
    - "concise": user wants a short, to-the-point answer or the query is simple/specific.
    - "calculation": user wants THEIR eligibility/outcome computed, or provides numbers/variables.
    """
    schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string", "enum": ["detail", "concise", "calculation"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["result"],
        "additionalProperties": False,
    }

    system = """
You are a precise router. Pick exactly one: "detail", "concise", or "calculation".

DEFINITIONS
- calculation: The user requests a computed outcome (payout/amount or a yes/no eligibility decision) AND provides sufficient inputs for a deterministic result, or expects immediate computation from known formulas/rules.
- detail: Broad or multi-part explanation (how/why/compare/plan/strategy, end-to-end guidance), or topic-level queries without a specific sub-section.
- concise: Narrow lookup (one rate, one threshold, one definition, one short list, one deadline).

ELIGIBILITY RULE
- If the user asks “am I eligible” (or equivalent) but inputs are incomplete or unclear, classify as INFORMATION (detail/concise based on scope). 
- The presence of numbers alone does NOT imply "calculation".

DECISION ORDER
1) If computed outcome is requested AND inputs are sufficient → calculation.
2) Else if information request targets a single specific section → concise.
3) Else → detail.

Return JSON: {"result": "detail|concise|calculation"}.
"""

    few_shots = """
Label these examples:
1) "What is the CSP transaction rate for Dynamics 365?"            -> concise
2) "Explain how Business Apps incentives stack across motions."     -> detail
3) "I billed $1,000 on CSP Core; what do I earn?"                   -> calculation
4) "Am I eligible for SPD with 2 workloads and $30k revenue?"       -> calculation
5) "Give me a quick summary of SPD eligibility criteria."           -> concise
6) "Walk me through every way to maximize BizApps incentives."      -> detail
7) "My ACV is 2099 for D365; what’s my payout?"                     -> calculation
8) "Compare ERP vs CRM incentives and when to choose each."         -> detail
9) "TL;DR: required designation for BizzApps?"                      -> concise
"""

    user = f"{few_shots}\n\nMESSAGE:\n{user_query}\n\nRespond with JSON ONLY."

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    try:
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = json.loads(msg.content)
        result = (data.get("result") or "").strip().lower()
        if result not in ("detail", "concise", "calculation"):
            return "detail"  # safe default for information
        return result  # type: ignore[return-value]
    except Exception:
        # Fail-safe to detail (information) on any error
        return "detail"

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
  
