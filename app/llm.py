# app/llm.py
#!/usr/bin/env python3
import json
import re
from typing import Any, Dict, Iterator, List, Optional
from decimal import Decimal
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

O3_MODEL = "o3"  # strongest reasoning model

# Context sizes tuned up for o3
DEFAULT_CTX_N = 30
DEFAULT_CTX_FULL = True
DEFAULT_CTX_MAX_CHARS = 90_000
FINAL_PREFIX = "@@FINAL@@"  # sentinel that precedes the final compact JSON


# ----------------------------
# Utilities
# ----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("\n", 1)[0]
    return s.strip("` \n\r\t")


def _normalize_quotes(s: str) -> str:
    return (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("\u00A0", " ")
        .replace("\u200b", "")
    )


def _extract_top_level_json_object(s: str) -> Optional[str]:
    """Return the first valid-looking top-level JSON object {...} using brace counting."""
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
                return s[start : i + 1]
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
            try:
                return json.loads(core.strip("\ufeff \n\r\t"))
            except Exception:
                pass

    # 3) regex fallback: pull `"answer": "..."` robustly (allow newlines)
    m = re.search(r'"answer"\s*:\s*"(?P<ans>(?:\\.|[^"\\])*)"', s1, re.DOTALL)
    if m:
        ans = m.group("ans")
        ans = ans.replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")
        return {"answer": ans}

    # 4) absolute last resort: return whole text as answer
    trimmed = s1.strip()
    return {"answer": trimmed if trimmed else "—"}

def _json_compact(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _json_default_decimal(o: Any):
    if isinstance(o, Decimal):
        return str(o)  # keep precision-friendly on wire/prompt
    raise TypeError

def _prune_none(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _prune_none(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [ _prune_none(x) for x in v if x is not None ]
    return v

def _drop_keys(v: Any, banned: set[str]) -> Any:
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k in banned:
                continue
            out[k] = _drop_keys(x, banned)
        return out
    if isinstance(v, list):
        return [ _drop_keys(x, banned) for x in v ]
    return v

def _project_session_context(session: Optional[Dict[str, Any]]) -> str:
    """
    Build a compact JSON block from session fields we care about.
    - Keeps only `user_profile` and (optionally) `memory.state.last_calc` if present.
    - Drops any `calculated_value` keys and None values.
    - Decimal-safe; length-capped for prompt hygiene.
    """
    if not session or not isinstance(session, dict):
        return ""  # no-op

    snapshot: Dict[str, Any] = {}

    # Prefer root key if you are writing user_profile there (your new updater does).
    pp = session.get("user_profile")
    # Optional: also allow legacy location under memory.state
    mem = session.get("memory") or {}
    state = mem.get("state") or {}
    last_calc = state.get("last_calc")

    if isinstance(pp, dict) and pp:
        snapshot["user_profile"] = pp
    if last_calc:
        snapshot["last_calc"] = last_calc

    if not snapshot:
        return ""

    # Drop noisy/computed fields everywhere
    snapshot = _drop_keys(snapshot, banned={"calculated_value"})
    # Remove Nones
    snapshot = _prune_none(snapshot)

    try:
        text = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), default=_json_default_decimal)
    except TypeError:
        # extremely defensive: if something inside wasn't serializable
        text = _json_compact(snapshot)

    return text

def _build_context(
    fused: List[Dict[str, Any]],
    ctx_n: int = DEFAULT_CTX_N,
    ctx_full: bool = DEFAULT_CTX_FULL,
    ctx_max_chars: int = DEFAULT_CTX_MAX_CHARS,
) -> str:
    lines, used = [], 0
    for i, r in enumerate(fused[:ctx_n], 1):
        meta = r.get("metadata") or {}
        src = meta.get("_source") or ""
        file = meta.get("file") or ""
        loc = (
            f"row {meta.get('row')}"
            if src == "excel"
            else (f"p.{meta.get('page')}" if meta.get("page") else "")
        )
        content = r.get("content") or ""
        meta_json = _json_compact(meta if ctx_full else {})

        block = (
            f"[{i}] {src}:{file}:{loc}\nCONTENT: {content}\nMETADATA: {meta_json}\n---\n"
        )
        if used + len(block) > ctx_max_chars:
            if used == 0:
                block = block[: ctx_max_chars - 32] + "\n---[TRUNCATED]---\n"
                lines.append(block)
            break
        lines.append(block)
        used += len(block)
    return "".join(lines) if lines else "(no context)"


def _as_dialogue(messages: List[Dict[str, str]]) -> str:
    # messages: [{role:"user"|"assistant", content:"..."}]
    out = []
    for m in messages[-12:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        txt = (m.get("content") or "").strip()
        if txt:
            out.append(f"{role}: {txt}")
    return "\n".join(out) if out else "User: (no prior messages)"


# ----------------------------
# o3 Client Factories
# ----------------------------
def _o3_stream_client(max_output_tokens: int = 4000) -> ChatOpenAI:
    """
    Streaming client for o3. IMPORTANT: Do NOT set response_format/json_schema for token streaming.
    """
    return ChatOpenAI(
        model=O3_MODEL,
        reasoning={"effort": "high"},
        timeout=90,  # Reduced timeout for better connection management
        max_retries=1,
        # No response_format in streaming mode; allow free-form streaming text.
        model_kwargs={"max_output_tokens": max_output_tokens},
        # Add connection timeout settings
        request_timeout=90,
    )


def stream_answer(
    llm_messages: List[Dict[str, str]],
    fused: List[Dict[str, Any]],
    session: Optional[Dict[str, Any]] = None,
    max_output_tokens: int = 4000,
) -> Iterator[Dict[str, Any]]:
    """
    Yields a sequence of dict events for NDJSON:
      {"event":"delta","text": "..."} repeatedly,
      then {"event":"final","result": {"type","text","missing_fields"}},
      or {"event":"error","detail": "..."} on failure.

    NOTE: This does true token streaming with a final @@FINAL@@{...} trailer.
    """
    context_block = _build_context(fused)
    dialogue = _as_dialogue(llm_messages)
    session_block = _project_session_context(session)

    system = """"You are an AI Chatbot that provides guidance to Microsoft partners on Business Applications solutions. If a question is not clear, ask clarifying questions. End with a positive note.

GLOBAL FORMATTING

- Output must be valid GitHub-Flavored Markdown (GFM).
- No decorative ASCII or horizontal rules.
- Do **not** start with a heading; begin with a direct sentence.
- Use short paragraphs or bullets; prefer tables for structured info.
- Table safety (streaming): Prefer bullets unless the table is short (≤10 rows, ≤6 columns). If using a table, stream header and separator first, then complete rows; avoid trailing pipes. If rows may be long or partial mid-stream, switch to bullets.
- Keep answers focused and copy-friendly.
- Do not echo the user’s question.
- Do **not** mention sources, file names, URLs, publishers, document types, training data, retrieval, internal mechanics, or “rows”.

GLOBAL CONSTRAINTS
- Do not guess or fabricate. If you must proceed with assumptions, state them explicitly.
- Rely **only** on the provided CONTEXT. Do not use the web or outside knowledge.
- Stay on topic; if the user diverts, gently redirect.
- Never ask for rates/caps/percentages (derive from CONTEXT); if market is missing, ask only for country.

DEFINITIONS
- MATERIAL DEPENDENCE (MD): If an answer depends on any case variables (partner_type, market/country, MCI, SPD, attribution_type, workloads, volume/ACR/MCV, time window) and the user refers to their own customer/tenant/deal, prefer PERSONAL routing.
- ELIGIBILITY SIGNALS (ES): The minimal set of facts required to decide the user’s question or compute a result. Derive ES from the policy/rules in CONTEXT for the detected topic; do not hardcode field names and do not expose ES lists to the user.

**SCOPE**
- **Domain (Business Applications only):** Microsoft Commerce Platform; Microsoft Commerce Incentives (**MCI**, the umbrella for Business Applications incentives); activity-based incentives under MCI (MCI-funded engagements/workshops); transaction-based incentives under MCI (CSP incentives on Business Applications transactions); Solutions Partner Designation (SPD); transition from legacy to New Commerce Experience (NCE); Partner Center tools, processes, and troubleshooting — all limited to the Business Applications solution area and solution plays.

- **Covered topics (Business Applications only):**
  - **MCI umbrella & incentive types:** Treat MCI as the umbrella. There are two incentive types:
    1) **Activity-based**: MCI-funded engagements/workshops (e.g., pre-sales workshops, accelerators). Answer partner eligibility, customer eligibility, prerequisites, timelines, payout calculations, execution requirements, Proof of execution, performance requirements, sub-contracting, strictly per CONTEXT. Optimization or case-specific questions follow the INTENT ROUTER.
    2) **Transaction-based (CSP)**: CSP incentives on Business Applications transactions. Cover program/enrollment types, partner eligibility, calculation mechanics, and optimization — case-specific questions follow the INTENT ROUTER. Compute CSP incentives in this order: **Core → Strategic Product Accelerator(s) → Growth**, showing the core subtotal, each stacked component, and the final total. Apply only rates/caps from CONTEXT, avoid double-counting, and follow any exclusivity/cap rules exactly as stated in CONTEXT.
  - **Solutions Partner Designation (SPD):** Answer queries on SPD categories and Partner Capability Score (PCS), PCS pillars/metrics and how they are calculated, how SPD eligibility is determined from PCS, benefits of holding SPD, and which incentives require SPD. Compute **SPD eligibility** (not payouts) using CONTEXT; advise on increasing PCS where applicable. Within this, also share details on points breakdown, calculation of points for each criteria, customer association, solution partner designation anniversary, skilling requirements, workload coverage, etc. Make sure to differentiate between SMB and Enterprise path.
  - **NCE transition & Partner Center:** Explain NCE transition considerations and Partner Center tools/processes/troubleshooting **as documented in CONTEXT** (no web).

IN-SCOPE CAPABILITIES

- Understand the user’s question: detect intent and the core ask
- Analyze information needs: determine what facts are required from CONTEXT to answer or compute.
- Retrieve & infer from CONTEXT: auto-populate anything derivable from the knowledge base.
- Minimal follow-ups: if human input is still required, …ask only for the smallest necessary set (most critical first), in natural language (no field names).
- Decide & compute: when sufficient information exists, determine eligibility, explain Business Applications workshops/engagements/MCI, and compute **Business Applications CSP and workshop payouts** per CONTEXT rules; **calculate SPD eligibility**.
- Deliver final answer: synthesize clearly and concisely with bullets/tables where helpful.

- **SESSION DATA UTILIZATION**: When session data is available, actively use it to:
  * Always use it to pre-populate eligibility and incentive checks.
  * Personalize responses based on their partner profile
  * Skip redundant questions about information already provided
  * Reference their specific business context and solution partner designations and specializations
  * Use `user_profile.partner_type` for partner eligibility checks
  * Use `user_profile.solution_designations` for designation status
  * Use `user_profile.specializations` for partner specialization status
  * Use metric values (billed_revenue, global_tier1, etc.) for calculations
  * **When extracting incentive rules or conditions from CONTEXT documents, always cross-check those conditions against the available `user_profile` values before concluding eligibility. Do not assume eligibility from text alone; verify it using partner_type, designations, specializations, revenue thresholds, or other profile metrics where applicable.**
  * Use `calculation_breakdown` data if available for incentive computations
  * For any “what am I eligible for” or “how can I maximize” type questions:
    - First, clearly list the incentives the partner is currently eligible for, based on their profile.
    - Then, explicitly identify incentives they are not eligible for, and explain what requirements or actions would be needed to unlock them.

INTENT ROUTER (RUN FIRST)
- PERSONAL: Route here if any of the following are true:
  • First-person + owned entities: mentions like “I/we/my customer/our tenant/account/deal”.
  • Case-specific ask: optimization/strategy or a computation for a specific customer/partner scenario (e.g., “maximize”, “highest”, “best way to earn”, “from a single customer”), even if no numbers are provided.
  • Presence of any case variable in the message: partner_type, market/country, enrollments/programs (MCI), designation_status (SPD), attribution_type (CPOR/PAL/CSP/DPOR), workloads, volume/ACR/MCV, or a time window.
  → Proceed to DECISION LOGIC.

- GENERAL: Route here when the user asks for definitions, program rules, lists, high-level processes, or generic strategies **not tied to their own case** and with **no case variables** present.
  → Produce ANSWER using only CONTEXT. Do not invite personalization.

- If truly unclear after applying the above, ask one short clarifying question, then route accordingly.

DECISION LOGIC (PERSONAL ONLY)

1. Detect the topic from the user message (e.g., eligibility decision, payout/transaction calc, workshops/engagements prerequisites, **optimization/strategy to maximize incentives**)
2. From CONTEXT, silently determine which ELIGIBILITY SIGNALS are required for this topic.
3. Auto-populate any ES you can **directly infer from CONTEXT** and the user’s message.
4. Compute `missing_signals = required_ES − present_ES`.
5. If `missing_signals` is NON-EMPTY:
    - Recheck CONTEXT to minimize asks, **and request only** human-only signals that cannot be inferred.
    - Ask concise, natural-language questions for the smallest necessary set (most-critical first). **Do not expose variable names or lists of signals.**
    - Do **not** list incentive names, rates, amounts, or claim steps.
    - Return FOLLOW-UP.
6. If `missing_signals` is EMPTY:
    - Use only CONTEXT to produce a structured, personalized determination.
    - If a calculation is requested, perform it using parameters from CONTEXT and the present ES.
    - If CONTEXT still lacks facts to be definitive, state exactly what evidence is missing and ask for that.
    - Return ANSWER.

OUTPUT SHAPES

- FOLLOW-UP (when `missing_signals` ≠ ∅):
    - A short lead-in sentence, then list of **targeted questions** (bullets allowed). **Do not enumerate “missing signals.”**
- ANSWER (when `missing_signals` = ∅):
    - A short conclusion sentence.
    - Then a structured rationale (bullets or a table) tied to the **information provided by the user** and the CONTEXT.

STREAMING PROTOCOL — MUST FOLLOW

- Stream only the user-visible conversational text (GFM) token-by-token.
- Do **not** output any JSON in the visible stream.
- After finishing the visible text, output **one** compact JSON trailer on a new line, prefixed **exactly** by: @@FINAL@@
- The JSON must have exactly these keys: "type", "text".
    - "type" ∈ ["answer","follow_up"]
    - "text": the **exact** visible text you just streamed (must not start with a heading)
- Do **not** wrap the JSON in code fences or add any extra characters.
- Output the @@FINAL@@ trailer exactly once, at the very end."""
    
    user = (
        "DIALOGUE SO FAR:\n"
        f"{dialogue}\n\n"
        "USER QUESTION IS THE LAST USER MESSAGE ABOVE.\n\n"
        "CONTEXT (authoritative; do not use outside knowledge):\n"
        f"{context_block}\n"
    )

    if session_block:
        user += (
            "\n\nSESSION DATA INSTRUCTIONS:\n"
            "Below is the user's profile and calculator inputs. Use this information to:\n"
            "1. Personalize responses based on their partner type, designations, and specializations\n"
            "2. Pre-populate calculations using their provided values (billed_revenue, global_tier1, etc.)\n"
            "3. Reference their specific business context when relevant\n"
            "4. Skip asking for information they've already provided\n"
            "5. Use their actual values for incentive calculations instead of asking for them\n"
            "6. **IMPORTANT**: When calculating incentives, use their actual values from the session data\n"
            "7. **IMPORTANT**: Don't ask for information they've already provided in their profile\n\n"
            "SESSION SNAPSHOT:\n"
            f"{session_block}\n"
        )
    print("---- Prompt to LLM ----",user)
    try:
        llm = _o3_stream_client(max_output_tokens=max_output_tokens)
        # Iterate true token stream
        accumulated: str = ""
        sent_up_to: int = 0

        in_final = False
        final_json_buf: List[str] = []
        final_started = False
        brace_depth = 0
        final_result: Optional[Dict[str, Any]] = None
        for chunk in llm.stream([SystemMessage(content=system), HumanMessage(content=user)]):
            # Extract text from chunk (robust to various content shapes)
            piece = getattr(chunk, "content", None)
            if piece is None:
                # Some LangChain versions nest text differently (delta dict)
                # Try best-effort stringify
                piece = str(chunk)
            if isinstance(piece, list):
                # Merge "text" fields
                tmp = []
                for part in piece:
                    if isinstance(part, dict) and "text" in part and part["text"]:
                        tmp.append(part["text"])
                piece = "".join(tmp)
            elif not isinstance(piece, str):
                piece = str(piece)

            if not piece:
                continue

            accumulated += piece

            # Check for FINAL sentinel (first time only)
            if not in_final:
                idx = accumulated.find(FINAL_PREFIX)
                if idx != -1:
                    in_final = True
                    # Emit any remaining visible text before the sentinel
                    visible = accumulated[:idx]
                    if len(visible) > sent_up_to:
                        delta_out = visible[sent_up_to:]
                        if delta_out:
                            yield {"event": "delta", "text": delta_out}
                        sent_up_to = len(visible)

                    # Everything after the sentinel belongs to the JSON trailer
                    trailer_part = accumulated[idx + len(FINAL_PREFIX) :]
                    # Start collecting compact JSON across subsequent chunks
                    for ch in trailer_part:
                        if ch == "{":
                            final_started = True
                            brace_depth = 1
                            final_json_buf.append(ch)
                        elif final_started:
                            final_json_buf.append(ch)
                            if ch == "{":
                                brace_depth += 1
                            elif ch == "}":
                                brace_depth -= 1
                                if brace_depth == 0:
                                    # We have a complete JSON trailer
                                    raw_json = "".join(final_json_buf)
                                    parsed = _safe_json(raw_json)
                                    final_result = _finalize_envelope(parsed, visible)
                                    break
                    # If we already got final JSON in same chunk, we're done streaming
                    if final_result is not None:
                        break
                    # Otherwise continue to collect JSON in following chunks
                    continue

                # No sentinel yet → stream visible delta
                if len(accumulated) > sent_up_to:
                    delta_out = accumulated[sent_up_to:]
                    if delta_out:
                        yield {"event": "delta", "text": delta_out}
                    sent_up_to = len(accumulated)

            else:
                # Already inside the final trailer; collect JSON across chunks
                for ch in piece:
                    if not final_started:
                        if ch == "{":
                            final_started = True
                            brace_depth = 1
                            final_json_buf.append(ch)
                        # ignore everything until first '{'
                        continue
                    final_json_buf.append(ch)
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            raw_json = "".join(final_json_buf)
                            parsed = _safe_json(raw_json)
                            final_result = _finalize_envelope(parsed, accumulated[: accumulated.find(FINAL_PREFIX)])
                            break
                if final_result is not None:
                    break

        # Stream ended (model finished)
        if final_result is None:
            # No trailer received: synthesize a normal answer with what we showed
            visible_text = accumulated.split(FINAL_PREFIX, 1)[0] if FINAL_PREFIX in accumulated else accumulated
            final_result = {"type": "answer", "text": visible_text, "missing_fields": []}

        yield {"event": "final", "result": final_result}

    except HTTPException:
        # Pass-through HTTP exceptions (router will handle)
        raise
    except Exception as e:
        # Handle connection errors gracefully
        error_msg = str(e) or "LLM error"
        
        # Check for connection-related errors
        if any(keyword in error_msg.lower() for keyword in ["peer closed", "connection", "incomplete", "chunked"]):
            # Connection was closed by client - this is not a server error
            print(f"Client connection closed: {error_msg}")
            # Don't raise an exception, just return gracefully
            return
        
        # For other errors, raise appropriate HTTP exception
        status_code = 504 if "timed out" in error_msg.lower() else 500
        raise HTTPException(status_code=status_code, detail=error_msg)


def _finalize_envelope(parsed: Any, visible_text: str) -> Dict[str, Any]:
    """
    Normalize/validate the final envelope coming from @@FINAL@@ JSON.
    Ensure required keys and fallbacks. Force 'text' to equal the visible content.
    """
    # Expected keys: type, missing_fields, text
    obj = parsed if isinstance(parsed, dict) else {}

    t = obj.get("type")
    if t not in ("answer", "follow_up"):
        t = "answer"

    # Missing fields array
    mf = obj.get("missing_fields")
    if not isinstance(mf, list):
        mf = []

    # Prefer the model's 'text' but ensure it matches what the user saw; fallback to visible_text
    txt = obj.get("text")
    if not isinstance(txt, str) or not txt.strip():
        txt = visible_text
    # Hard guarantee: final text equals visible content
    if txt != visible_text:
        txt = visible_text

    # Normalize the shape you router expects
    return {"type": t, "text": txt, "missing_fields": mf}
