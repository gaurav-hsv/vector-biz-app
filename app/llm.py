# app/llm.py
#!/usr/bin/env python3
import json
import re
from typing import Any, Dict, Iterator, List, Optional

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
            if isinstance(part, dict):
                t = part.get("text")
                if t:
                    out.append(t)
        return "\n".join(out).strip()
    return str(c).strip()


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
        timeout=110,
        max_retries=1,
        # No response_format in streaming mode; allow free-form streaming text.
        model_kwargs={"max_output_tokens": max_output_tokens},
    )


# ----------------------------
# Policy / Instructions
# ----------------------------
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
   - IN case of calculations make sure never ask user the market rate(for this only ask country if not provided), the cap, and the percentage(as it will always in context)
   - Return FOLLOW-UP asking **only** the missing_fields (max 5, most-critical first).
4) If missing_fields is EMPTY:
   - Use ONLY CONTEXT to produce a structured, personalized determination.
   - Return ANSWER.
5) - When user asking to CALCULATE the SPD/PARTNER ELIGIBILITY- Do calcultion using the context provided and if any field is missing ask for that.

STYLE & GUARDS
- All `question`/`answer` text must be valid GFM.
- **Do not start with a heading.** Begin with a direct sentence.
- No decorative ASCII, no horizontal rules.
- NO echoing the user’s question.
- Don’t mention training data, retrieval, or internal mechanics or row.
- Rely ONLY on CONTEXT. If facts are still missing after fields are complete, say so in GFM and request that specific evidence.
""".strip()

BASE_SYSTEM = """
### Role
- Primary Function: You are an AI chatbot who helps users with their inquiries, issues and requests. Provide professional, efficient replies. If a question is not clear, ask clarifying questions. End with a positive note.
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

STREAMING_ADDENDUM = f"""
### Streaming Protocol — MUST FOLLOW
- During generation, stream the **user-visible conversational text** token-by-token.
- You MUST NOT output any JSON in the visible stream.
- When the conversational text is completely finished, output **one single-line** compact JSON on its own line, prefixed by `{FINAL_PREFIX}`.
- That JSON MUST include exactly these keys: "type", "missing_fields", "text".
  - "type" ∈ ["answer","follow_up"]
  - "missing_fields" is an array (empty for "answer")
  - "text" MUST equal the EXACT final user-visible text you just streamed (no headings at start).
- Do **not** wrap the JSON in code fences or any extra characters.
- Ignore any earlier instruction suggesting to “return exactly one JSON”; the ONLY JSON must be this final trailer line after streaming the text.
- Output `{FINAL_PREFIX}` **exactly once** at the very end.
""".strip()


# ----------------------------
# Streaming Orchestrator
# ----------------------------
def stream_answer(
    llm_messages: List[Dict[str, str]],
    fused: List[Dict[str, Any]],
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

    system = (WRAPPER + "\n\n" + BASE_SYSTEM + "\n\n" + STREAMING_ADDENDUM).strip()
    user = (
        "DIALOGUE SO FAR:\n"
        f"{dialogue}\n\n"
        "USER QUESTION IS THE LAST USER MESSAGE ABOVE.\n\n"
        "CONTEXT (authoritative; do not use outside knowledge):\n"
        f"{context_block}\n"
    )

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
        # Upstream will send this as an error frame then raise 500/504
        raise HTTPException(status_code=504 if "timed out" in str(e).lower() else 500, detail=str(e) or "LLM error")


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
