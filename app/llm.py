#!/usr/bin/env python3
import json
from typing import List, Dict, Any
from langchain_openai import  ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
from .config import settings


LLM_MODEL = settings.LLM_MODEL 

# Hard defaults (tweak here if ever needed)
DEFAULT_CTX_N = 5
DEFAULT_CTX_FULL = True         
DEFAULT_CTX_MAX_CHARS = 40000  


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


