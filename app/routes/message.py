# app/routes/message.py
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Iterator, List
import json

# Retrieval + LLM
from ..search import vector_search
from ..llm import stream_answer  # NEW: true streaming generator from o3

# Sessions (Redis-backed)
from ..sessions_redis import get_session, append_message

# ---------------------------
# Router
# ---------------------------
router = APIRouter()


class MessageIn(BaseModel):
    session_id: Optional[str] = None
    text: str = Field(min_length=1)


def _ndjson(obj: Dict[str, Any]) -> bytes:
    """Compact NDJSON encoder."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@router.post("/message")
def post_message(inp: MessageIn, debug: bool = Query(False, description="return debug info")):
    """
    Strict streaming endpoint. Always returns application/x-ndjson with frames:
      start → delta* → final (or error)
    """
    # Ensure session & log the user message
    session = get_session(inp.session_id)
    append_message(session["session_id"], "user", inp.text)
    session = get_session(session["session_id"])
    session_id = session["session_id"]

    # Retrieval
    search_results = vector_search(inp.text)
    fused: List[Dict[str, Any]] = search_results.get("sources") or []

    # Build short dialogue window
    history = session.get("messages", [])[-10:]
    llm_messages: List[Dict[str, str]] = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("text", "")
        if content:
            llm_messages.append({"role": role, "content": content})
    if not llm_messages or llm_messages[-1]["content"] != inp.text:
        llm_messages.append({"role": "user", "content": inp.text})

    def event_stream() -> Iterator[bytes]:
        # 1) START frame
        yield _ndjson({"event": "start", "session_id": session_id})

        # 2) TRUE MODEL STREAM
        full_text: str = ""
        final_payload: Optional[Dict[str, Any]] = None

        try:
            for ev in stream_answer(llm_messages, fused):
                et = ev.get("event")
                if et == "delta":
                    # Append text & stream to client
                    delta = ev.get("text", "")
                    if not isinstance(delta, str):
                        continue
                    full_text += delta
                    yield _ndjson({"event": "delta", "text": delta})

                elif et == "final":
                    # Single final payload with {type, text, missing_fields}
                    final_payload = ev.get("result") or {}
                    break

                elif et == "error":
                    # Forward model-layer error to client, then end
                    detail = ev.get("detail") or "Unknown error"
                    yield _ndjson({"event": "error", "detail": detail})
                    return

            # 3) Session write (exactly once) + FINAL frame
            if not final_payload:
                # Safety: if model didn't provide final envelope, synthesize one
                final_payload = {"type": "answer", "text": full_text, "missing_fields": []}

            # Persist assistant message once
            if final_payload.get("type") == "follow_up":
                # For follow_up, text is the question we show the user
                append_message(session_id, "assistant", final_payload.get("text", ""))
            else:
                # Normal answer
                append_message(session_id, "assistant", final_payload.get("text", ""))

            yield _ndjson({"event": "final", "result": final_payload})

        except HTTPException as he:
            # Propagate known HTTP exceptions (e.g., timeout)
            yield _ndjson({"event": "error", "detail": he.detail})
            raise
        except Exception as e:
            # Unknown failures → error frame then 500
            detail = str(e) or "Internal server error"
            yield _ndjson({"event": "error", "detail": detail})
            raise HTTPException(status_code=500, detail=detail)

    headers = {
        "Content-Type": "application/x-ndjson",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # important for Nginx to not buffer stream
    }

    return StreamingResponse(event_stream(), media_type="application/x-ndjson", headers=headers)
