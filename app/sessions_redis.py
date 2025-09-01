# app/sessions_redis.py
import json, time, uuid
from typing import Optional, Dict, Any, List, Union
import redis
from .config import settings
from decimal import Decimal
import re

_r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Redis key & constants
_TTL_SECONDS = 1800  # 30 min sliding session
_KEY = lambda sid: f"sess:{sid}"
_MAX_MESSAGES = 200  # hard cap to prevent growth


# ---------- JSON helpers (Decimal-safe) ----------
def _json_default(o):
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_json_default)



def _new_session(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "messages": [],
        "memory": {"summary": "", "state": {}},
    }

def get_session(session_id: Optional[str]) -> Dict[str, Any]:
    if not session_id:
        session_id = str(uuid.uuid4())
    key = _KEY(session_id)
    raw = _r.get(key)
    if not raw:
        session = _new_session(session_id)
        _r.setex(key, _TTL_SECONDS, json.dumps(session))
        return session
    return json.loads(raw)

def _trim_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(msgs) > _MAX_MESSAGES:
        # keep the newest window
        return msgs[-_MAX_MESSAGES:]
    return msgs

def append_message(session_id: str, role: str, text: str) -> None:
    key = _KEY(session_id)
    for _ in range(4):  # small retry budget
        with _r.pipeline() as p:
            try:
                p.watch(key)
                raw = p.get(key)
                if not raw:
                    session = _new_session(session_id)
                else:
                    session = json.loads(raw)

                session["messages"].append({
                    "role": role,
                    "text": text,
                    "ts": int(time.time())
                })
                session["messages"] = _trim_messages(session["messages"])

                p.multi()
                p.setex(key, _TTL_SECONDS, json.dumps(session))
                p.execute()
                return
            except redis.WatchError:
                # concurrent writer; retry
                continue
    raise RuntimeError("append_message failed due to concurrent updates")

_PATH_TOKEN_RE = re.compile(r"""
    ([^. \[\]]+)      # bare key
    |                 # or
    \[(\d+)\]         # [index]
""", re.X)

def _tokenize_path(path: str) -> List[Union[str, int]]:
    tokens: List[Union[str, int]] = []
    for m in _PATH_TOKEN_RE.finditer(path):
        key, idx = m.groups()
        if key is not None:
            tokens.append(key)
        else:
            tokens.append(int(idx))
    if not tokens:
        raise ValueError(f"Invalid path: {path!r}")
    return tokens

def _ensure_container(parent: Any, token: Union[str, int], next_token: Union[str, int]) -> Any:
    # Ensure container at parent[token] based on next token type
    if isinstance(token, str):
        if not isinstance(parent, dict):
            raise TypeError("Path traverses non-dict container")
        curr = parent.get(token)
        if curr is None:
            curr = [] if isinstance(next_token, int) else {}
            parent[token] = curr
        return curr
    else:
        # token is int → parent must be a list
        if not isinstance(parent, list):
            raise TypeError("Attempted list indexing on non-list container")
        while len(parent) <= token:
            parent.append(None)
        if parent[token] is None:
            parent[token] = [] if isinstance(next_token, int) else {}
        return parent[token]

def _set_by_path(root: Any, tokens: List[Union[str, int]], value: Any) -> None:
    if not tokens:
        raise ValueError("Empty path")
    curr = root
    for i, tk in enumerate(tokens):
        is_last = (i == len(tokens) - 1)
        if is_last:
            if isinstance(tk, str):
                if not isinstance(curr, dict):
                    raise TypeError("Cannot set key on non-dict container")
                curr[tk] = value
            else:
                if not isinstance(curr, list):
                    raise TypeError("Cannot set index on non-list container")
                while len(curr) <= tk:
                    curr.append(None)
                curr[tk] = value
        else:
            nxt = tokens[i + 1]
            if isinstance(tk, str):
                if not isinstance(curr, dict):
                    raise TypeError("Path traverses non-dict container")
                curr = _ensure_container(curr, tk, nxt)
            else:
                if not isinstance(curr, list):
                    raise TypeError("Path traverses non-list container")
                while len(curr) <= tk:
                    curr.append(None)
                if curr[tk] is None:
                    curr[tk] = [] if isinstance(nxt, int) else {}
                curr = curr[tk]

def update_session_key(session_id: str, key_path: str, value: Any) -> Dict[str, Any]:
    key = _KEY(session_id)
    tokens = _tokenize_path(key_path)

    for _ in range(4):
        with _r.pipeline() as p:
            try:
                p.watch(key)
                raw = p.get(key)
                session = json.loads(raw) if raw else _new_session(session_id)

                _set_by_path(session, tokens, value)

                p.multi()
                p.setex(key, _TTL_SECONDS, _dump(session))
                p.execute()
                return session
            except redis.WatchError:
                continue

    raise RuntimeError("update_session_key failed due to concurrent updates")