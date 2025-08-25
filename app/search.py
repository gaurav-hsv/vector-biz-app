import psycopg2
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from .config import settings
import re
import json

# ---- Config ----
PG_DSN = settings.PG_DSN
EMBED_MODEL = settings.EMBED_MODEL
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # override with env if you prefer a different model

emb = OpenAIEmbeddings(model=EMBED_MODEL)

# Hard defaults (tweak here if ever needed)
DEFAULT_TOP_K = 15
DEFAULT_VEC_LIMIT = 90
DEFAULT_FTS_LIMIT = 90
DEFAULT_CTX_N = 8

def _vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

# map user phrasing -> metadata key
_DISTINCT_KEY_PATTERNS = [
    (re.compile(r"\b(incentive\s*types?|types?\s*of\s*incentives?)\b", re.I), "incentive_type"),
    (re.compile(r"\b(engagement\s*types?|types?\s*of\s*engagements?)\b", re.I), "engagement_type"),
]

# normalize common variants
_NORMALIZE_VALUE = {
    "pre sales": "Pre-sales",
    "presales": "Pre-sales",
    "pre-sales": "Pre-sales",
    "csp transaction": "CSP Incentive (Transaction)",
    "csp incentive (transaction)": "CSP Incentive (Transaction)",
}

def _detect_distinct_key(q: str) -> str | None:
    for pat, key in _DISTINCT_KEY_PATTERNS:
        if pat.search(q):
            return key
    return None

def _norm_val(s: str) -> str:
    v = (s or "").strip()
    low = v.lower()
    return _NORMALIZE_VALUE.get(low, v)

def _synthesize_sources_for_distinct(title: str, distinct_key: str, raw_values: List[Any]) -> List[Dict[str, Any]]:
    # normalize + stable-unique
    values: List[str] = []
    seen = set()
    for v in raw_values or []:
        if isinstance(v, str):
            s = _norm_val(v.strip())
        elif isinstance(v, dict):
            s = _norm_val((v.get("value") or v.get("name") or v.get("label") or json.dumps(v, ensure_ascii=False)).strip())
        else:
            s = _norm_val(str(v).strip())
        if s and s not in seen:
            seen.add(s)
            values.append(s)

    docs: List[Dict[str, Any]] = []
    # aggregate doc so LLM can see the whole menu in one place
    docs.append({
        "id": f"facet::{distinct_key}::summary",
        "content": f"{title}: " + ", ".join(values),
        "metadata": {
            "doc_type": "facet_summary",
            "title": title,
            "distinct_key": distinct_key,
            "values": values,
        },
        "pretty_source": {"source": "facet", "file": "", "location": ""},
    })

    # one doc per value (helps picking a single option)
    singular_title = title[:-1] if title.endswith("s") else title
    for v in values:
        docs.append({
            "id": f"facet::{distinct_key}::{v}",
            "content": f"{singular_title} option: {v}",
            "metadata": {
                "doc_type": "facet_option",
                "title": title,
                "distinct_key": distinct_key,
                "value": v,
                "name": v,  # hits your _TOPIC_KEYS ('name') for topic derivation
            },
            "pretty_source": {"source": "facet", "file": "", "location": ""},
        })

    return docs

def _fetch_distinct_values(conn, meta_key: str) -> list[str]:
    # Pull DISTINCT (case-insensitive), drop empties, normalize, then unique again
    sql = """
      SELECT DISTINCT NULLIF(TRIM(metadata->>%s), '') AS val
      FROM rag_chunks
      WHERE metadata ? %s  -- key exists
    """
    with conn.cursor() as cur:
        cur.execute(sql, (meta_key, meta_key))
        raw = [r[0] for r in cur.fetchall() if r[0]]
    normed = [_norm_val(v) for v in raw]
    # stable unique (preserve first-seen order)
    seen, out = set(), []
    for v in normed:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

def _vector_search(conn, qvec_lit: str, where_sql: str, params: list, limit: int):
    sql = f"""
      SELECT id, content, metadata,
             1 - (embedding <=> %s::vector) AS sim
      FROM rag_chunks
      WHERE {where_sql}
      ORDER BY embedding <=> %s::vector
      LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, [qvec_lit, *params, qvec_lit, limit])
        return cur.fetchall()  # (id, content, metadata, sim)

def _fts_search(conn, qtext: str, where_sql: str, params: list, limit: int):
    sql = f"""
      SELECT id, content, metadata,
             ts_rank_cd(fts, websearch_to_tsquery('english', %s)) AS lex
      FROM rag_chunks
      WHERE {where_sql}
        AND fts @@ websearch_to_tsquery('english', %s)
      ORDER BY lex DESC
      LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, [*params, qtext, qtext, limit])
        return cur.fetchall()  # (id, content, metadata, lex)

def _rrf_fuse(vec_rows, fts_rows, top_k: int, K: int = 60):
    rank_v = {r[0]: i+1 for i, r in enumerate(vec_rows)}
    rank_f = {r[0]: i+1 for i, r in enumerate(fts_rows)}
    ids = list({*rank_v.keys(), *rank_f.keys()})
    scored = []
    for _id in ids:
        rv = rank_v.get(_id)
        rf = rank_f.get(_id)
        score = (1.0/(K+rv) if rv else 0.0) + (1.0/(K+rf) if rf else 0.0)
        scored.append((_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    order = [_id for _id, _ in scored[:top_k]]

    by_id: Dict[int, Dict[str, Any]] = {}
    for r in vec_rows:
        by_id[r[0]] = {"id": r[0], "content": r[1], "metadata": r[2], "sim": float(r[3])}
    for r in fts_rows:
        by_id.setdefault(r[0], {"id": r[0], "content": r[1], "metadata": r[2]}).update({"lex": float(r[3])})
    return [by_id[_id] for _id in order]

def parse_kv(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"Bad --filter item '{it}', use key=value")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out

# -------------------------------------------------------
def vector_search(query: str,
                  top_k: int = DEFAULT_TOP_K,
                  vec_limit: int = DEFAULT_VEC_LIMIT,
                  fts_limit: int = DEFAULT_FTS_LIMIT) -> Dict[str, Any]:
    """
    Runs the same logic as `main()` but returns structured data for an API.
    """
    dsn = PG_DSN  # or os.getenv("PG_DSN", PG_DSN)

    # SPECIAL CASE: "what types ..." -> DISTINCT values
    distinct_key = _detect_distinct_key(query)
    if distinct_key:
        with psycopg2.connect(dsn) as conn:
            values = _fetch_distinct_values(conn, distinct_key)
        if values:
            title = "Incentive types" if distinct_key == "incentive_type" else "Engagement types"
            synth_sources = _synthesize_sources_for_distinct(title, distinct_key, values)
            return {
                "mode": "distinct",
                "distinct_key": distinct_key,
                "title": title,
                "values": values,
                "query": query,
                "top_k": 0,
                "returned": len(synth_sources),
                "sources": synth_sources
            }
        # fall through if nothing found

    # Default: hybrid retrieval
    qvec = emb.embed_query(query)
    qvec_lit = _vector_literal(qvec)
    where_sql, params = "TRUE", []

    with psycopg2.connect(dsn) as conn:
        vrows = _vector_search(conn, qvec_lit, where_sql, params, vec_limit)
        frows = _fts_search(conn, query, where_sql, params, fts_limit)
        fused = _rrf_fuse(vrows, frows, top_k=top_k)

    # Return compact, API-friendly payload
    sources = []
    for r in fused[:DEFAULT_CTX_N]:
        meta = r["metadata"] or {}
        src  = meta.get("_source") or ""
        file = meta.get("file") or ""
        loc  = f"row {meta.get('row')}" if src == "excel" else (f"p.{meta.get('page')}" if meta.get('page') else "")
        sources.append({
            "id": r["id"],
            "content": r["content"],
            "metadata": meta,
            "sim": r.get("sim"),
            "lex": r.get("lex"),
            "pretty_source": {"source": src, "file": file, "location": loc}
        })

    return {
        "mode": "hybrid",
        "query": query,
        "top_k": top_k,
        "returned": len(sources),
        "sources": sources,
    }
