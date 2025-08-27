import psycopg2
from typing import List, Dict, Any, Tuple, Optional
from langchain_openai import OpenAIEmbeddings
from .config import settings
import re
import json
from collections import Counter

# -----------------------------
# Config
# -----------------------------
PG_DSN = settings.PG_DSN
EMBED_MODEL = settings.EMBED_MODEL

emb = OpenAIEmbeddings(model=EMBED_MODEL)

DEFAULT_TOP_K = 30
DEFAULT_VEC_LIMIT = 90
DEFAULT_FTS_LIMIT = 90
DEFAULT_CTX_N = 15

# Keys used to infer a "family" across results
FAMILY_KEYS = [
    "solution_play",
    "incentive_type",
    "engagement_type",
    "workload",
]

# -----------------------------
# Normalization & Query Helpers
# -----------------------------
_GB_US = {
    r"\bmaximise\b": "maximize",
    r"\bprogramme\b": "program",
}

# keep generic, non-domain specific, but money/benefit friendly
_GENERIC_SYNONYMS = [
    "price pricing cost",
    "benefit incentive rebate coop bonus",
    "growth increase delta yoy year-over-year",
    "workshop assessment accelerator briefing",
    "crm erp sales service finance supply chain business central",
    "dynamics 365 d365 power platform"

]

_STOP = {
    "a","an","and","are","as","at","be","but","by","for","from","how","i","in","is",
    "it","my","of","on","or","that","the","this","to","we","what","when","where","with","your"
}

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-_/]{1,}", re.I)
_QUOTE_RE = re.compile(r"\"([^\"]{2,}?)\"|'([^']{2,}?)'")

_STOPJUNK = re.compile(r"[“”‘’•·–—]+")  # bullets/smart quotes/dashes → space


def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def _tokenize(q: str) -> Tuple[List[str], List[List[str]]]:
    """
    Returns (terms, phrases):
      - terms: lowercased tokens (no stopwords), hyphen/symbols normalized
      - phrases: list of token lists for quoted phrases (each >=2 tokens)
    """
    ql = q.strip()
    phrases: List[List[str]] = []
    # capture quoted phrases first
    for m in _QUOTE_RE.finditer(ql):
        phrase = (m.group(1) or m.group(2) or "").lower()
        toks = [t for t in _WORD_RE.findall(phrase) if t.lower() not in _STOP]
        if len(toks) >= 2:
            phrases.append(toks)

    # remove quotes for remaining term scan
    q_clean = _QUOTE_RE.sub(" ", ql).lower()
    terms = [t for t in _WORD_RE.findall(q_clean) if t not in _STOP]
    # expand hyphenated or slash terms a bit (e.g., business-apps -> business, apps)
    extra = []
    for t in terms:
        for sep in ("-", "/", "_"):
            if sep in t:
                parts = [p for p in t.split(sep) if p and p not in _STOP]
                if len(parts) >= 2:
                    extra.extend(parts)
    terms = _dedupe(terms + extra)
    return terms, phrases

def _escape_ts(token: str) -> str:
    # escape special chars for to_tsquery
    return re.sub(r"([&|!:()\[\]<>])", r"\\\1", token)

def _prefix_or_tsquery(terms: List[str]) -> str:
    """
    Build a permissive OR tsquery with prefix matching: foo:* | bar:* | baz:*
    Use this when you want high recall.
    """
    if not terms:
        return ""
    parts = [f"{_escape_ts(t)}:*" for t in terms]
    return " | ".join(parts)

def _phrase_tsquery(phrases: List[List[str]]) -> str:
    """
    Convert phrases into to_tsquery using <-> (phrase operator).
    Each phrase becomes "t1:* <-> t2:* <-> t3:*", phrases are OR'ed together.
    """
    if not phrases:
        return ""
    ph = []
    for toks in phrases:
        ph.append(" <-> ".join(f"{_escape_ts(t)}:*" for t in toks))
    return " | ".join(ph)

def _normalize_query(q: str) -> str:
    q = _STOPJUNK.sub(" ", (q or "").strip().lower())
    for pat, repl in _GB_US.items():
        q = re.sub(pat, repl, q)
    # collapse whitespace
    q = re.sub(r"\s+", " ", q)
    return q

def _expand_query_generic(q: str) -> str:
    base = _normalize_query(q)
    syn = " ".join(f'"{s}"' for s in _GENERIC_SYNONYMS)
    return f"{base} {syn}".strip()

def _broad_intent(q: str) -> bool:
    return bool(re.search(r"\b(maximi[sz]e|list|compare|all|catalog(ue)?|overview|guide|cheatsheet|catalogue)\b", (q or "").lower()))

def _vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

# -----------------------------
# "Distinct values" facet support
# -----------------------------
_DISTINCT_KEY_PATTERNS = [
    (re.compile(r"\b(incentive\s*types?|types?\s*of\s*incentives?)\b", re.I), "incentive_type"),
    (re.compile(r"\b(engagement\s*types?|types?\s*of\s*engagements?)\b", re.I), "engagement_type"),
]

_NORMALIZE_VALUE = {
    "pre sales": "Pre-sales",
    "presales": "Pre-sales",
    "pre-sales": "Pre-sales",
    "csp transaction": "CSP Incentive (Transaction)",
    "csp incentive (transaction)": "CSP Incentive (Transaction)",
}

def _detect_distinct_key(q: str) -> Optional[str]:
    for pat, key in _DISTINCT_KEY_PATTERNS:
        if pat.search(q or ""):
            return key
    return None

def _norm_val(s: str) -> str:
    v = (s or "").strip()
    low = v.lower()
    return _NORMALIZE_VALUE.get(low, v)

def _fetch_distinct_values(conn, meta_key: str) -> List[str]:
    sql = """
      SELECT DISTINCT NULLIF(TRIM(metadata->>%s), '') AS val
      FROM rag_chunks_large
      WHERE metadata ? %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (meta_key, meta_key))
        raw = [r[0] for r in cur.fetchall() if r[0]]
    normed = [_norm_val(v) for v in raw]
    seen, out = set(), []
    for v in normed:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

def _synthesize_sources_for_distinct(title: str, distinct_key: str, raw_values: List[Any]) -> List[Dict[str, Any]]:
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
            seen.add(s); values.append(s)

    docs: List[Dict[str, Any]] = []
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
                "name": v,
            },
            "pretty_source": {"source": "facet", "file": "", "location": ""},
        })
    return docs

# -----------------------------
# DB Search Primitives
# -----------------------------
def _vector_search(conn, qvec_lit: str, where_sql: str, params: list, limit: int):
    
    sql = f"""
      SELECT id, content, metadata,
             1 - ((embedding::halfvec(3072)) <=> (%s::vector)::halfvec) AS sim
      FROM rag_chunks_large
      WHERE {where_sql}
      ORDER BY (embedding::halfvec(3072)) <=> (%s::vector)::halfvec
      LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, [qvec_lit, *params, qvec_lit, limit])
        return cur.fetchall()

def _fts_search(conn, qtext: str, where_sql: str, params: list, limit: int):
    """
    Multi-strategy FTS:
      1) permissive prefix-OR (to_tsquery) + phrases
      2) plainto_tsquery
      3) websearch_to_tsquery
      4) (optional) pg_trgm similarity fallback on content if FTS empty

    Returns rows: (id, content, metadata, lex)
    """

    terms, phrases = _tokenize(qtext)
    tsq_prefix = _prefix_or_tsquery(terms)  # high recall
    tsq_phrase = _phrase_tsquery(phrases)   # quoted phrases, if any

    def _run(sql, args, label):
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    # 1) to_tsquery (prefix OR) possibly combined with phrases (OR)
    # Build a single tsquery text like: "(prefix_or) | (phrases)"
    tsq_parts = [p for p in [tsq_prefix, tsq_phrase] if p]
    tsq_combo = " | ".join(tsq_parts) if tsq_parts else ""
    sql1 = f"""
      SELECT id, content, metadata,
             ts_rank_cd(fts, q.tsq) AS lex
      FROM rag_chunks_large, (SELECT to_tsquery('english', %s) AS tsq) q
      WHERE {where_sql} AND (%s = '' OR fts @@ q.tsq)
      ORDER BY lex DESC
      LIMIT %s
    """
    rows = _run(sql1, [tsq_combo, tsq_combo, limit], "to_tsquery(prefix|phrase)")
    if rows:
        return rows

    # 2) plainto_tsquery (for natural language without operators)
    sql2 = f"""
      SELECT id, content, metadata,
             ts_rank_cd(fts, q.tsq) AS lex
      FROM rag_chunks_large, (SELECT plainto_tsquery('english', %s) AS tsq) q
      WHERE {where_sql} AND fts @@ q.tsq
      ORDER BY lex DESC
      LIMIT %s
    """
    rows = _run(sql2, [qtext, limit], "plainto_tsquery")
    if rows:
        return rows

    # 3) websearch_to_tsquery (good at handling quotes/AND/OR like Google)
    sql3 = f"""
      SELECT id, content, metadata,
             ts_rank_cd(fts, q.tsq) AS lex
      FROM rag_chunks_large, (SELECT websearch_to_tsquery('english', %s) AS tsq) q
      WHERE {where_sql} AND fts @@ q.tsq
      ORDER BY lex DESC
      LIMIT %s
    """
    rows = _run(sql3, [qtext, limit], "websearch_to_tsquery")
    if rows:
        return rows

    # 4) OPTIONAL: pg_trgm similarity fallback if available (broad recall)
    # This helps when the tsquery becomes too strict or vocabulary mismatches.
    # Safe to try; if extension/index not present it will still work (just slower).
    try:
        sql4 = f"""
          SELECT id, content, metadata,
                 /* fake 'lex' using similarity so caller can sort/compare */
                 GREATEST(similarity(content, %s), 0.0001) AS lex
          FROM rag_chunks_large
          WHERE {where_sql} AND content %% %s
          ORDER BY lex DESC
          LIMIT %s
        """
        rows = _run(sql4, [qtext, qtext, limit], "pg_trgm_similarity")
        if rows:
            return rows
    except Exception as e:
        print(f"[search] trigram fallback skipped: {e!r}")

    # Nothing found
    return []
# -----------------------------
# Family Discovery & Sibling Recall (generic)
# -----------------------------
def _shared_family_filters(rows: List[Tuple], max_scan: int = 20) -> Dict[str, str]:
    """
    rows: rows from vector search [(id, content, metadata, ...), ...]
    Returns stable (key -> value) present in >= 60% of top rows
    """
    N = min(max_scan, len(rows))
    counters: Dict[str, Counter] = {k: Counter() for k in FAMILY_KEYS}
    for r in rows[:N]:
        meta = r[2] or {}
        for k in FAMILY_KEYS:
            v = (meta.get(k) or "").strip()
            if v:
                counters[k][v] += 1
    filters: Dict[str, str] = {}
    for k, ctr in counters.items():
        if not ctr:
            continue
        val, cnt = ctr.most_common(1)[0]
        if cnt >= max(2, int(0.6 * N)):
            filters[k] = val
    return filters

def _fetch_siblings(conn, fam_filters: Dict[str, str], limit: int = 200):
    if not fam_filters:
        return []
    clauses, params = [], []
    for k, v in fam_filters.items():
        clauses.append("(metadata->>%s) = %s")
        params.extend([k, v])
    where = " AND ".join(clauses) if clauses else "TRUE"
    sql = f"""
      SELECT id, content, metadata, NULL::float8 AS sim, NULL::float8 AS lex
      FROM rag_chunks_large
      WHERE {where}
      LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, [*params, limit])
        return cur.fetchall()

# -----------------------------
# Fusion, Diversity & Packing
# -----------------------------
def _rrf_fuse(vec_rows, fts_rows, top_k: int, K_vec: int = 60, K_fts: int = 60):
    rank_v = {r[0]: i+1 for i, r in enumerate(vec_rows)}
    rank_f = {r[0]: i+1 for i, r in enumerate(fts_rows)}
    ids = list({*rank_v.keys(), *rank_f.keys()})
    scored = []
    for _id in ids:
        rv = rank_v.get(_id)
        rf = rank_f.get(_id)
        s = (1.0/(K_vec + rv) if rv else 0.0) + (1.0/(K_fts + rf) if rf else 0.0)
        scored.append((_id, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    order = [_id for _id, _ in scored[:top_k]]

    by_id: Dict[Any, Dict[str, Any]] = {}
    for r in vec_rows:
        by_id[r[0]] = {"id": r[0], "content": r[1], "metadata": r[2], "sim": float(r[3]) if r[3] is not None else None}
    for r in fts_rows:
        by_id.setdefault(r[0], {"id": r[0], "content": r[1], "metadata": r[2]}).update({"lex": float(r[3]) if r[3] is not None else None})
    return [by_id[_id] for _id in order]

def _diverse_pack(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """
    Ensure diversity across (solution_play, incentive_type, workload).
    """
    seen = set()
    picked: List[Dict[str, Any]] = []

    def sig(meta):
        return (meta.get("solution_play",""), meta.get("incentive_type",""), meta.get("workload",""))

    for r in rows:
        s = sig(r.get("metadata", {}))
        if s not in seen:
            seen.add(s); picked.append(r)
        if len(picked) >= n:
            return picked[:n]

    for r in rows:
        if len(picked) >= n:
            break
        if r not in picked:
            picked.append(r)

    return picked[:n]

# -----------------------------
# Public API
# -----------------------------
def vector_search(query: str,
                  top_k: int = DEFAULT_TOP_K,
                  vec_limit: int = DEFAULT_VEC_LIMIT,
                  fts_limit: int = DEFAULT_FTS_LIMIT) -> Dict[str, Any]:
    dsn = PG_DSN

    # DISTINCT facet mode
    distinct_key = _detect_distinct_key(query)
    if distinct_key:
        with psycopg2.connect(dsn) as conn:
            values = _fetch_distinct_values(conn, distinct_key)
        if values:
            title = "Incentive types" if distinct_key == "incentive_type" else "Engagement types"
            synth = _synthesize_sources_for_distinct(title, distinct_key, values)
            return {
                "mode": "distinct",
                "distinct_key": distinct_key,
                "title": title,
                "values": values,
                "query": query,
                "top_k": 0,
                "returned": len(synth),
                "sources": synth
            }
        # fallthrough if no values

    # Hybrid retrieval
    qvec = emb.embed_query(query)
    qvec_lit = _vector_literal(qvec)
    where_sql, params = "TRUE", []

    expanded_q = _expand_query_generic(query)
    ctx_n = DEFAULT_CTX_N + 5

    with psycopg2.connect(dsn) as conn:
        # Vector
        vrows = _vector_search(conn, qvec_lit, where_sql, params, vec_limit)

        # FTS (generic-expanded)
        frows = _fts_search(conn, expanded_q, where_sql, params, fts_limit)

        # Family discovery on vector hits, then sibling recall (generic)
        fam_filters = _shared_family_filters(vrows)
        if fam_filters:
            sibs = _fetch_siblings(conn, fam_filters)
            # Union-dedup by id into vector side (so fusion ranks all)
            by = {r[0]: r for r in vrows}
            for r in sibs:
                by.setdefault(r[0], r)
            vrows = list(by.values())

        # Adaptive RRF weights: if FTS hit something, give it a bit more weight
        K_vec = 60
        K_fts = 30 if frows else 60
        fused = _rrf_fuse(vrows, frows, top_k=top_k, K_vec=K_vec, K_fts=K_fts)

    # Final packing (diversity)
    final_rows = _diverse_pack(fused, n=ctx_n)

    # API payload
    sources = []
    for r in final_rows:
        meta = r.get("metadata") or {}
        src  = meta.get("_source") or ""
        file = meta.get("file") or ""
        loc  = f"row {meta.get('row')}" if src == "excel" else (f"p.{meta.get('page')}" if meta.get('page') else "")
        sources.append({
            "id": r.get("id"),
            "content": r.get("content"),
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
