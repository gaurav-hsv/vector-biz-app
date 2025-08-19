# app/search.py
from typing import List, Dict, Any, Optional, Tuple
from psycopg.rows import dict_row
from .db import get_pool

def vector_search(
    query_vec: str,
    k: int = 30,
    *,
    field: Optional[str] = None,             # e.g. "Step_1", "Formula", or canonical col
    engagement_like: Optional[str] = None,   # e.g. "D365 CSP Core" (used in ILIKE)
    column_name: Optional[str] = None,       # original CSV label, e.g. "Step 1"
    doc_id: Optional[str] = None,            # if you need to scope to one document
    prefilter_kw: Optional[str] = None,      # extra text prefilter (ILIKE)
    use_prefilter: bool = True,
) -> List[Dict[str, Any]]:
    """
    Hybrid vector search over incentive_chunks with optional filters.
    - Filters: field, engagement_like, column_name, doc_id, prefilter_kw
    - Uses cosine distance on pgvector (1 - dist = score)
    - Dedupe by (engagement_name, field) to keep one best chunk per field for an engagement.

    Returns: [{id, title, content, field, engagement_name, column_name, score}]
    """
    # --- Build WHERE filters safely ---
    wheres = []
    params: List[Any] = []

    if doc_id:
        wheres.append("document_id = %s")
        params.append(doc_id)

    # Prefer canonical field filter (fast, surgical)
    if field:
        wheres.append("field = %s")
        params.append(field)

    # Allow original column label as alternative or extra filter
    if column_name:
        wheres.append("column_name = %s")
        params.append(column_name)

    # Engagement name prefilter (ILIKE to be robust to spacing/case)
    if engagement_like:
        wheres.append("engagement_name ILIKE %s")
        params.append(f"%{engagement_like}%")

    # Optional free-text prefilter (shrinks candidate set before vector ordering)
    if prefilter_kw:
        wheres.append("(content ILIKE %s OR title ILIKE %s)")
        params.extend([f"%{prefilter_kw}%", f"%{prefilter_kw}%"])

    where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    base_select_cols = "id, title, content, field, engagement_name, column_name, embedding"

    # --- Two-stage (recommended) or single-stage (fallback) plan ---
    sql: str
    if use_prefilter and where_sql:
        # Stage 1: prefilter by WHERE (and optional LIMIT to keep it cheap)
        # Stage 2: vector order + score compute
        sql = f"""
        WITH pre AS (
            SELECT {base_select_cols}
            FROM incentive_chunks
            {where_sql}
            LIMIT %s
        )
        SELECT id, title, content, field, engagement_name, column_name,
               1 - (embedding <=> %s::vector) AS score
        FROM pre
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        # params: filters..., pre_limit, query_vec (twice), final limit
        # keep a fairly generous pre-limit to avoid missing good hits
        final_params = params + [max(k * 8, 200), query_vec, query_vec, max(k, 30)]
    else:
        # Direct vector order without prefilter
        sql = f"""
        SELECT id, title, content, field, engagement_name, column_name,
               1 - (embedding <=> %s::vector) AS score
        FROM incentive_chunks
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        final_params = [query_vec] + params + [query_vec, max(k, 30)] if where_sql else [query_vec, query_vec, max(k, 30)]

        # If WHERE exists, the positions differ; normalize:
        if where_sql:
            # need to place query_vec after filters for the first %s::vector and before ORDER BY
            # Rebuild carefully
            # Build: [filters..., query_vec, query_vec, limit]
            final_params = params + [query_vec, query_vec, max(k, 30)]

    pool = get_pool()
    with pool.connection() as conn:
        # Optional: set ivfflat probes for better recall/accuracy
        # conn.execute("SET ivfflat.probes = 10;")  # uncomment if using ivfflat
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, final_params)
            rows = cur.fetchall() or []

    # --- Dedupe by (engagement_name, field) while keeping highest score ---
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        key = ((r.get("engagement_name") or "").strip(), (r.get("field") or "").strip())
        prev = best.get(key)
        if not prev or float(r.get("score", 0.0)) > float(prev.get("score", 0.0)):
            best[key] = {
                "id": r.get("id"),
                "title": (r.get("title") or "").strip(),
                "content": r.get("content") or "",
                "field": r.get("field"),
                "engagement_name": r.get("engagement_name"),
                "column_name": r.get("column_name"),
                "score": float(r.get("score", 0.0)),
            }

    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:k]
