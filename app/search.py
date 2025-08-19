# app/search.py
from typing import List, Dict, Any, Iterable, Tuple
from psycopg.rows import dict_row
from .db import get_pool

def _dedupe_keep_best(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Key by title (your titles are "Engagement · Column"), keep highest score
    best = {}
    for r in rows:
        title = (r.get("title") or "").strip()
        score = float(r.get("score", 0.0))
        if title not in best or score > best[title]["score"]:
            best[title] = {
                "id": r.get("id"),
                "title": title,
                "content": r.get("content") or "",
                "score": score,
                # return these too (helpful for exact cell extraction)
                "field": r.get("field"),
                "engagement_name": r.get("engagement_name"),
                "column_name": r.get("column_name"),
            }
    return list(best.values())

def vector_search(query_vec: str, k: int = 30) -> List[Dict[str, Any]]:
    """
    PURE semantic search over incentive_chunks using pgvector (cosine).
    - Larger candidate pool for better recall of tiny details.
    - Per-query ivfflat.probes bump for accuracy.
    - No filters, no text search.
    """
    sql = """
    SELECT id, title, content, field, engagement_name, column_name,
           1 - (embedding <=> %s::vector) AS score
    FROM incentive_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    pre_limit = max(k * 20, 600)  # pull a big pool; tune as needed

    pool = get_pool()
    with pool.connection() as conn:
        # improve ANN precision without changing global setting
        with conn.cursor() as cur:
            try:
                cur.execute("SET LOCAL ivfflat.probes = 10;")
            except Exception:
                # if not using ivfflat or older pgvector, ignore
                pass

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (query_vec, query_vec, pre_limit))
            rows = cur.fetchall() or []

    deduped = _dedupe_keep_best(rows)
    return sorted(deduped, key=lambda x: x["score"], reverse=True)[:k]
