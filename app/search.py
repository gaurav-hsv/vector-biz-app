# app/search.py
from typing import List, Dict, Any, Iterable, Tuple, Optional
from psycopg.rows import dict_row
import re
from .db import get_pool

def _dedupe_keep_best(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep the highest-scoring row per logical key.
    For CSV chunks we key by (engagement_name, field).
    For DOCX chunks (no engagement_name/field), we key by (doc_name, section_path).
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in rows:
        engagement = (r.get("engagement_name") or "").strip()
        field      = (r.get("field") or "").strip()
        doc_name   = (r.get("doc_name") or "").strip()
        section    = (r.get("section_path") or (r.get("title") or "")).strip()

        if engagement or field:
            key = (engagement, field or r.get("column_name") or "")
        else:
            key = (doc_name, section)

        score = float(r.get("score", 0.0))
        keep  = best.get(key)
        if keep is None or score > keep["score"]:
            best[key] = {
                "id": r.get("id"),
                "title": (r.get("title") or "").strip(),
                "content": r.get("content") or "",
                "score": score,
                "field": field or None,
                "engagement_name": engagement or None,
                "column_name": r.get("column_name"),
                "doc_name": doc_name or None,
                "kind": r.get("kind"),  # 'table' | 'doc'
                "section_path": r.get("section_path"),
                "heading_level": r.get("heading_level"),
                "span_index": r.get("span_index"),
            }

    return list(best.values())
def _stitch_groups(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group siblings by (engagement, field, group_id) when available,
    stitch FULL parts in order, fallback to best content if no FULL present.
    """
    def norm(s): return (s or "").strip()

    groups: dict[tuple, list] = {}
    for r in rows:
        eng   = norm(r.get("engagement_name"))
        fld   = norm(r.get("field")) or norm(r.get("column_name"))
        doc   = norm(r.get("doc_name"))
        sect  = norm(r.get("section_path") or r.get("title"))
        gid   = norm(r.get("group_id"))
        # prefer row-level grouping; fallback to doc/section
        if eng or fld:
            gkey = (eng, fld, gid or f"{eng}::{fld}")
        else:
            gkey = (doc, sect, gid or f"{doc}::{sect}")
        r["score"] = float(r.get("score", 0.0))
        groups.setdefault(gkey, []).append(r)

    stitched: List[Dict[str, Any]] = []
    for _, sibs in groups.items():
        # pick best row for metadata & ranking
        best = max(sibs, key=lambda x: x["score"])
        # collect FULL parts
        full_parts = [s for s in sibs if (s.get("variant") == "full")]
        full_parts.sort(key=lambda x: (x.get("part_index") or 0))
        if full_parts:
            content = "\n".join(p.get("content") or "" for p in full_parts if p.get("content"))
            variant = "full"
        else:
            content = best.get("content") or ""
            variant = best.get("variant") or None

        stitched.append({
            "id": best.get("id"),
            "title": (best.get("title") or "").strip(),
            "content": content,
            "score": best["score"],
            "field": best.get("field"),
            "engagement_name": best.get("engagement_name"),
            "column_name": best.get("column_name"),
            "doc_name": best.get("doc_name"),
            "kind": best.get("kind"),
            "section_path": best.get("section_path"),
            "heading_level": best.get("heading_level"),
            "span_index": best.get("span_index"),
            "group_id": best.get("group_id"),
            "variant": variant,
        })

    stitched.sort(key=lambda x: x["score"], reverse=True)
    return stitched


def vector_search(
    query_vec: str,
    k: int = 30,
    *,
    prefer_kind: Optional[str] = None,  # 'table' | 'doc' | None
    bias: float = 0.07,                 # re-rank bonus for preferred kind
    user_query: Optional[str] = None,   # NEW: raw query text for smart bias
) -> List[Dict[str, Any]]:
    """
    PURE semantic search over incentive_chunks (cosine) with optional small
    re-ranking bias toward a source kind ('table' vs 'doc').

    Enhancements:
    - Engagement bias: if user query mentions an engagement, boost those hits.
    - Formula bias: if user query looks numeric (ACV/%, payout, rate), boost formula rows.
    """
    pre_limit = max(k * 20, 600)

    sql = """
   SELECT
    c.id, c.title, c.content, c.field, c.engagement_name, c.column_name,
    c.section_path, c.heading_level, c.span_index,
    d.name AS doc_name, d.kind AS kind,
    c.meta,
    (c.meta->>'variant')    AS variant,
    (c.meta->>'group_id')   AS group_id,
    NULLIF((c.meta->>'part_index'),'')::int   AS part_index,
    NULLIF((c.meta->>'total_parts'),'')::int  AS total_parts,
    1 - (c.embedding <=> %s::vector) AS score
FROM incentive_chunks c
JOIN incentive_documents d ON d.id = c.document_id
ORDER BY
    CASE (c.meta->>'variant')
      WHEN 'full' THEN 0
      WHEN 'overview' THEN 1
      ELSE 2
    END,
    c.embedding <=> %s::vector
LIMIT %s
    """

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SET LOCAL ivfflat.probes = 10;")
            except Exception:
                pass
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (query_vec, query_vec, pre_limit))
            rows = cur.fetchall() or []

    # Kind bias (unchanged)
    if prefer_kind in ("table", "doc"):
        for r in rows:
            if r.get("kind") == prefer_kind:
                r["score"] = float(r.get("score", 0.0)) * (1.0 + float(bias))

    # Engagement bias (NEW)
    if user_query:
        uq = user_query.lower()
        for r in rows:
            eng = (r.get("engagement_name") or "").lower()
            title = (r.get("title") or "").lower()
            if eng and eng in uq:
                r["score"] *= 1.15  # boost engagement match
            elif title and title in uq:
                r["score"] *= 1.10  # small boost

    # Formula bias (NEW)
    if user_query and re.search(r"\b(acv|arr|revenue|payout|earning|%|percent|rate|hours|seats)\b", user_query.lower()):
        for r in rows:
            field = (r.get("field") or "").lower()
            title = (r.get("title") or "").lower()
            content = (r.get("content") or "").lower()
            if "formula" in field or "formula" in title or "formula" in content:
                r["score"] *= 1.20  # strong boost for formula passages

    deduped = _stitch_groups(rows)
    return sorted(deduped, key=lambda x: x["score"], reverse=True)[:k]
