# in app/llm.py
from typing import List, Dict, Any
from .nlu import extract_from_passage

def generate_clarify(
    missing: List[str],
    passages: List[Dict[str, Any]],
    need_engagement: bool = False,
    need_workload: bool = False,
    need_incentive_type: bool = False
) -> Dict[str, Any]:
    """
    Build a crisp clarify prompt. Prefer engagement name; else ask for workload/incentive_type.
    Provide smart options from top hits.
    """
    titles = [p.get("title","") for p in passages[:6] if p.get("title")]
    wl_opts, it_opts = [], []

    for p in passages[:8]:
        fields = extract_from_passage(p.get("content",""))
        if fields.get("workload"):
            wl_opts.append(fields["workload"])
        if fields.get("incentive_type"):
            it_opts.append(fields["incentive_type"])

    def uniq(seq):
        out = []
        for x in seq:
            x = (x or "").strip()
            if x and x not in out:
                out.append(x)
        return out[:5]

    titles = uniq(titles)
    wl_opts = uniq(wl_opts)
    it_opts = uniq(it_opts)

    if need_engagement:
        return {
            "type": "clarify",
            "message": "Which engagement are you asking about?",
            "options": titles if titles else None
        }

    if need_workload and need_incentive_type:
        return {
            "type": "clarify",
            "message": "Please specify both the workload and the incentive type.",
            "options": (
                [f"Workload: {wl}" for wl in wl_opts] +
                [f"Incentive type: {it}" for it in it_opts]
            ) if (wl_opts or it_opts) else None
        }

    if need_workload:
        return {
            "type": "clarify",
            "message": "Which workload are you referring to?",
            "options": wl_opts if wl_opts else None
        }

    if need_incentive_type:
        return {
            "type": "clarify",
            "message": "Which incentive type are you referring to?",
            "options": it_opts if it_opts else None
        }

    # fallback
    return {
        "type": "clarify",
        "message": "Please provide either the engagement name or both workload and incentive type.",
        "options": titles[:3] if titles else None
    }
