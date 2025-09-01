# app/routes/message.py
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Iterator, List
from decimal import Decimal
import json
from ..search import vector_search
from ..llm import stream_answer 
from ..sessions_redis import get_session, append_message, update_session_key


router = APIRouter()

class MessageIn(BaseModel):
    session_id: Optional[str] = None
    text: str = Field(min_length=1)

class CSPCore(BaseModel):
    billed_revenue: Optional[Decimal]
    calculated_value: Optional[Decimal]

class CSPProductTier1(BaseModel):
    billed_from_tier_1: Optional[Decimal]
    calculated_value: Optional[Decimal]

class CSPProductTier2(BaseModel):
    billed_from_tier_2: Optional[Decimal]
    calculated_value: Optional[Decimal]

class CSPGrowthAccelerator(BaseModel):
    incremental_value_from_last_year: Optional[Decimal]
    calculated_value: Optional[Decimal]

class ProjectDetails(BaseModel):
    csp_core: Optional[CSPCore]
    product_accelerator_tier_1: Optional[CSPProductTier1]
    product_accelerator_tier_2: Optional[CSPProductTier2]
    growth_accelerator: Optional[CSPGrowthAccelerator]

class EngagementIncentive(BaseModel):
    name: str
    no_of_hours: Optional[Decimal]
    calculated_value: Optional[Decimal]

class ServiceProfitability(BaseModel):
    has_service_profitability: bool
    value: Optional[Decimal]
    expected_profit_margin: Optional[Decimal]
    calculated_value: Optional[Decimal]

class ManagedServicesProfitability(BaseModel):
    has_managed_services_profitability: bool
    monthly_value: Optional[Decimal]
    expected_profit_margin: Optional[Decimal]
    calculated_value: Optional[Decimal]

class UserProfile(BaseModel):
    partner_name: str
    partner_type: Optional[str]
    has_solution_partner_designation: Optional[bool]
    solution_designations: Optional[List[str]]
    has_specialization: Optional[bool]
    specializations: Optional[List[str]]
    have_1_million_threshold: Optional[bool] = None
    have_25k_threshold: Optional[bool] = None
    # csp_incentives : Optional[ProjectDetails]
    # margin_percentage_from_ingram_micro: Optional[float]
    # engagement_incentives: Optional[List[EngagementIncentive]]
    # implementations_services_profitability: Optional[ServiceProfitability]
    # managed_services_profitability: Optional[ManagedServicesProfitability]

class ProfileUpsertIn(BaseModel):
    session_id: Optional[str] = None
    user_profile: UserProfile

class CalculationBreakdown(BaseModel):
    total: Optional[Decimal] = None
    rebate: Optional[Decimal] = None
    coop: Optional[Decimal] = None
    fee: Optional[Decimal] = None

class SessionKeyUpdateIn(BaseModel):
    session_id: str
    key: str
    value: Optional[Any] = None
    calculation_breakdown: CalculationBreakdown
    no_of_hours: Optional[Decimal] = None
    month: Optional[str] = None
    year: Optional[int] = None
    
  

def _ndjson(obj: Dict[str, Any]) -> bytes:
    """Compact NDJSON encoder."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@router.post("/message")
def post_message(inp: MessageIn, debug: bool = Query(False, description="return debug info")):
    """
    Strict streaming endpoint. Always returns application/x-ndjson with frames:
      start → delta* → final (or error)
    """
    session = get_session(inp.session_id)
    append_message(session["session_id"], "user", inp.text)
    session = get_session(session["session_id"])
    session_id = session["session_id"]

    search_results = vector_search(inp.text)
    fused: List[Dict[str, Any]] = search_results.get("sources") or []

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
        yield _ndjson({"event": "start", "session_id": session_id})

        full_text: str = ""
        final_payload: Optional[Dict[str, Any]] = None

        try:
            for ev in stream_answer(llm_messages, fused, session=session):
                et = ev.get("event")
                if et == "delta":
                    delta = ev.get("text", "")
                    if not isinstance(delta, str):
                        continue
                    full_text += delta
                    yield _ndjson({"event": "delta", "text": delta})

                elif et == "final":
                    final_payload = ev.get("result") or {}
                    break

                elif et == "error":
                    detail = ev.get("detail") or "Unknown error"
                    yield _ndjson({"event": "error", "detail": detail})
                    return

            if not final_payload:
                final_payload = {"type": "answer", "text": full_text, "missing_fields": []}

            if final_payload.get("type") == "follow_up":
                append_message(session_id, "assistant", final_payload.get("text", ""))
            else:
                append_message(session_id, "assistant", final_payload.get("text", ""))

            yield _ndjson({"event": "final", "result": final_payload})

        except HTTPException as he:
            yield _ndjson({"event": "error", "detail": he.detail})
            raise
        except Exception as e:
            detail = str(e) or "Internal server error"
            yield _ndjson({"event": "error", "detail": detail})
            raise HTTPException(status_code=500, detail=detail)

    headers = {
        "Content-Type": "application/x-ndjson",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(event_stream(), media_type="application/x-ndjson", headers=headers)

@router.put("/session/profile")
def put_session_profile(inp: ProfileUpsertIn):
    try:
        session = get_session(inp.session_id)
        session_id = session["session_id"]

        data = inp.user_profile.model_dump(exclude_none=True)

        # --- ensure has_specialization is always present ---
        if "has_specialization" not in data:
            specs = data.get("specializations") or []
            data["has_specialization"] = bool(specs)

        # drop any nulls again after defaulting
        filtered_data = {k: v for k, v in data.items() if v is not None}

        # save under user_profile (your codebase expectation)
        updated = update_session_key(session_id, "user_profile", filtered_data)

        return {"ok": True, "session_id": session_id, "session": updated}
    except Exception as e:
        print("Error in put_session_profile:", e)
        raise HTTPException(status_code=500, detail="Failed to update profile")

@router.put("/session/key")
def put_session_key(inp: SessionKeyUpdateIn):
    """
    Update a specific key-value pair in the session.
    Useful for real-time updates when users change metric values.
    """
    try:
        session = get_session(inp.session_id)
        session_id = session["session_id"]

        # Convert key to snake_case format
        key_name = inp.key.replace(" ", "_").replace("-", "_")
        
        # Create a single object with all the data
        key_data = {}
        
        # Add value if provided
        if inp.value is not None:
            key_data["value"] = inp.value
            
        # Add calculation breakdown if provided
        if inp.calculation_breakdown:
            breakdown_data = inp.calculation_breakdown.model_dump(exclude_none=True)
            key_data["calculation_breakdown"] = breakdown_data
            
        # Add optional fields if provided
        if inp.no_of_hours is not None:
            key_data["no_of_hours"] = inp.no_of_hours
        if inp.month is not None:
            key_data["month"] = inp.month
        if inp.year is not None:
            key_data["year"] = inp.year

        # Get existing user_profile data
        existing_profile = session.get("user_profile", {})
        
        # Update the specific key within user_profile
        existing_profile[key_name] = key_data
        
        # Store the updated user_profile
        updated = update_session_key(session_id, "user_profile", existing_profile)

        return {"ok": True, "session_id": session_id, "session": updated}
    except Exception as e:
        print("Error in put_session_key:", e)
        raise HTTPException(status_code=500, detail="Failed to update session key")
