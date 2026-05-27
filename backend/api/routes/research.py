from fastapi import APIRouter, HTTPException
from shared.schemas import (
    ResearchPlanRequest,
    ResearchPlanResponse,
    ResearchRunRequest,
    ResearchRunResponse,
)
from backend.services.research_service import generate_research_plan, run_research

router = APIRouter()


@router.post("/research/plan", response_model=ResearchPlanResponse)
async def plan_research(request: ResearchPlanRequest):
    """Runs the Planner Agent only."""
    try:
        return await generate_research_plan(
            query=request.query,
            model=request.model,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/research/run", response_model=ResearchRunResponse)
async def run_research_pipeline(request: ResearchRunRequest):
    """
    Synchronous full pipeline run.
    Blocks until complete — use /research/run_async for long queries.
    """
    try:
        return await run_research(
            query=request.query,
            model=request.model,
            collection_name=request.collection_name,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")