from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from backend.services.report_service import (
    get_report_by_id,
    list_reports,
    delete_report,
)

router = APIRouter()


@router.get("/report/list", summary="List all generated reports")
async def list_all_reports():
    reports = list_reports()
    return {"total": len(reports), "reports": reports}


@router.get(
    "/report/{report_id}",
    response_class=PlainTextResponse,
    summary="Retrieve a report by ID",
)
async def get_report(report_id: str):
    """Returns the raw markdown content of the report."""
    content = get_report_by_id(report_id)
    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found.",
        )
    return content


@router.delete("/report/{report_id}", summary="Delete a report by ID")
async def remove_report(report_id: str):
    deleted = delete_report(report_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found.",
        )
    return {"deleted": True, "report_id": report_id}