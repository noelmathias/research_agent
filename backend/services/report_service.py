from pathlib import Path
from typing import List, Optional, Dict, Any
from shared.config import get_settings
from datetime import datetime
settings = get_settings()


def get_report_by_id(report_id: str) -> Optional[str]:
    """
    Load a persisted report from disk by ID.
    Returns markdown string or None if not found.
    """
    path = Path(settings.reports_dir) / f"{report_id}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def list_reports() -> List[Dict[str, Any]]:
    """
    List all persisted reports in data/reports/.
    Returns metadata list sorted by modification time descending.
    """
    reports_dir = Path(settings.reports_dir)
    if not reports_dir.exists():
        return []

    reports = []
    for path in sorted(
        reports_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        stat = path.stat()
        reports.append(
            {
                "report_id": path.stem,
                "filename": path.name,
                "size_kb": round(stat.st_size / 1024, 2),
                "created_at": datetime.fromtimestamp(
                    stat.st_mtime
                    ).isoformat(),
            }
        )

    return reports


def delete_report(report_id: str) -> bool:
    """Delete a report by ID. Returns True if deleted, False if not found."""
    path = Path(settings.reports_dir) / f"{report_id}.md"
    if not path.exists():
        return False
    path.unlink()
    return True