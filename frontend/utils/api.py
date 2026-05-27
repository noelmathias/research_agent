import httpx
import json
from typing import Any, Dict, Optional, List

API_BASE = "http://localhost:8000/api/v1"
DEFAULT_TIMEOUT = 600


def _get(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{API_BASE}{endpoint}", params=params)
        r.raise_for_status()
        return r.json()


def _post(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{API_BASE}{endpoint}", json=payload)
        r.raise_for_status()
        return r.json()


def _delete(endpoint: str) -> Dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        r = client.delete(f"{API_BASE}{endpoint}")
        r.raise_for_status()
        return r.json()


# ── Health ────────────────────────────────────────────────────────────────────

def get_health() -> Dict[str, Any]:
    return _get("/health")


def test_ollama(prompt: str, model: str) -> Dict[str, Any]:
    return _post("/test-ollama", {"prompt": prompt, "model": model})


# ── Research ──────────────────────────────────────────────────────────────────

def run_research_plan(query: str, model: str) -> Dict[str, Any]:
    return _post("/research/plan", {"query": query, "model": model})


def run_research_pipeline(
    query: str,
    model: str,
    collection_name: str,
) -> Dict[str, Any]:
    return _post(
        "/research/run",
        {"query": query, "model": model, "collection_name": collection_name},
        timeout=600,
    )


# ── Vector ────────────────────────────────────────────────────────────────────

def get_collection_count(collection_name: str) -> Dict[str, Any]:
    return _get("/vector/status")


def ingest_text(
    texts: List[str],
    metadatas: List[Dict],
    collection_name: str,
) -> Dict[str, Any]:
    return _post(
        "/vector/ingest/text",
        {
            "texts": texts,
            "metadatas": metadatas,
            "collection_name": collection_name,
        },
        timeout=60,
    )


def run_research_pipeline_async(
    query: str,
    model: str,
    collection_name: str,
) -> Dict[str, Any]:
    return _post(
        "/research/run_async",
        {"query": query, "model": model, "collection_name": collection_name},
        timeout=30,
    )


def get_research_job(job_id: str) -> Dict[str, Any]:
    return _get(f"/research/job/{job_id}")


def stream_research_job(job_id: str):
    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{API_BASE}/research/stream/{job_id}") as response:
            response.raise_for_status()
            event_name = "message"
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    payload = line.split(":", 1)[1].strip()
                    yield {
                        "event": event_name,
                        "data": json.loads(payload),
                    }


def search_vector(
    query: str,
    top_k: int,
    collection_name: str,
) -> Dict[str, Any]:
    return _post(
        "/vector/search",
        {
            "query": query,
            "top_k": top_k,
            "collection_name": collection_name,
        },
        timeout=30,
    )


# ── PDF ───────────────────────────────────────────────────────────────────────

def upload_pdf(
    file_bytes: bytes,
    filename: str,
    collection_name: str,
) -> Dict[str, Any]:
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f"{API_BASE}/pdf/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            params={"collection_name": collection_name},
        )
        r.raise_for_status()
        return r.json()


def list_pdfs() -> Dict[str, Any]:
    return _get("/pdf/list")


# ── MCP ───────────────────────────────────────────────────────────────────────

def list_mcp_tools() -> Dict[str, Any]:
    return _get("/mcp/tools")


def invoke_mcp_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tool_name": tool_name,
        "parameters": parameters,
    }
    if collection_name:
        payload["collection_name"] = collection_name
    return _post("/mcp/invoke", payload, timeout=60)


# ── Reports ───────────────────────────────────────────────────────────────────

def list_reports() -> Dict[str, Any]:
    return _get("/report/list")


def get_report(report_id: str) -> str:
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{API_BASE}/report/{report_id}")
        r.raise_for_status()
        return r.text


def delete_report(report_id: str) -> Dict[str, Any]:
    return _delete(f"/report/{report_id}")
